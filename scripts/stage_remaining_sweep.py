"""Stage only future steps in private RAM storage, then persist verified bytes.

Never moves an existing/active step. Numerical capture code is unchanged.
Canonical paths are temporary symlinks until all writers and coverage finish.
The finalizer replaces only its own links with byte-verified durable copies.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"out/full-sweep-wan-20261101"


def status(data):
    p=DEST/"storage-finalization.json";tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n");tmp.replace(p)


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    memory=shutil.disk_usage("/dev/shm")
    assert memory.free>40*1024**3,"Insufficient free RAM filesystem; keep canonical storage unchanged"
    assert not (DEST/"storage-staging.json").exists(),"An existing staging finalizer must not be duplicated"
    stage=Path(tempfile.mkdtemp(prefix="tri-sweep-",dir="/dev/shm"))
    routes=[]
    for pid in ("p01","p03"):
        for step in range(16,21):
            canonical=DEST/pid/f"step-{step:03d}"
            if canonical.exists():continue
            target=stage/pid/canonical.name;target.mkdir(parents=True)
            try:canonical.symlink_to(target,target_is_directory=True)
            except FileExistsError:continue  # GPU began this step; leave it alone.
            routes.append({"canonical":str(canonical),"temporary":str(target)})
    info={"pid":os.getpid(),"stage_root":str(stage),"routes":routes,"start_unix":time.time(),
          "method":"Only future steps routed; same computations and logical paths. Byte-verify complete durable copies before replacing own symlinks."}
    (DEST/"storage-staging.json").write_text(json.dumps(info,indent=2)+"\n")
    status({**info,"stage":"waiting_for_all_writers","persisted":[]})
    print(json.dumps(info),flush=True)
    if not routes:
        stage.rmdir();status({**info,"stage":"complete","persisted":[],"note":"No future steps remained; no data moved"});return
    while not all((DEST/name).exists() for name in ("complete.json","render-complete.json","render-aux-complete.json")):
        time.sleep(5)
    persisted=[]
    for route in routes:
        canonical=Path(route["canonical"]);source=Path(route["temporary"])
        assert canonical.is_symlink() and canonical.resolve()==source
        durable=canonical.with_name("."+canonical.name+".persisting")
        status({**info,"stage":"copying_and_verifying","current":str(canonical),"persisted":persisted})
        shutil.copytree(source,durable,dirs_exist_ok=True)
        files=[p for p in source.rglob("*") if p.is_file()]
        hashes={};total=0
        for p in files:
            relative=p.relative_to(source);copy=durable/relative
            assert copy.is_file() and p.stat().st_size==copy.stat().st_size,relative
            h=digest(p);assert digest(copy)==h,relative
            hashes[str(relative)]=h;total+=p.stat().st_size
        # All GPU/render writers have exited. Replace only the link created
        # above; if a rename fails, immediately restore that known-good link.
        assert canonical.is_symlink() and canonical.resolve()==source
        canonical.unlink()
        try:durable.replace(canonical)
        except BaseException:
            canonical.symlink_to(source,target_is_directory=True);raise
        manifest={"canonical":str(canonical),"files":len(files),"bytes":total,"sha256":hashes}
        (canonical/"storage-byte-verification.json").write_text(json.dumps(manifest)+"\n")
        persisted.append({"canonical":str(canonical),"files":len(files),"bytes":total,"all_sha256_verified":True})
        status({**info,"stage":"persisting","persisted":persisted})
        print("PERSISTED",canonical,len(files),total,flush=True)
    assert stage.parent==Path("/dev/shm") and stage.name.startswith("tri-sweep-") and stage.stat().st_uid==os.getuid()
    shutil.rmtree(stage)  # Only our private, fully persisted temporary tree.
    status({**info,"stage":"complete","persisted":persisted,"finish_unix":time.time()})
    print("STORAGE FINALIZATION COMPLETE",flush=True)


if __name__=="__main__":
    try:main()
    except BaseException as error:
        status({"stage":"failed","error":repr(error),"traceback":traceback.format_exc()})
        raise
