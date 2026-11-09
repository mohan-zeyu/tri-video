"""Mark delivery ready only after audit, reports and durable copies pass."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default="out/full-sweep-wan-20261101")
    ap.add_argument("--pdf-sha256",required=True)
    args=ap.parse_args();r=ROOT/args.root
    previous=json.loads((r/"delivery.json").read_text()) if (r/"delivery.json").exists() else {}
    while True:
        storage=json.loads((r/"storage-finalization.json").read_text())
        if storage["stage"]=="complete":break
        if storage["stage"]=="failed":raise RuntimeError(storage)
        time.sleep(5)
    assert len(storage["persisted"])==10
    assert all(x["all_sha256_verified"] for x in storage["persisted"])
    assert not Path(storage["stage_root"]).exists()
    assert not [p for p in r.glob("p*/step-*") if p.is_symlink()]
    c=json.loads((r/"coverage.json").read_text())
    assert c["observed"]["layer_captures"]==2400 and c["observed"]["head_figures"]==86400
    assert c["bit_exact_checks"]==c["attention_output_validation_checks"]==144
    assert c["independent_real_tensor_spot_checks"]["passed"]
    assert json.loads((r/"supervisor-status.json").read_text())["stage"]=="complete"
    assert sha(r/"report.pdf")==args.pdf_sha256
    assert (r/"index.html").is_file()
    names=["report.pdf","REPORT.md","index.html","coverage.json","per-head-results.csv","config.json","real-tensor-spot-checks.json","storage-finalization.json"]
    selection=json.loads((r/"report-figures/selection.json").read_text())
    assert selection["source_csv_sha256"]==sha(r/"per-head-results.csv")
    names += ["report-figures/selection.json","report-figures/index.html"]
    for axis in ("t","h","w"):
        for kind in ("fixed","best_pair","best_reconstruction"):
            example=selection["examples"][axis][kind]
            assert sha(r/example["source_layer_json"])==example["source_layer_sha256"]
            for suffix in (".png",".pdf",".svg"):
                names.append(str(Path(example["figure"]).with_suffix(suffix)))
            names.append(example["report_figure"])
    heatmaps=None
    if (r/"mrl-heatmaps/manifest.json").exists():
        heatmaps=json.loads((r/"mrl-heatmaps/manifest.json").read_text())
        assert heatmaps["conditions"]==80 and heatmaps["appendix_pages"]==160
        assert heatmaps["source_inputs"]["coverage_sha256"]==sha(r/"coverage.json")
        assert heatmaps["source_inputs"]["rope_sha256"]==sha(r/"rope.pt")
        expected_sources={item["path"]:item["sha256"] for item in c["files"]}
        for condition in heatmaps["source_inputs"]["conditions"]:
            path=r/"mrl-heatmaps/data"/condition["path"].removesuffix(".gz")
            assert sha(path)==condition["sha256"]
            values=json.loads(path.read_text())
            assert len(values["source_layers"])==30
            assert all(expected_sources[item["path"]]==item["sha256"] for item in values["source_layers"])
            names.append(str(path.relative_to(r)))
        names += ["mrl-heatmaps.js","mrl-heatmaps/manifest.json","wan-mrl-heatmaps-complete.pdf"]
        names += [str(path.relative_to(r)) for path in sorted((r/"mrl-heatmaps/figures").glob("*.png"))]
        if (r/"mrl-heatmaps/pattern-analysis.json").exists():
            from mrl_pattern_summary import load_summary
            analysis=load_summary(r)
            assert len(analysis["findings"])==7
            names.append("mrl-heatmaps/pattern-analysis.json")
    result={"status":"complete","verified_unix":time.time(),"scope":{"model":"Wan2.1-T2V-1.3B","prompts":4,"steps_per_prompt":20,"layers":30,"heads":12,"axes":["t","h","w"],"video_tokens_per_condition":10800},
            "provenance":{"scatter_and_concentration":"one axis-selected 2D pair per prompt/step/layer/head; every video token",
                          "logit_prediction":"full 128D head, all 64 pairs; same-condition centers average all 10,800 video tokens; every query and every same-axis-line key",
                          "primary_pooling":"no prompt, step, layer or head pooling in primary head figures; curve points average same-offset query/key logits; distributions pool 12 heads at one prompt/step/layer/axis"},
            "report_revision":"Complete A/B/C/D examples, frequency heatmaps and concise interpretation with per-result averaging scopes",
            "atlas_revision":"All-condition frequency heatmaps plus shared scope-labelled analysis and source profiles",
            "coverage":c["observed"],"native_bit_exact_checks":144,"staged_folders_byte_verified":10,
            "staged_bytes_verified":sum(x["bytes"] for x in storage["persisted"]),
            "remaining_temporary_step_symlinks":0,"sha256":{name:sha(r/name) for name in names}}
    if "private_sharing" in previous:result["private_sharing"]=previous["private_sharing"]
    if heatmaps is not None:
        result["mrl_heatmaps"]={"conditions":80,"layers":30,"heads":12,"pairs":64,"tokens_per_pair":10800,
                               "appendix_pages":160,"groups":heatmaps["band_definition"],"cell":heatmaps["cell_definition"]}
    (r/"delivery.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
