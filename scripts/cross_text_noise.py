"""Crossed inputs for step 1: the TEXT of one prompt with the initial NOISE of another.

At denoising step 1 the entering state is the initial noise itself, so a crossed input needs no sampling run:
it is the stored step-1 state (`states.pt[0]`) of the noise prompt together with the stored text embeddings
(`embeddings.pt`) of the text prompt, both read from an existing teacher trajectory cache (never written to).

Writes <out>/t<text digits>n<noise digits>/ in the layout `TrajectoryStore` reads (scripts/train_learned_center.py):

  embeddings.pt   a byte-for-byte copy of the text prompt's embeddings.pt (positive and negative)
  states.pt       ONLY the noise prompt's step-1 state, shape [1, ...]. Steps 2-20 are deliberately absent, so
                  `cell_pair_scores.py capture --step N` with N > 1 raises IndexError instead of running.
                  (NaN states would not be caught: the capture's validation `relative_max_err > 1e-3` is
                  false for NaN.)
  meta.json       the noise prompt's `timesteps` (TrajectoryStore needs all of them to index step 1), a `prompt`
                  entry naming the text prompt, the noise prompt and the noise seed, and `meaningful_steps` [1].

After writing, it reloads both files and checks them against the sources (check 1 of DISTILL_SCORE_SEED_PROMPT.md):
the step-1 state must equal the noise prompt's `states.pt[0]` exactly and every embedding tensor must equal the
text prompt's exactly. It exits non-zero if not.

  python scripts/cross_text_noise.py --traj <trajectory cache> --text d0000 --noise d0006 --out out/seed-traj
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def crossed_id(text, noise):
    return f"t{text.lstrip('d')}n{noise.lstrip('d')}"


def cross(traj, text, noise, out):
    traj, out = Path(traj).resolve(), Path(out).resolve()
    if out == traj or traj in out.parents:
        raise SystemExit(f"refusing to write inside the trajectory cache {traj}")
    src_text, src_noise = traj / text, traj / noise
    for path in (src_text / "embeddings.pt", src_noise / "states.pt", src_noise / "meta.json"):
        if not path.exists():
            raise SystemExit(f"missing input {path}")
    name = crossed_id(text, noise)
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    meta_noise = json.loads((src_noise / "meta.json").read_text())
    states = torch.load(src_noise / "states.pt", mmap=True)
    first = states[0:1].clone()
    shutil.copyfile(src_text / "embeddings.pt", folder / "embeddings.pt")
    torch.save(first, folder / "states.pt")
    meta = {"prompt": {"id": name, "text_from": text, "noise_from": noise,
                       "seed_of_noise": meta_noise.get("prompt", {}).get("seed")},
            "timesteps": meta_noise["timesteps"], "steps": len(meta_noise["timesteps"]),
            "states_shape": list(first.shape), "meaningful_steps": [1],
            "note": "step-1 input only: states.pt holds the noise prompt's stored step-1 state and nothing else",
            "sources": {"embeddings": str(src_text / "embeddings.pt"), "embeddings_sha256": sha256(src_text / "embeddings.pt"),
                        "states": str(src_noise / "states.pt"), "states_sha256": sha256(src_noise / "states.pt")}}

    # check 1: reload what was written and compare with the sources, exactly
    written_states = torch.load(folder / "states.pt")
    written_emb = torch.load(folder / "embeddings.pt")
    source_emb = torch.load(src_text / "embeddings.pt")
    check = {"state_shape": list(written_states.shape),
             "state_equals_noise_step1": bool(written_states.shape[0] == 1 and torch.equal(written_states[0], states[0])),
             "state_max_abs_diff": float((written_states[0].float() - states[0].float()).abs().max()),
             "embedding_keys": sorted(written_emb),
             "embeddings_equal_text": bool(sorted(written_emb) == sorted(source_emb)
                                           and all(torch.equal(written_emb[k], source_emb[k]) for k in source_emb)),
             "embeddings_sha256_equal": sha256(folder / "embeddings.pt") == meta["sources"]["embeddings_sha256"]}
    meta["check"] = check
    meta["states_sha256"] = sha256(folder / "states.pt")
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    print("CROSSED", folder, json.dumps(check), flush=True)
    if not (check["state_equals_noise_step1"] and check["embeddings_equal_text"] and check["embeddings_sha256_equal"]):
        raise SystemExit(f"check 1 failed for {name}")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", required=True, help="existing teacher trajectory cache (read only)")
    parser.add_argument("--text", required=True, help="prompt id whose text embeddings are used")
    parser.add_argument("--noise", required=True, help="prompt id whose stored step-1 state (initial noise) is used")
    parser.add_argument("--out", required=True, help="new trajectory root; must not be inside --traj")
    args = parser.parse_args()
    cross(args.traj, args.text, args.noise, args.out)


if __name__ == "__main__":
    main()
