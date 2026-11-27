# Study specs

One JSON file per experiment; `python scripts/center_study.py configs/studies/<name>.json` runs selection →
training-free screen → stage-2 fine-tune → 100-prompt benchmarks of the trained means and of the zeros swap →
evidence under `distill-results/<name>/` (idempotent, resumable, budget-enforced; `--dry-run` prints every command).

## Fields

| field | meaning |
|---|---|
| `name`, `description` | study id (letters, digits, `_`, `-`) and the paragraph that opens its README |
| `selection` | which pairs get which role: a preset name or `{"policies": [...], "wide": bool}` (below) |
| `bypass` | what stays native: `{}` (nothing), or any of `layers`, `steps`, `branches`, `heads`, `tokens` (below) |
| `train` | recipe overrides: `steps, accum, lr, lr_center, warmup, lambda_cfg, lambda_feat, ema, clip, val_every, val_prompts, save_every, resume_state_every, seed, ranks, max_retries, gate` (`gate = {"step": 1000, "max_guided": 0.10}` stops and asks) |
| `arms` | `{"mean": "<arm name>", "delete": "<arm name>"}`: trained means, and the same pairs zeroed at inference |
| `references`, `curves` | arms and validation curves the paired tables and the figure are drawn against |
| `approved_gpu_hours` | the user-approved allocation; the runner stops with an `ASK` state instead of exceeding it |
| `screen_prompts`, `reserve_benchmark_gpu_hours`, `gpus`, `lab`, `publish` | defaults in `scripts/center_study.py` |

Legacy fields `variant` + `means_pairs` (a preset with that size) and `exempt` (layer 0 / step 1 native) are still
accepted and map onto `selection` / `bypass`.

## Selection policies (`scripts/pair_selection.py`)

Each policy scores pairs on the pooled calibration statistics, ranks them per cell (layer, step, CFG branch,
head) and gives the chosen pairs a role. Policies apply in order; roles accumulate.

```json
{"criterion": "error", "combine": "shared", "pool": [], "pairs": 16, "rank": "lowest", "role": "mean"}
```

- `criterion` (per side; lower = safer to replace): `damage` (replacement error × partner energy, the original),
  `error` (replacement error alone), `error_normalized` (error / own energy = 1 − RMS-MRL², pure concentration),
  `energy` (energy × partner energy, the zero-replacement damage), `fixed_high` (the original high third), `all`.
- `combine`: `shared` (one set for Q and K from score_q + score_k) or `separate`.
- `pool`: axes summed before ranking, any of `steps`, `branches`, `heads` (the chosen set is then the same across them).
- `pairs` 1..64, `rank` `lowest` | `highest`.
- `role`: `mean` (learned constant; zeroed in the `delete` arm) or `content_nope` (kept as content, no rotary
  embedding). `{"complement_of": "mean", "role": "content_nope"}` gives the remaining pairs a role.
- `wide: true` selects the widened head (means on all pairs + the content pairs appended to Q/K; `learned_center.py`).

Presets (reproduce the historical variants exactly, tested): `shared`, `separate`, `pooled`, `split`, `wide`, `fixed_high`.
A preset can be used with overrides: `{"preset": "shared", "policies": [...]}`.

## Bypass rules

```json
{"layers": [0], "steps": [0], "branches": [1], "heads": {"3": [0, 5]}, "tokens": {"kind": "frames", "native": [0]}}
```

`layers`/`steps`: whole layers / denoising steps native (as the old exemptions). `branches`: CFG branch ids native
(0 conditional, 1 unconditional). `heads`: a list for every layer or `{"<layer>": [heads]}`. `tokens`:
`{"kind": "frames", "native": [...]}` keeps those latent frames native, `{"kind": "border", "width": w}` keeps a
spatial border of `w` patches native; token rules apply inside every intervention (means, zeros, NoPE, widened head).
Every checkpoint and benchmark record stores the selection spec, its hash and the bypass rules.

## The 16-pair criterion studies (27 November)

| spec | selection | why |
|---|---|---|
| `error16.json` | 16 lowest by `error`, shared | the senior's proposal: drop the partner-energy factor so the choice is not biased to low-energy pairs |
| `damage16.json` | 16 lowest by `damage`, shared | the control at the same budget |
| `errnorm16.json` | 16 lowest by `error_normalized`, shared | optional: energy ignored entirely (concentration only) |

All three: nothing exempt, the tuned 2,500-step recipe, arms = trained means and the zeros swap, 14 GPU-hours
approved each. Run them one at a time or on disjoint idle GPUs:

```sh
python scripts/center_study.py configs/studies/error16.json
python scripts/center_study.py configs/studies/damage16.json
```

Each README under `distill-results/<name>/` reports the selected pairs' pooled energy relative to all pairs
(`energy_q_mean_pairs_over_all` in `selection-report.json`), which is the quantity the senior's argument is about.
