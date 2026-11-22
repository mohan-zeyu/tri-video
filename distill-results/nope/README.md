Content/position split, A+B complete.

Selection uses 64 calibration prompts; per-prompt Q/K means and pair energies average all 10,800 tokens after full-width RMSNorm and before RoPE. Rank shared damage separately per layer, step, branch and head. The 42 means pairs and 22 NoPE pairs cover all 64; original shared22 matches exactly. Layer 0 and the first denoising step stay native. The screen averages full-latent relative squared errors over 32 held-out prompts × 20 teacher entering states, with both branches combined at CFG 5 for the guided column.

No training-free videos. C uses direct teacher initialization, without the optional means-only warm start. Lab outputs: `/mnt/data/liuzeyu/tri-video-nope-2979d3e/out/distill-selection-v3/`. GPU accounting is in `screen-b-budget.json` and `screen-b2-budget.json`.

The user approved a 27.5 GPU-hour total cap on 19 November 2026 (Asia/Singapore), recorded in `budget-approval.json`. Exact C resume from update 3,000 continues the original 5,000-update recipe; D and E follow. The historical `progress-3000/` snapshot records the preceding budget pause.
