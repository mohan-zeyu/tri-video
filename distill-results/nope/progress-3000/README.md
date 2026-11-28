Interim NoPE C evidence at the complete update-3,000 checkpoint.

The run is paused for the brief budget ASK. It has not reached 5,000 updates, and D/E have not run. The step-1,000 guided gate passed. Validation uses the same eight held-out prompts × 20 teacher entering states; per-state whole-latent relative squared errors are averaged equally, with both branches combined at CFG 5. Means replace 42 rotary pairs on Q and K for every token in each active layer/step/branch/head cell; the 22 complementary pairs use NoPE. Layer 0 and denoising step 1 remain native.

`budget-review.json` separates measured allocated GPU time from the completion-cost estimate. `c-finish5000-status.json` records exit status 1 from the intentional SIGTERM after the full save; `budget-pause-3000.json` records that decision. Model weights remain on the lab host; `ema-step-003000.sha256` hashes the preserved EMA export. The proposed 27.5 GPU-hour cap is awaiting user approval.
