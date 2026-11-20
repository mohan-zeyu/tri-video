# Archived lab runner source

Copied from the active lab `out/distill-noexempt-run/` directory for takeover/review. `source-manifest.json` records capture time, source paths, byte lengths and SHA-256 values. These files are not a new launch request.

**Do not run `supervisor.py` blindly. It is not restart-idempotent against an existing study.** It assumes an initial B attempt when entering that stage. The existing supervisor may already own live workers or their outputs. Read `AGENT_TAKEOVER.md` and inspect live state first. Copies here and in the Mac snapshot are historical; live remote status/checkpoints are authoritative.

The current runner adopts the A screen that was already launched separately, then handles B and C, bounded GPU allocation, stage reports/commits and a terminal hand-off. Its Git transport uses verified SSH port 443 with the existing trusted host key. No credentials are stored here.
