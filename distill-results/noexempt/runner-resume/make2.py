src = open('supervisor-resume.py').read()
def sub(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sub('"""Targeted continuation of the no-exemption runner after the external SIGTERM of train-b01 (update ~1,880).',
    '"""Second targeted continuation (make2.py from supervisor-resume.py): train-b02 was SIGTERMed by earlyoom at update ~2,010\n'
    '(rank 0 the largest host process at ~32 GB RSS, MemAvailable ~25 GB); train-b03 waited on GPUs 1/6 that falsely reported\n'
    '100% utilisation and was stopped before claiming any GPU. User approved on 21 Nov 2026 moving the retry to idle GPUs 2/3.\n'
    'Resumes from the complete update-2000 save as train-b04 with the exact recipe; cap unchanged at 18 GPU-hours.\n\n'
    'Earlier: targeted continuation of the no-exemption runner after the external SIGTERM of train-b01 (update ~1,880).')
sub("attempt=1;assert read(train/'checkpoint/latest.json')['step']==1000;", "attempt=3;assert read(train/'checkpoint/latest.json')['step']==2000;")
sub("assert not any(read(p)['state']=='running' for p in LOG.glob('train-b*-status.json'))",
    "assert not any(read(p)['state'] in ('running','waiting_for_idle_gpus') for p in LOG.glob('train-b*-status.json'))")
sub("'--candidates','1,5,0,6'", "'--candidates','2,3,1,6,5,0'")
sub("or not cp.exists() or attempt>=4:raise", "or not cp.exists() or attempt>=5:raise")
sub("B resumed from the complete update-1000 save under attempt train-b02+ with the exact recipe.",
    "B resumed from the complete update-1000 save as train-b02, SIGTERMed again by earlyoom at ~2010, resumed from update 2000 as train-b04 on GPUs 2/3 (b03 never claimed GPUs).")
open('supervisor-resume2.py', 'w').write(src)
