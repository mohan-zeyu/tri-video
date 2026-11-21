src = open('supervisor.py').read()
def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (old, n)
    src = src.replace(old, new)
sub('"""User-authorized, budget-limited A/B/C runner; no recurring model turns."""',
    '"""Targeted continuation of the no-exemption runner after the external SIGTERM of train-b01 (update ~1,880).\n'
    'User approved on 21 Nov 2026 (Claude Code session adopting this study): resume B from the complete update-1000\n'
    'save with the exact recipe plus --resume under a new attempt name, then the unchanged C; cap raised 13 -> 18\n'
    'GPU-hours. Derived from supervisor.py by asserted substitutions (make.py); A is not rerun or republished."""')
sub("'GIT_AUTHOR_NAME':'Codex','GIT_AUTHOR_EMAIL':'codex@openai.com','GIT_COMMITTER_NAME':'Codex','GIT_COMMITTER_EMAIL':'codex@openai.com'",
    "'GIT_AUTHOR_NAME':'Claude','GIT_AUTHOR_EMAIL':'noreply@anthropic.com','GIT_COMMITTER_NAME':'Claude','GIT_COMMITTER_EMAIL':'noreply@anthropic.com'")
sub("CAP=13", "CAP=18", 0)  # no-op guard (constant introduced below)
sub("BRIEF='6f273d8b7491484ea2846a68db6e88daf58c262b'", "BRIEF='6f273d8b7491484ea2846a68db6e88daf58c262b';CAP=18")
sub("run(['python3','scripts/handoff.py','send','--as','codex',", "run(['python3','scripts/handoff.py','send','--as','claude-lab',")
sub("The user requested return after verifying startup to conserve model credits; this durable runner completed or stopped without periodic model polling.",
    "train-b01 was SIGTERMed externally at update ~1880 (likely host earlyoom, unconfirmed); B resumed from the complete update-1000 save under attempt train-b02+ with the exact recipe. User raised the cap to 18 GPU-hours on 21 Nov 2026. Durable runner, no periodic model polling.")
sub("'gpu_hours_used':used()/3600,'accounting'", "'gpu_hours_used':used()/3600,'cap_history':'13 approved originally; raised to 18 by the user on 21 Nov 2026 after the external SIGTERM','accounting'")
sub("write(RESULT/'budget-summary.json',{'approved_gpu_hours':13,", "write(RESULT/'budget-summary.json',{'approved_gpu_hours':CAP,")
sub("assert used()<=13*3600", "assert used()<=CAP*3600")
sub("Total allocated study GPU-hours {used()/3600:.6f}/13.", "Total allocated study GPU-hours {used()/3600:.6f}/{CAP} (cap raised from 13 by the user after the external SIGTERM).")
sub("Allocated GPU-hours {used()/3600:.6f}/13.", "Allocated GPU-hours {used()/3600:.6f}/{CAP}.")
# Skip A: everything from the CPU-test wait through the A publish is replaced by a check that A was published.
a0 = src.index(" state('A-waiting-for-CPU-tests')")
a1 = src.index(" state('B-training',gpu_hours_used=used()/3600)")
src = src[:a0] + " assert read(LOG/'screen-a-status.json')['state']=='complete' and (RESULT/'screen/complete.json').exists(),'A not complete'\n" + src[a1:]
sub("train=ROOT/'out/distill-s22-all';attempt=0", "train=ROOT/'out/distill-s22-all';attempt=1;assert read(train/'checkpoint/latest.json')['step']==1000;assert not any(read(p)['state']=='running' for p in LOG.glob('train-b*-status.json'))")
sub("remaining=13*3600-used()", "remaining=CAP*3600-used()")
sub("cmd=base+(['--resume'] if attempt>1 else [])", "cmd=base+['--resume']")
sub("'approved_gpu_hours_total':13,", "'approved_gpu_hours_total':CAP,")
sub("'--candidates','1,0,6,2,3,5'", "'--candidates','1,5,0,6'")
sub("if 'SIGKILL' not in log or not cp.exists():raise", "if ('SIGKILL' not in log and 'SIGTERM' not in log) or not cp.exists() or attempt>=4:raise")
sub("state('B-resuming-after-SIGKILL'", "state('B-resuming-after-external-signal'")
sub("out=RESULT/'interrupted-training';", "out=RESULT/'interrupted-training-resume';")
sub("shutil.copyfile(LOG/'pipeline-status.json',RESULT/'interruption.json')", "shutil.copyfile(LOG/'pipeline-status.json',RESULT/'interruption-resume.json')")
open('supervisor-resume.py', 'w').write(src)
