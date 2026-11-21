"""Budgeted D/E benchmark driver for the widened-head study (DISTILL_WIDE_PROMPT.md); run after C.
Derived from the no-exemption benchmark.py by asserted substitutions (make-benchmark.py)."""
import sys, fcntl, hashlib, json, os, signal, subprocess, time
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'scripts'))
from full_sweep import write_json
PHASE=sys.argv[1]
assert PHASE in ('D','E')
LAB=ROOT.parent
PYTHON='/dev/shm/tri-center-runtime/bin/python'
VBench='/dev/shm/tri-vbench-source'
FROZEN=LAB/'tri/out/center-benchmark-wan-20261109'
REUSE=LAB/'tri-video-distill-bf334ec/out/learned-center-wan-student'
SCRATCH=Path('/tmp/tri-distill-pilot.EG6asH')
STUDENT=ROOT/'out/distill-wide32/student'
LOGS=ROOT/'out/distill-wide-run'
RUN=ROOT/('out/learned-center-wan-wide32' + ('-zeros' if PHASE=='E' else ''))
ARM='wide32' if PHASE=='D' else 'wide32_zeros'
MODE='mean' if PHASE=='D' else 'delete'
DIMENSIONS=['overall_consistency','aesthetic_quality','imaging_quality','dynamic_degree','subject_consistency','background_consistency','motion_smoothness']
MAX_GPU_WORKERS=2
ASSIGNMENT_ORDER=[2,3,1,6,0,5]
LEDGER=LOGS/('phase-'+PHASE.lower()+'-gpu.tsv')
ENV={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'','HF_HOME':str(LAB/'hf-cache'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WAN_SNAPSHOT':'/dev/shm/tri-wan-snapshot','VBENCH_CACHE_DIR':str(SCRATCH/'evaluator-cache-static'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
cb_used=sum(json.loads(p.read_text()).get('gpu_seconds',0) for p in LOGS.glob('*-status.json') if p.name.startswith(('screen-b','train-c')))
other_ledger=LOGS/('phase-e-gpu.tsv' if PHASE=='D' else 'phase-d-gpu.tsv')
other_used=sum(float(line.split('\t')[2]) for line in other_ledger.read_text().splitlines()) if other_ledger.exists() else 0
APPROVAL=json.loads((LOGS/'budget-approval.json').read_text())
STUDY_BUDGET_GPU_SECONDS=float(APPROVAL['approved_gpu_hours_total'])*3600
BUDGET_GPU_SECONDS=STUDY_BUDGET_GPU_SECONDS-cb_used-other_used

def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=10).strip()


def lease_idle_devices():
    for device in ASSIGNMENT_ORDER:
        lock = (SCRATCH / f"gpu-{device}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        fields = smi("-i", str(device), "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits")
        uuid, memory, utilization = (field.strip() for field in fields.split(","))
        if int(memory) < 32 and int(utilization) == 0 and not smi("-i", str(device), "--query-compute-apps=pid", "--format=csv,noheader"):
            yield device, uuid, lock
        else:
            lock.close()


def run_cpu(name, script, args):
    with (LOGS / f"{name}.log").open("ab") as stream:
        subprocess.run([PYTHON, "-u", f"scripts/{script}", *args], env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_wave(tasks):
    pending = list(tasks)
    active = []
    used = sum(float(row.split("\t")[2]) for row in LEDGER.read_text().splitlines()) if LEDGER.exists() else 0.0
    sent_stop = None
    failure = None
    print("WAITING FOR IDLE GPU", [task[0] for task in pending], flush=True)
    while pending or active:
        now = time.monotonic()
        for worker in active[:]:
            device, lock, stream, proc, name, started = worker
            rc = proc.poll()
            if rc is None:
                continue
            elapsed = now - started
            used += elapsed
            with LEDGER.open("a") as ledger:
                ledger.write(f"{name}\t{device}\t{elapsed:.3f}\t{rc}\n")
            lock.close()
            stream.close()
            active.remove(worker)
            print(f"EXIT {name} GPU={device} seconds={elapsed:.1f} status={rc}", flush=True)
            if rc:
                failure = f"{name} exited with status {rc}"
        allocated = used + sum(now - w[-1] for w in active)
        if allocated + 60 >= BUDGET_GPU_SECONDS:
            failure = "ASK: widened-head study reached its approved GPU-hour budget"
        if failure:
            pending.clear()
            if not active:
                raise RuntimeError(failure)
            if sent_stop is None:
                sent_stop = now
                for _, _, _, proc, _, _ in active:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            elif now - sent_stop >= 10:
                for _, _, _, proc, _, _ in active:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        elif pending:
            for device, uuid, lock in lease_idle_devices():
                if not pending or len(active) == MAX_GPU_WORKERS:
                    lock.close()
                    break
                name, script, args = pending.pop(0)
                stream = (LOGS / f"{name}-gpu{device}.log").open("ab")
                proc = subprocess.Popen([PYTHON, "-u", f"scripts/{script}", *args],
                                        env={**ENV, "CUDA_VISIBLE_DEVICES": uuid},
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((device, lock, stream, proc, name, time.monotonic()))
                print(f"START {name} GPU={device} pid={proc.pid}", flush=True)
        if pending or active:
            time.sleep(2 if active else 15)


def read(path):
    return json.loads(path.read_text())


def verify_finished_training():
    state=read(ROOT/'out/distill-wide32/checkpoint/latest.json')
    assert state['step']==5000, 'D/E need the complete resumable update-5000 save'
    rows=[json.loads(line) for line in (ROOT/'out/distill-wide32/validation.jsonl').read_text().splitlines()]
    assert rows[-1]['step']==5000
    import torch
    from learned_center import LearnedCenter
    center=LearnedCenter.from_checkpoint(torch.load(STUDENT/'learned_center.pt',map_location='cpu'))
    original=torch.load(ROOT/'out/distill-selection-v4/selection.pt',map_location='cpu')['variants']['wide']
    assert center.is_wide and center.content_pairs==32
    for side in ('q','k'):
        assert torch.equal(center.selected_pairs(side),original['mask_'+side])
    assert torch.equal(center.selected_pairs('nope'),original['mask_nope'])
    assert (center.selected_pairs('q').sum(-1)==64).all() and (center.selected_pairs('nope').sum(-1)==32).all()
    assert center.layer_active.nonzero().flatten().tolist()==list(range(30))
    assert center.step_active.nonzero().flatten().tolist()==list(range(20))


def encoder_parity(cfg, prompt):
    target=RUN/'encoder-parity.json'
    if target.exists():
        return read(target)['used_resident_text_encoder']
    roots={False:LOGS/('parity-'+PHASE+'-offload'),True:LOGS/('parity-'+PHASE+'-resident')}
    for resident,root in roots.items():
        if not (root/'records'/prompt/ARM/'complete.json').exists():
            run_wave([(f'{PHASE}-encoder-{resident}','generate_learned_center.py',
                ['--config',str(cfg),'--out',str(root),'--student',str(STUDENT),'--arm',ARM,'--center-mode',MODE,
                 '--prompts',prompt]+(['--resident-text-encoder'] if resident else []))])
    import torch
    a,b=[path/'records'/prompt/ARM for path in roots.values()]
    same_latent=torch.equal(*(torch.load(path/'final-latents.pt',map_location='cpu') for path in (a,b)))
    same_video=read(a/'complete.json')['video_sha256']==read(b/'complete.json')['video_sha256']
    resident=same_latent and same_video
    for relative in [Path('videos')/ARM/(prompt+'.mp4'),*(Path('records')/prompt/ARM/f for f in ('complete.json','final-latents.pt'))]:
        dest=RUN/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(roots[resident]/relative,dest)
    write_json(target,{'prompt_id':prompt,'identical_final_latents':same_latent,'identical_mp4_sha256':same_video,'used_resident_text_encoder':resident})
    return resident


def main():
    verify_finished_training()
    RUN.mkdir(parents=True,exist_ok=True)
    config=read(ROOT/'configs/learned_center_benchmark_wan.json')
    config['modes']=['baseline',ARM]
    config['center_policy']='Widened-head EMA (wide32) with no exempt layers or denoising steps: per head Q/K = [RoPE(128-dim learned mean) | 32 unrotated post-QK-norm content pairs], logit scale 1/sqrt(128), in all 30 layers and 20 steps; '+('position channel active' if MODE=='mean' else 'position channel zeroed (content-only attention)')+'.'
    cfg=RUN/'config.json'
    if cfg.exists():assert read(cfg)==config
    else:write_json(cfg,config)
    common=['--config',str(cfg),'--out',str(RUN)]
    run_cpu(PHASE+'-link-baseline','generate_learned_center.py',common+['--arm','baseline','--link-baseline',str(FROZEN)])
    if PHASE=='E':
        screen=ROOT/'out/distill-selection-v4/screen-trained'
        if not (screen/'complete.json').exists():
            run_wave([('E-trained-screen','screen_selections.py',['--traj',str(SCRATCH/'traj'),'--val-prompts','32',
                '--student',str(STUDENT),'--modes','mean','delete','--out',str(screen)])])
    resident=encoder_parity(cfg,config['prompts'][0]['id'])
    pending=[p['id'] for p in config['prompts'] if not (RUN/'records'/p['id']/ARM/'complete.json').exists()]
    if pending:
        args=common+['--arm',ARM,'--student',str(STUDENT),'--center-mode',MODE]+(['--resident-text-encoder'] if resident else [])
        # Atomic per-prompt claim directories allow both workers to share the manifest.
        run_wave([(PHASE+'-render-'+str(i),'generate_learned_center.py',args) for i in range(2)])
    center_hash=hashlib.sha256((STUDENT/'learned_center.pt').read_bytes()).hexdigest()
    selection=[]
    for prompt in config['prompts']:
        rec=read(RUN/'records'/prompt['id']/ARM/'complete.json')
        env=rec['environment']; sel=env['selection']
        assert rec['prompt']==prompt and rec['mode']==ARM and rec['finite_step_latents']==20
        assert rec['intervention_calls']==rec['expected_intervention_calls']==2400
        assert rec['nope_calls']==rec['expected_nope_calls']==1200
        assert sel['pairs_per_cell_q']==sel['pairs_per_cell_k']==sel['q_k_intersection_mean']==64.0
        assert sel['nope_pairs_per_cell']==32.0 and not sel['default_high_masks']
        assert sel['wide_head'] is True and sel['content_pairs']==32
        assert env['center_mode']==MODE and env['learned_center_sha256']==center_hash
        assert env['weights']==str(STUDENT/'transformer') and env['exempt']=={'layers':[],'steps':[]}
        assert rec['frames']==[45,480,480,3]
        selection.append({'prompt':prompt,'arm':ARM,'center_mode':MODE,'selection':sel,'exempt':env['exempt'],
                          'intervention_calls':2400,'processor_calls':1200,'video_sha256':rec['video_sha256']})
    write_json(RUN/'selection-records.json',selection)
    tasks=[(PHASE+'-'+dim,'evaluate_center_benchmark.py',['--root',str(RUN),'--config',str(cfg),'--source',VBench,
            '--dimension',dim,'--batch-prompts','100','--reuse-baseline-from',str(REUSE)])
           for dim in DIMENSIONS if not (RUN/'evaluation'/dim/'complete.json').exists()]
    if tasks:run_wave(tasks)
    run_cpu(PHASE+'-summary','summarize_center_benchmark.py',['--root',str(RUN),'--config',str(cfg),
            '--baseline-config','configs/center_benchmark_wan.json','--extra-dimension','overall_consistency'])
    refs=[('baseline',RUN/'summary.json')]
    if PHASE=='D':
        refs += [('dense_control',ROOT/'distill-results/dense-paired/summary.json'),
                 ('shared22',ROOT/'distill-results/shared22/full100/summary.json'),
                 ('shared22_delete',ROOT/'distill-results/shared22-delete/full100/summary.json'),
                 ('split42',ROOT/'distill-results/nope/full100/split42/summary.json')]
    else:
        refs += [('wide32',ROOT/'out/learned-center-wan-wide32/summary.json')]
    args=['--candidate',ARM,str(RUN/'summary.json')]
    for name,path in refs:args += ['--reference',name,str(path)]
    run_cpu(PHASE+'-paired','compare_distill_arms.py',args+['--out',str(RUN/'paired-comparisons.json')])
    run_cpu(PHASE+'-agreement','measure_distill_agreement.py',['--root',str(RUN),'--config',str(cfg)])
    used=sum(float(row.split('\t')[2]) for row in LEDGER.read_text().splitlines())
    write_json(LOGS/('phase-'+PHASE.lower()+'-summary.json'),{'phase':PHASE,'complete':True,'gpu_seconds':used,
                'study_gpu_seconds_total':cb_used+other_used+used})
    print(PHASE,'COMPLETE',used,'GPU seconds',flush=True)


if __name__=='__main__':
    main()
