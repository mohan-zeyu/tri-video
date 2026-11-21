"""Second targeted continuation (make2.py from supervisor-resume.py): train-b02 was SIGTERMed by earlyoom at update ~2,010
(rank 0 the largest host process at ~32 GB RSS, MemAvailable ~25 GB); train-b03 waited on GPUs 1/6 that falsely reported
100% utilisation and was stopped before claiming any GPU. User approved on 21 Nov 2026 moving the retry to idle GPUs 2/3.
Resumes from the complete update-2000 save as train-b04 with the exact recipe; cap unchanged at 18 GPU-hours.

Earlier: targeted continuation of the no-exemption runner after the external SIGTERM of train-b01 (update ~1,880).
User approved on 21 Nov 2026 (Claude Code session adopting this study): resume B from the complete update-1000
save with the exact recipe plus --resume under a new attempt name, then the unchanged C; cap raised 13 -> 18
GPU-hours. Derived from supervisor.py by asserted substitutions (make.py); A is not rerun or republished."""
import fcntl,hashlib,json,math,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/mnt/data/liuzeyu/tri-video-noexempt-6f273d8');os.chdir(ROOT)
LOG=ROOT/'out/distill-noexempt-run';RESULT=ROOT/'distill-results/noexempt';RESULT.mkdir(parents=True,exist_ok=True)
PYTHON='/dev/shm/tri-center-runtime/bin/python';BRANCH='claude/magical-goodall-dmq9i1';BRIEF='6f273d8b7491484ea2846a68db6e88daf58c262b';CAP=18
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/mnt/data/liuzeyu/hf-cache','WAN_SNAPSHOT':'/dev/shm/tri-wan-snapshot','GIT_TERMINAL_PROMPT':'0','GIT_SSH_COMMAND':'ssh -o HostName=ssh.github.com -p443 -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2','GIT_AUTHOR_NAME':'Claude','GIT_AUTHOR_EMAIL':'noreply@anthropic.com','GIT_COMMITTER_NAME':'Claude','GIT_COMMITTER_EMAIL':'noreply@anthropic.com'}
os.environ.update(ENV)
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def state(phase,**kw):write(LOG/'pipeline-status.json',{'phase':phase,'time':time.time(),'pid':os.getpid(),**kw})
def run(argv,log):
 with (LOG/log).open('ab') as f:subprocess.run(argv,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
def git(*args):return subprocess.run(['git',*args],cwd=ROOT,env=ENV,text=True,capture_output=True,check=True).stdout

def publish(subject,terminal=None):
 for p in RESULT.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl','.log','.csv'):
   assert 'Wan-AI/' not in p.read_text() and 'models--' not in p.read_text(),p
 git('diff','--check');git('add','DISTILL_REPORT.md','handoff.md','distill-results/noexempt')
 if terminal:
  run(['python3','scripts/handoff.py','send','--as','claude-lab','--to','claude','--ack',BRIEF,'--brief','DISTILL_NOEXEMPT_PROMPT.md','--branch',BRANCH,'--status',terminal,'-m',subject,'-m','User-assigned no-exemption study. See DISTILL_REPORT.md, handoff.md section16 and distill-results/noexempt for measured scopes, exact flags, budget and preserved checkpoint paths. train-b01 was SIGTERMed externally at update ~1880 (likely host earlyoom, unconfirmed); B resumed from the complete update-1000 save as train-b02, SIGTERMed again by earlyoom at ~2010, resumed from update 2000 as train-b04 on GPUs 2/3 (b03 never claimed GPUs). User raised the cap to 18 GPU-hours on 21 Nov 2026. Durable runner, no periodic model polling.'], 'handoff-send.log')
 else:
  git('commit','-q','-m',subject)
  try:git('push','origin','HEAD:'+BRANCH)
  except subprocess.CalledProcessError:
   git('fetch','origin',BRANCH);git('merge','--no-edit','origin/'+BRANCH);git('push','origin','HEAD:'+BRANCH)
def append(text):
 for p in (ROOT/'DISTILL_REPORT.md',ROOT/'handoff.md'):
  with p.open('a') as f:f.write('\n\n'+text+'\n')
def used():
 total=sum(read(p).get('gpu_seconds',0) for p in LOG.glob('*-status.json') if p.name.startswith(('screen-a','train-b')))
 for phase in ('d','e'):
  p=LOG/('phase-'+phase+'-gpu.tsv')
  if p.exists():total+=sum(float(x.split('\t')[2]) for x in p.read_text().splitlines())
 return total

def collect_screen(src,dest):
 shutil.copytree(src,dest,dirs_exist_ok=True)
 p=dest/'complete.json';d=read(p);raw=p.read_bytes();d['source_copy']={'raw_sha256':hashlib.sha256(raw).hexdigest(),'scope_annotation':'Mean uses calibration pooled means for A; trained final EMA means for the later student screen. All layers/steps active; no NoPE pairs.'};write(p,d)
 return d

def mask_audit():
 import sys;sys.path.insert(0,str(ROOT/'scripts'))
 import torch
 from learned_center import LearnedCenter
 train=ROOT/'out/distill-s22-all';saved=torch.load(train/'student/learned_center.pt',map_location='cpu');c=LearnedCenter.from_checkpoint(saved)
 selection=torch.load(ROOT/'out/distill-selection/selection.pt',map_location='cpu')['variants']['shared']
 assert c.layer_active.all() and c.step_active.all() and not c.uses_nope
 assert all(torch.equal(c.selected_pairs(s),selection['mask_'+s]) for s in ('q','k'))
 assert (c.selected_pairs('q').sum(-1)==22).all() and torch.equal(c.mask_q,c.mask_k)
 checkpoint=read(train/'checkpoint/latest.json');assert checkpoint['step']==2500
 cp=train/'checkpoint'/checkpoint['directory'];cs=torch.load(cp/'center.pt',map_location='cpu');assert cs['step']==2500
 assert all((cp/p).exists() for p in ('model/.metadata','rng-rank00.pt','rng-rank01.pt'))
 return {'all_layers_active':30,'all_steps_active':20,'cells':14400,'shared_q_k_pairs_per_cell':22,'nope_pairs':0,'selection_tensor_equal':True,'expected_calls_per_video':2400,'complete_resume_step':2500,'scope':'All layer-step-CFG branch-head cells, post-full-width RMSNorm/pre-RoPE constants over all10800 video tokens; zero exemptions.'}

def collect_training():
 train=ROOT/'out/distill-s22-all';out=RESULT/'training';out.mkdir(exist_ok=True)
 for name in ('args.json','history.jsonl','validation.jsonl'):shutil.copyfile(train/name,out/name)
 write(out/'mask-exemption-audit.json',mask_audit())
 rows=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()];vals=[json.loads(x) for x in (out/'validation.jsonl').read_text().splitlines()]
 assert rows[-1]['step']==vals[-1]['step']==2500
 assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
 assert all(math.isfinite(v) for r in vals for v in r['mean'])
 import statistics
 peaks=[]
 for p in LOG.glob('train-b*.log'):
  for line in p.read_text().splitlines():
   if not line.startswith('{'):continue
   try:r=json.loads(line)
   except ValueError:continue
   if 'peak_cuda_allocated_bytes' in r:peaks.append({'log':p.name,**r})
 summary={'updates':2500,'median_logged_rank_mean_seconds_per_update':statistics.median(r['optimizer_step_seconds'] for r in rows),'time_scope':'Logged optimizer updates, excluding validation/export/checkpoint I/O; allocated wall time separately includes these costs.','reported_peaks':peaks,'mu_norm_drift':{s:{'first':rows[0]['mu_norm_'+s],'last':rows[-1]['mu_norm_'+s],'percent':100*(rows[-1]['mu_norm_'+s]/rows[0]['mu_norm_'+s]-1)} for s in ('q','k')},'mu_scope':'Per-cell Euclidean norm over all128 stored mean coordinates, equal average over all14400 cells including unused coordinates; no token/prompt average at measurement time.','allocated_gpu_seconds_through_B':used()}
 write(out/'summary.json',summary)
 for p in LOG.glob('train-b*-status.json'):shutil.copyfile(p,out/p.name)
 manifest=[]
 for n in ('learned_center.pt','transformer/config.json','transformer/diffusion_pytorch_model.safetensors'):
  h=hashlib.sha256()
  with (train/'student'/n).open('rb') as f:
   for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
  manifest.append(h.hexdigest()+'  student/'+n)
 (out/'final-ema.sha256').write_text('\n'.join(manifest)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8.5,5),constrained_layout=True)
 for name,path in [('No exemptions',out/'validation.jsonl'),('Layer0/step1 exempt',ROOT/'distill-results/shared22/validation.jsonl')]:
  r=[json.loads(x) for x in path.read_text().splitlines()];ax.plot([x['step'] for x in r],[x['mean'][2] for x in r],marker='.',label=name)
 ax.set_xlabel('Optimizer update');ax.set_ylabel('Guided relative squared velocity error');ax.set_title('Live weights: eight prompts x20 teacher states, CFG5\nWhole-latent ratios, equal prompt-state average');ax.grid(alpha=.2);ax.legend();fig.savefig(out/'validation-curve.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {r['step']} | {r['mean'][2]:.6f} |" for r in vals)
 append(f"### No-exemption B: 2500-update fine-tune\n\nFrom teacher weights, original tuned shared22 recipe with --include-layer0 --include-step1, two ranks and accumulation2; seed20261114, model lr1e-5, means lr2e-4, warmup250, feature weight0.1, EMA0.999. All logged losses/validations finite. Full2500 optimizer/sampler/CPU+CUDA RNG/EMA save and final EMA masks audited. Whole-latent relative squared velocity errors average the first eight held-out prompts x20 teacher entering states; both branches combine at CFG5.\n\n| Update | Guided |\n|---:|---:|\n{table}\n\n![Validation curve](distill-results/noexempt/training/validation-curve.png)\n\nMedian logged rank-mean optimizer seconds/update: {summary['median_logged_rank_mean_seconds_per_update']:.6f}, excluding validation/saves. Peak allocator bytes per reported invocation/rank and all arguments are in training/summary.json and args.json. Mean-norm drift: {json.dumps(summary['mu_norm_drift'])}; scope: {summary['mu_scope']} Allocated A+B GPU-hours: {used()/3600:.6f}, including startup, validation, saves/I/O and all attempts. All historical outputs retained.")

def collect_benchmark(phase):
 arm='s22_all_means' if phase=='D' else 's22_all_zeros';src=ROOT/('out/learned-center-wan-s22-all'+('-zeros' if phase=='E' else ''));dest=RESULT/'full100'/arm;dest.mkdir(parents=True,exist_ok=True)
 d=read(src/'summary.json');raw=(src/'summary.json').read_bytes();d['config'].pop('model',None);d['source_copy']={'raw_sha256':hashlib.sha256(raw).hexdigest(),'omitted_fields':['config.model'],'selection_note':'Legacy high_pairs field unused; actual shared22 masks come from final EMA and selection records. No exempt layers or steps.'};write(dest/'summary.json',d)
 for name in ('paired-comparisons.json','per-prompt-results.csv','selection-records.json','encoder-parity.json'):shutil.copyfile(src/name,dest/name)
 shutil.copytree(src/'agreement',dest/'agreement',dirs_exist_ok=True)
 for name in ('phase-'+phase.lower()+'-gpu.tsv','phase-'+phase.lower()+'-summary.json'):shutil.copyfile(LOG/name,dest/name)
 provenance=[]
 for dim in d['dimensions']:
  for p in (src/'evaluation'/dim).glob('batch-*-results.json'):
   b=read(p);provenance.append({'dimension':dim,**{k:b[k] for k in ('source_file','source_sha256','vbench_commit','reused_baseline_from') if k in b}})
 write(dest/'scorer-provenance.json',provenance)
 c=read(dest/'paired-comparisons.json');a=read(dest/'agreement/complete.json')['arms'][arm]
 header=f'### No-exemption C: {arm}, 100 prompts\n\n'
 scope='One original seed per each of100 original prompts;20 UniPC steps, shift3, CFG5, empty negative,45frames480x480 at16fps. Shared22 Q/K means/zeros after full-width RMSNorm before RoPE, all10800 tokens per active cell; all30layers and20steps active, no exemptions, no NoPE. All100 videos audited at2400 replacement calls each. VBench scores are equal prompt averages of original within-video aggregation: aesthetics/MUSIQ all45 frames, text-video eight midpoint frames, other dimensions original VBench handling. Imaging divided by100. Frozen teacher videos/scores reused, not regenerated. Intervals are pointwise95% from10000 paired prompt resamples, NumPy seed20261109, one fixed seed per prompt; no across-seed uncertainty.\n\n'
 table='| Dimension | Candidate | Candidate minus references, mean [95% CI] |\n|---|---:|---|\n'
 compatible=[];different=[]
 for dim,item in d['dimensions'].items():
  comparisons='; '.join(k+f": {v['mean']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]" for k,v in c['dimensions'][dim].items())
  table+=f"| {dim} | {item['means'][arm]:.6f} | {comparisons} |\n"
  key=arm+'_minus_shared22'
  if key in c['dimensions'][dim]:
   ci=c['dimensions'][dim][key]['ci95'];(compatible if ci[0]<=0<=ci[1] else different).append(dim)
 interpretation=f"Paired differences versus exempted shared22 include zero for {compatible}; exclude zero for {different}. This is detection on these100 prompt/seed pairs, not proof of equivalence." if phase=='D' else 'This is an inference-only means-to-zeros swap on identical trained weights; interpret its paired changes dimension by dimension, not as a new zero-trained control.'
 append(header+scope+table+'\n'+interpretation+f"\n\nTeacher agreement: mean full-coordinate final-latent relative L2 {a['mean_final_latent_relative_l2']:.6f} over100 final-state pairs (step20, no intermediate-step average); frame PSNR {a['mean_finite_frame_psnr_db']:.3f}dB over4500 aligned decoded frames, all RGB pixels/channels per-frame then equal dB average.\n")

lock=(LOG/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 assert read(LOG/'screen-a-status.json')['state']=='complete' and (RESULT/'screen/complete.json').exists(),'A not complete'
 state('B-training',gpu_hours_used=used()/3600)
 train=ROOT/'out/distill-s22-all';attempt=3;assert read(train/'checkpoint/latest.json')['step']==2000;assert not any(read(p)['state'] in ('running','waiting_for_idle_gpus') for p in LOG.glob('train-b*-status.json'))
 base=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node','2','scripts/train_learned_center.py','--stage','2','--out','out/distill-s22-all','--steps','2500','--accum','2','--lr','1e-5','--lr-center','2e-4','--warmup','250','--lambda-cfg','1','--lambda-feat','0.1','--ema','0.999','--clip','1','--val-every','250','--val-prompts','8','--val-at-start','--save-every','500','--ema-checkpoints','--resume-state-with-ema','--resume-state-every','1000','--seed','20261114','--selection','out/distill-selection/selection.pt','--variant','shared','--calibration','out/distill-selection/pair-statistics.pt','--center-mode','mean','--include-layer0','--include-step1','--traj','/tmp/tri-distill-pilot.EG6asH/traj','--snapshot','/dev/shm/tri-wan-snapshot']
 write(LOG/'training-command.json',base)
 while True:
  attempt+=1;remaining=CAP*3600-used();hard=int((remaining-9000)/2)-120
  if hard<1200:raise RuntimeError('ASK: insufficient remaining training allocation; preserve full checkpoint, request budget/scope decision')
  cmd=base+['--resume']+['--budget-seconds',str(hard-600)]
  name=f'train-b{attempt:02d}';write(LOG/(name+'-request.json'),{'command':cmd,'approved_gpu_hours_total':CAP,'reserved_C_gpu_seconds':9000,'hard_wall_seconds':hard})
  worker=['python3',str(LOG/'gpu-worker.py'),'--root',str(ROOT),'--name',name,'--gpus','2','--candidates','2,3,1,6,5,0','--wall-limit',str(hard),'--',*cmd]
  try:run(worker,name+'-worker.log')
  except subprocess.CalledProcessError:
   log=(LOG/(name+'.log')).read_text();cp=train/'checkpoint/latest.json'
   if ('SIGKILL' not in log and 'SIGTERM' not in log) or not cp.exists() or attempt>=5:raise
   state('B-resuming-after-external-signal',resume_step=read(cp)['step'],gpu_hours_used=used()/3600);continue
  if read(train/'checkpoint/latest.json')['step']==2500:break
  raise RuntimeError('ASK: budget guard saved a partial training checkpoint; full recipe needs budget/scope decision')
 collect_training();publish('No-exemption B: 2500 updates, all-active mask and resumable EMA audit')
 for phase in ('D','E'):
  state('C-'+('means' if phase=='D' else 'zeros'),gpu_hours_used=used()/3600)
  run([PYTHON,'-u',str(LOG/'benchmark.py'),phase],'phase-'+phase.lower()+'-driver.log')
  collect_benchmark(phase)
  if phase=='E':collect_screen(ROOT/'out/distill-noexempt/screen-trained',RESULT/'screen-trained')
 # All four inference-only swaps in one comparison, retaining each model's own trained means reference.
 dims=('aesthetic_quality','imaging_quality','overall_consistency','subject_consistency')
 swaps=[('Shared22 exempted',ROOT/'distill-results/swap/full100/s22_zeros-comparisons.json','s22_zeros_minus_shared22'),('Fixed high',ROOT/'distill-results/swap/full100/fh_zeros/fh_zeros-comparisons.json','fh_zeros_minus_student'),('Split42',ROOT/'distill-results/nope/full100/split42_zeros/paired-comparisons.json','split42_zeros_minus_split42'),('Shared22 no exemptions',RESULT/'full100/s22_all_zeros/paired-comparisons.json','s22_all_zeros_minus_s22_all_means')]
 table='| Swap | Aesthetics | Imaging | Text-video | Subject |\n|---|---:|---:|---:|---:|\n'
 for name,path,key in swaps:
  d=read(path)['dimensions'];table+='| '+name+' | '+' | '.join(f"{d[x][key]['mean']:+.6f} [{d[x][key]['ci95'][0]:+.6f},{d[x][key]['ci95'][1]:+.6f}]" for x in dims)+' |\n'
 screen=read(RESULT/'screen-trained/complete.json');write(RESULT/'budget-summary.json',{'approved_gpu_hours':CAP,'gpu_hours_used':used()/3600,'cap_history':'13 approved originally; raised to 18 by the user on 21 Nov 2026 after the external SIGTERM','accounting':'All allocated worker wall seconds times GPU count, including A, all B attempts and C screen/parity/generation/scorers; CPU-only work excluded.'});assert used()<=CAP*3600
 append('### No-exemption study complete\n\nAll four swaps, each on100 original prompt/seed pairs with10000 paired prompt resamples seed20261109; decoded-video scores with within-video frame scopes stated above:\n\n'+table+'\nTrained-student screen (32 prompts x20 teacher states, whole-latent ratios, both CFG branches combined at5): '+json.dumps({k:v['mean'] for k,v in screen['arms'].items()})+f". Columns conditional/unconditional/guided. Total allocated study GPU-hours {used()/3600:.6f}/{CAP} (cap raised from 13 by the user after the external SIGTERM). Full2500 optimizer/sampler/RNG/EMA state and all prior outputs preserved. Evidence under distill-results/noexempt/. No additional recipe or zeros-trained arm was run.")
 publish('Complete no-exemption shared22 A-C and paired inference-only zero swap','done');state('complete',gpu_hours_used=used()/3600)
except Exception as e:
 state('needs-attention',error=str(e),gpu_hours_used=used()/3600,traceback=traceback.format_exc())
 shutil.copyfile(LOG/'pipeline-status.json',RESULT/'interruption-resume.json')
 train=ROOT/'out/distill-s22-all'
 if train.exists():
  out=RESULT/'interrupted-training-resume';out.mkdir(exist_ok=True)
  for n in ('args.json','history.jsonl','validation.jsonl'):
   if (train/n).exists():shutil.copyfile(train/n,out/n)
 append('### No-exemption runner stopped for attention\n\n'+str(e)+f'. Allocated GPU-hours {used()/3600:.6f}/{CAP}. All outputs preserved; see distill-results/noexempt/interruption.json and out/distill-noexempt-run/ logs. No further GPU work is launched by the runner.')
 try:publish('No-exemption study stopped: '+str(e)[:100],'question' if 'ASK:' in str(e) else 'interrupted')
 except Exception:state('needs-attention-publication-failed',error=traceback.format_exc(),gpu_hours_used=used()/3600)
 raise
