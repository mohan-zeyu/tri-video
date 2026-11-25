#!/usr/bin/env python3
"""Part 1 driver: profile one optimizer update of the no-exemption shared22 Stage 2 recipe (DISTILL_PROFILE_PROMPT.md).

    python scripts/profile_update.py --stage host      # host.json (CPU only)
    python scripts/profile_update.py --stage pair      # P0, P1, P2, P4a-d on one leased pair of idle GPUs, in order
    python scripts/profile_update.py --stage single    # P5 then P3 on one leased idle GPU
    python scripts/profile_update.py --stage summary   # statistics, checks, README and figure -> distill-results/profile

Every arm writes into a new directory out/distill-profile/<arm>/ (the trainer refuses an existing non-empty one);
telemetry (nvidia-smi every 0.5 s, all GPUs) and host snapshots before/after go to the same directory after the arm.
The pair arms hold one two-GPU lease throughout, so they use the same two cards. GPU leasing (idle only: no compute
process, 0% utilisation, < 32 MiB, on four reads over 45 s, exclusive lock) and the ledger come from center_study.py.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_steps import Hybrid

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'out' / 'distill-profile'
SELECTION = '/mnt/data/liuzeyu/tri-video-selection-67c101b/out/distill-selection'
RECIPE = ['--stage', '2', '--steps', '2500', '--accum', '2', '--lr', '1e-5', '--lr-center', '2e-4', '--warmup', '250',
          '--lambda-cfg', '1.0', '--lambda-feat', '0.1', '--ema', '0.999', '--clip', '1.0', '--seed', '20261114',
          '--selection', SELECTION + '/selection.pt', '--variant', 'shared', '--calibration', SELECTION + '/pair-statistics.pt',
          '--center-mode', 'mean', '--include-layer0', '--include-step1',
          '--traj', '/tmp/tri-distill-pilot.EG6asH/traj', '--prompts', 'configs/distill_prompts.json', '--snapshot', '/dev/shm/tri-wan-snapshot']
PROFILE = ['--profile-warmup', '5', '--profile-updates', '30']


def with_flag(recipe, flag, value):
    out = list(recipe)
    out[out.index(flag) + 1] = value
    return out


PAIR_ARMS = [  # (arm, recipe, profile flags)
    ('P0', RECIPE, ['--profile-timers', 'none']),
    ('P1', RECIPE, ['--profile-timers', 'blocks']),
    ('P2', RECIPE, ['--profile-timers', 'phases', '--profile-torch-updates', '3']),
    ('P4a', with_flag(RECIPE, '--lambda-feat', '0'), ['--profile-timers', 'blocks']),
    ('P4b', with_flag(RECIPE, '--ema', '0'), ['--profile-timers', 'blocks']),
    ('P4c', with_flag(RECIPE, '--accum', '1'), ['--profile-timers', 'blocks']),
    ('P4d', RECIPE + ['--no-checkpointing'], ['--profile-timers', 'blocks']),
]
TELEMETRY = 'timestamp,index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.max.sm,clocks_event_reasons.active'
ARM_WALL_LIMIT = 1500


class Profile(Hybrid):
    def __init__(self, budget_hours):
        super().__init__(budget_hours)
        self.p.run = WORK / 'run'
        self.p.ledger = self.p.run / 'ledger.tsv'
        self.p.active = self.p.run / 'active'
        self.p.status = self.p.run / 'status.json'

    def sh(self, *cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        except Exception as error:
            return 'unavailable: %r' % error

    def host(self):
        py = self.lab['python']
        versions = self.sh(py, '-c', 'import json,torch,diffusers,transformers;print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,'
                                      '"nccl":".".join(map(str,torch.cuda.nccl.version())),"cudnn":torch.backends.cudnn.version(),'
                                      '"diffusers":diffusers.__version__,"transformers":transformers.__version__}))')
        gpus = self.sh('nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,'
                       'pcie.link.width.current,pcie.link.width.max,power.limit', '--format=csv')
        meminfo = {l.split(':')[0]: l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines()[:3]}
        traj = '/tmp/tri-distill-pilot.EG6asH/traj'
        result = {'scope': 'lab host at the time of the Part 1 run; PCIe link generation is the current (idle cards may down-train) and max',
                  'gpus_csv': gpus, 'topology': self.sh('nvidia-smi', 'topo', '-m'), 'cpu': self.sh('lscpu'),
                  'numa': self.sh('numactl', '--hardware'), 'ram': meminfo, 'versions': json.loads(versions or '{}'),
                  'trajectory_filesystem': self.sh('findmnt', '-T', traj, '-o', 'TARGET,SOURCE,FSTYPE,SIZE,USED', '-J'),
                  'trajectory_df': self.sh('df', '-h', traj),
                  'sdpa_backend': 'recorded by P5 (forward-anatomy.json, field sdpa)'}
        WORK.mkdir(parents=True, exist_ok=True)
        self.write(WORK / 'host.json', result)
        print('HOST', json.dumps(result['versions']), flush=True)

    def lease(self, n):
        while True:
            got = self.try_lease(n)
            if got:
                return got
            self.state('waiting-for-idle-gpus', gpus=n)
            time.sleep(30)

    def run_arm(self, arm, argv, lease, ok_codes=(0,)):
        dest = WORK / arm
        if (dest / 'done.json').exists():
            print('SKIP', arm, 'done', flush=True)
            return
        if dest.exists():
            raise RuntimeError('ASK: %s exists without done.json (an earlier attempt); not overwriting it' % dest)
        if self.used() + 2 * 600 > self.budget:
            raise RuntimeError('ASK: Part 1 guard of %.2f GPU-hours would be exceeded before %s' % (self.budget / 3600, arm))
        stage = self.p.run / arm
        stage.mkdir(parents=True, exist_ok=True)
        (stage / 'smi-before.txt').write_text(self.sh('nvidia-smi'))
        tele = subprocess.Popen(['nvidia-smi', '--query-gpu=' + TELEMETRY, '--format=csv', '-lms', '500'],
                                stdout=(stage / 'gpu-telemetry.csv').open('w'), stderr=subprocess.DEVNULL)
        job = self.start(arm, argv, lease)
        rc = None
        try:
            while rc is None:
                rc = job['proc'].poll()
                if rc is None and time.time() - job['started'] > ARM_WALL_LIMIT:
                    rc = self.stop(job) or 124
                time.sleep(2)
        finally:
            seconds = time.time() - job['started']
            with self.p.ledger.open('a') as f:
                f.write('%s\t%s\t%.3f\t%d\n' % (arm, ','.join(str(g[0]) for g in lease), seconds, rc if rc is not None else -1))
            (self.p.active / (arm + '.json')).unlink(missing_ok=True)
            job['log'].close()
            tele.terminate()
            tele.wait(timeout=30)
        (stage / 'smi-after.txt').write_text(self.sh('nvidia-smi'))
        oom = list(dest.glob('oom-rank*.json')) if dest.exists() else []
        if rc not in ok_codes and not oom:
            raise RuntimeError('%s exited %s, see %s' % (arm, rc, self.p.run / (arm + '.log')))
        dest.mkdir(parents=True, exist_ok=True)
        for f in stage.iterdir():
            shutil.move(str(f), dest / f.name)
        stage.rmdir()
        shutil.copyfile(self.p.run / (arm + '.log'), dest / 'stdout.log')
        self.write(dest / 'done.json', {'arm': arm, 'gpus': [g[0] for g in lease], 'gpu_uuids': [g[1] for g in lease], 'exit_code': rc,
                                        'allocated_seconds': seconds, 'gpu_seconds': seconds * len(lease), 'out_of_memory': bool(oom),
                                        'argv': argv})
        print('ARM', arm, 'rc', rc, 'seconds %.1f' % seconds, 'oom' if oom else '', flush=True)

    def pair(self):
        lease = self.lease(2)
        started = time.time()
        try:
            for arm, recipe, flags in PAIR_ARMS:
                argv = [self.lab['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
                        'scripts/train_learned_center.py', *recipe, '--out', str(WORK / arm), '--profile', arm, *PROFILE, *flags]
                self.run_arm(arm, argv, lease, ok_codes=(0,))
        finally:
            arms = sum(float(l.split('\t')[2]) for l in self.p.ledger.read_text().splitlines()
                       if l.split('\t')[0] in {a for a, _, _ in PAIR_ARMS}) if self.p.ledger.exists() else 0.0
            gap = max(0.0, time.time() - started - arms)
            with self.p.ledger.open('a') as f:
                f.write('pair-lease-gaps\t%s\t%.3f\t0\n' % (','.join(str(g[0]) for g in lease), gap))
            for g in lease:
                g[2].close()

    def single(self):
        lease = self.lease(1)
        started = time.time()
        try:
            self.run_arm('P5', [self.lab['python'], '-u', 'scripts/profile_forward_anatomy.py', '--traj', '/tmp/tri-distill-pilot.EG6asH/traj',
                                '--snapshot', '/dev/shm/tri-wan-snapshot', '--out', str(WORK / 'P5')], lease)
            self.run_arm('P3', [self.lab['python'], '-u', 'scripts/train_learned_center.py', *RECIPE, '--no-fsdp', '--out', str(WORK / 'P3'),
                                '--profile', 'P3', '--profile-floor', *PROFILE, '--profile-timers', 'blocks'], lease)
        finally:
            arms = sum(float(l.split('\t')[2]) for l in self.p.ledger.read_text().splitlines() if l.split('\t')[0] in ('P5', 'P3'))
            with self.p.ledger.open('a') as f:
                f.write('single-lease-gaps\t%s\t%.3f\t0\n' % (lease[0][0], max(0.0, time.time() - started - arms)))
            lease[0][2].close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=('host', 'pair', 'single', 'summary'), required=True)
    ap.add_argument('--budget-gpu-hours', type=float, required=True, help='Part 1 guard: the 6 GPU-hour cap minus Part 2 use')
    a = ap.parse_args()
    p = Profile(a.budget_gpu_hours)
    p.p.run.mkdir(parents=True, exist_ok=True)
    lock = (p.p.run / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit('another profile_update runner holds %s' % (p.p.run / 'runner.lock'))
    if a.stage == 'host':
        p.host()
    elif a.stage == 'pair':
        p.pair()
    elif a.stage == 'single':
        p.single()
    else:
        import summarize_update_profile
        summarize_update_profile.main(WORK, ROOT / 'distill-results' / 'profile')
    print('USED GPU-hours %.4f' % (p.used() / 3600), flush=True)


if __name__ == '__main__':
    main()
