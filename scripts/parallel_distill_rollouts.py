"""Split a Stage 3 refresh while keeping the original two training ranks."""

from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import torch.distributed as dist

from full_sweep import write_json


@contextmanager
def reserve_spare_pair(lock_dir, eligible):
    """Lease idle devices against this experiment's other workers."""
    with ExitStack() as stack:
        devices = []
        if eligible:
            training = set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
            rows = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits',
            ], text=True, timeout=10)
            for row in rows.splitlines():
                index, uuid, memory = (field.strip() for field in row.split(','))
                if uuid in training or index in training or int(memory) >= 32:
                    continue
                lock = stack.enter_context((lock_dir / f'gpu-{index}.lock').open('w'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                compute = subprocess.check_output([
                    'nvidia-smi', '-i', uuid, '--query-compute-apps=pid', '--format=csv,noheader',
                ], text=True, timeout=10)
                if compute.strip():
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    continue
                devices.append(uuid)
                if len(devices) == 2:
                    break
        if len(devices) != 2:
            stack.close()
            devices = []
        yield devices


def worker_seconds(path):
    if not path.exists():
        return 0
    return sum(json.loads(row)['charged_gpu_seconds'] for row in path.read_text().splitlines())


def split_rollout_jobs(jobs, args, rank, world, refresh, out, run_local, export_snapshot):
    """Generate the original jobs on up to four devices with a shared EMA export.

    The two FSDP ranks always run equal numbers of forwards. If either spare
    worker fails, both training ranks recapture their assigned half in order.
    A full worker timeout is charged before starting children so interruption
    cannot leave GPU time outside the budget ledger.
    """
    if world != 2 or not 2 <= len(jobs) <= 32 or not args.rollout_worker_budget:
        run_local(jobs)
        return
    ledger = out / 'rollout-workers.jsonl'
    remaining = args.rollout_worker_budget - worker_seconds(ledger)
    worker_timeout = 330
    kill_grace = 5
    allocation = 2 * (worker_timeout + kill_grace)
    eligible = rank == 0 and remaining >= allocation
    with reserve_spare_pair(args.rollout_lock_dir, eligible) as devices:
        plan = [devices]
        dist.broadcast_object_list(plan, src=0)
        if not plan[0]:
            run_local(jobs)
            return
        midpoint = (len(jobs) + 1) // 2
        head, tail = jobs[:midpoint], jobs[midpoint:]
        tails = [None] * world
        dist.all_gather_object(tails, tail)
        staging = [None]
        if rank == 0:
            staging[0] = tempfile.mkdtemp(prefix='rollout-refresh-', dir=out)
        dist.broadcast_object_list(staging, src=0)
        staging_dir = Path(staging[0])
        export_snapshot(staging_dir)
        procs = []
        success = [True]
        try:
            if rank == 0:
                with ledger.open('a') as stream:
                    stream.write(json.dumps(dict(refresh=refresh, devices=plan[0], charged_gpu_seconds=allocation)) + '\n')
                for index, (uuid, subset) in enumerate(zip(plan[0], tails)):
                    manifest = staging_dir / f'jobs-{index}.json'
                    write_json(manifest, subset)
                    command = [
                        'timeout', '--signal=TERM', f'--kill-after={kill_grace}s', f'{worker_timeout}s',
                        sys.executable, '-u', str(Path(__file__).with_name('capture_distill_rollouts.py')),
                        '--student', str(staging_dir / 'student'), '--snapshot', args.snapshot,
                        '--traj', str(Path(args.traj).resolve()), '--jobs', str(manifest),
                        '--out', str(staging_dir / 'states'), '--guidance', str(args.guidance),
                    ]
                    worker_log = out / f'rollout-{refresh:04d}-worker-{index}.log'
                    with worker_log.open('a') as log:
                        procs.append(subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid),
                                                      stdout=log, stderr=subprocess.STDOUT))
            started = time.monotonic()
            run_local(head)
            if rank == 0:
                statuses = [proc.wait() for proc in procs]
                expected = [job['name'] for subset in tails for job in subset]
                success[0] = not any(statuses) and all((staging_dir / 'states' / f'{name}.pt').is_file() for name in expected)
                if success[0]:
                    for name in expected:
                        file = f'{name}.pt'
                        (staging_dir / 'states' / file).replace(out / 'onpolicy' / file)
                print(f'SPLIT ROLLOUT refresh={refresh} success={success[0]} elapsed={time.monotonic()-started:.2f}', flush=True)
            dist.broadcast_object_list(success, src=0)
            if not success[0]:
                run_local(tail)
        finally:
            if rank == 0:
                for proc in procs:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
                shutil.rmtree(staging_dir)
