from huggingface_hub import snapshot_download
import concurrent.futures as cf

jobs = {
    'cog': ('zai-org/CogVideoX-2b', ['transformer/*']),
    'wan': ('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', ['transformer/*']),
    'ltx': ('Lightricks/LTX-Video', ['transformer/*']),
}
def dl(kv):
    k, (repo, pats) = kv
    p = snapshot_download(repo, allow_patterns=pats, max_workers=6)
    return k, p
with cf.ThreadPoolExecutor(3) as ex:
    for k, p in ex.map(dl, jobs.items()):
        print(k, 'DONE ->', p, flush=True)
