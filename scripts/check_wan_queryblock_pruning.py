"""Compare production scoring and gathering with pinned official LongLive code."""

import importlib.util
import json
from pathlib import Path
import sys

import torch
from triattention_wan_queryblock_pruning import temporal_coefficients, importance_scores, select_indices


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261109)
    spec = importlib.util.spec_from_file_location("official_pruning", "/dev/shm/tri-center-setup/kv_compression.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    device = torch.device("cuda")
    grid = (12, 30, 30)
    key = torch.randn(1, 10800, 12, 128, device=device)
    mean = torch.randn(12, 64, 2, device=device)
    abs_mean = mean.norm(dim=-1) + torch.rand(12, 64, device=device)
    cfg = official.KVCompressionConfig(stats_path=Path("unused"), budget_tokens=5400, compress_every_n_frames=3,
        keep_last_frames=3, frame_seq_length=900, pruning_mode="layer_perhead", grid_h=30, grid_w=30, sink_size=1)
    compressor = official.LongLiveKVCompressor(cfg)
    compressor.q_mean = torch.view_as_complex(mean.contiguous())[None]
    compressor.q_abs_mean = abs_mean[None]
    compressor._init_rotary_metadata(device, 64)
    positions = torch.arange(10800, device=device)
    frame, height, width = compressor._pos_to_grid(positions)
    angle = torch.cat((frame[:, None] * compressor.omega[None, :22],
                       height[:, None] * compressor.omega[None, 22:43],
                       width[:, None] * compressor.omega[None, 43:]), -1)
    pairs = torch.view_as_complex(key.reshape(1, 10800, 12, 64, 2).contiguous())
    rotated = torch.view_as_real(pairs * torch.polar(torch.ones_like(angle), angle)[None, :, None]).flatten(-2)
    cache = [{"k": rotated, "v": torch.randn_like(rotated), "local_end_index": torch.tensor(10800, device=device), "global_end_index": torch.tensor(10800, device=device)}]
    compressor._sync_position_state(cache, 1, 12, device)
    rows = []
    maximum_error = 0.0
    normalized_error = 0.0
    for query_start in (0, 3, 6, 9):
        query_end = query_start + 3
        compressor.offsets = torch.arange(-3, 0, device=device).float()
        reference = compressor._compute_layer_head_scores(cache, 10800, query_end, dynamic_offset=0)
        scores = importance_scores(key.flatten(-2), temporal_coefficients(mean, 12, device, query_start, query_end), abs_mean - mean.norm(dim=-1), grid)
        maximum_error = max(maximum_error, float((reference - scores).abs().max()))
        normalized_error = max(normalized_error, float(((reference - scores).abs() / reference.std(-1, keepdim=True)).max()))
        assert normalized_error < 1e-4
        protected = torch.cat((torch.arange(900, device=device), torch.arange(query_start * 900, query_end * 900, device=device))).unique()
        mask = torch.ones(10800, device=device, dtype=torch.bool)
        mask[protected] = False
        candidates = torch.arange(10800, device=device)[mask]
        for ratio in (0.5, 0.75):
            selected = select_indices(scores, round(10800 * ratio), grid, torch.Generator().manual_seed(0), query_start, query_end)
            assert selected.shape == (1, 12, round(10800 * ratio))
            assert torch.all(selected.diff(dim=-1) > 0)
            assert torch.isin(protected, selected[0, 0]).all()
            for head in range(12):
                assert torch.isin(torch.arange(query_start * 900, query_end * 900, device=device), selected[0, head]).all()
            gather = selected.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, 128)
            retained = rotated.gather(1, gather)
            retained_value = cache[0]["v"].gather(1, gather)
            for head in range(12):
                assert torch.equal(retained[0, :, head], rotated[0, selected[0, head], head])
                assert torch.equal(retained_value[0, :, head], cache[0]["v"][0, selected[0, head], head])
            count = selected.shape[-1] - protected.numel()
            expected_top = reference[..., candidates].topk(count, dim=-1).indices
            actual_top = scores[..., candidates].topk(count, dim=-1).indices
            overlap = [float(torch.isin(actual_top[0, h], expected_top[0, h]).float().mean()) for h in range(12)]
            assert min(overlap) > 0.999
            rows.append({"query_frames": [query_start, query_end], "retention": ratio, "tokens_per_head": selected.shape[-1],
                         "protected_union_tokens": protected.numel(), "ranked_tokens": count,
                         "minimum_official_topk_overlap": min(overlap), "all_live_query_keys_protected": True, "actual_kv_coordinates_bit_exact": True})
    error = maximum_error
    root = Path("out/triattention-pruning-wan-queryblocks-20261109")
    result = {"official_commit": "a4bc3c8f709db60f016ef42c3feb290fd0c00c1b",
              "scope": "Synthetic10800token keys,12heads,64pairs,one layer. Compare factored score to unmodified official LongLive scorer; three actual query-frame offsets and norm correction, all four query blocks.",
              "max_score_abs_error": error, "max_score_error_in_head_std_units": normalized_error, "retention_checks": rows}
    (root / "selection-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
