"""Compare production scoring and gathering with pinned official LongLive code."""

import importlib.util
import json
from pathlib import Path
import sys

import torch
from triattention_wan_pruning import temporal_coefficients, importance_scores, select_indices


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
    reference = compressor._compute_layer_head_scores(cache, 7200, 12, dynamic_offset=900)
    scores = importance_scores(key.flatten(-2), temporal_coefficients(mean, 12, device), abs_mean - mean.norm(dim=-1), grid)
    actual = scores[..., 900:8100]
    error = float((reference - actual).abs().max())
    normalized_error = float(((reference - actual).abs() / reference.std(-1, keepdim=True)).max())
    assert normalized_error < 1e-4, (error, normalized_error)
    rows = []
    for ratio in (0.5, 0.75):
        selected = select_indices(scores, round(10800 * ratio), grid, torch.Generator().manual_seed(0))
        assert selected.shape == (1, 12, round(10800 * ratio))
        assert torch.all(selected.diff(dim=-1) > 0)
        assert (selected < 900).sum(-1).eq(900).all()
        assert (selected >= 8100).sum(-1).eq(2700).all()
        gather = selected.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, 128)
        retained = rotated.gather(1, gather)
        retained_value = cache[0]["v"].gather(1, gather)
        for head in range(12):
            assert torch.equal(retained[0, :, head], rotated[0, selected[0, head], head])
            assert torch.equal(retained_value[0, :, head], cache[0]["v"][0, selected[0, head], head])
        reference_top = reference.topk(round(10800 * ratio) - 3600, dim=-1).indices
        actual_top = actual.topk(round(10800 * ratio) - 3600, dim=-1).indices
        overlap = [float(torch.isin(actual_top[0, h], reference_top[0, h]).float().mean()) for h in range(12)]
        assert min(overlap) > 0.999, overlap
        rows.append({"retention": ratio, "tokens_per_head": selected.shape[-1], "protected_sink_tokens": 900,
                     "protected_recent_tokens": 2700, "ranked_tokens": selected.shape[-1] - 3600,
                     "minimum_official_topk_overlap": min(overlap), "actual_kv_coordinates_bit_exact": True})
    root = Path("out/triattention-pruning-wan-20261109")
    result = {"official_commit": "a4bc3c8f709db60f016ef42c3feb290fd0c00c1b",
              "scope": "Synthetic10800token keys,12heads,64pairs,one layer. Compare factored score to unmodified official LongLive scorer; all eight logspaced offsets and norm correction.",
              "max_score_abs_error": error, "max_score_error_in_head_std_units": normalized_error, "retention_checks": rows}
    (root / "selection-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
