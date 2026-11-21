"""CPU position-bias tables for the trained widened head (wide32: 128-dim means under RoPE + 32 content pairs).

Adapted from diagnose_nope_bias.py. In the widened head the position channel is the full 128-dim constant
vector on all 64 rotary pairs, so each table sums all pairs of its axis; the content channel is excluded.

Layers are zero-based; reported denoising steps are one-based. Each table is an
additive, scaled QK-logit contribution from constant pairs on one axis, separately
for every head, layer, step and CFG branch. No token/prompt/step averages are used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from learned_center import LearnedCenter
from full_sweep import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    center_path = args.student / 'learned_center.pt'
    saved = torch.load(center_path, map_location='cpu')
    trained = LearnedCenter.from_checkpoint(saved).eval()
    initial = LearnedCenter.from_checkpoint(saved).eval()
    initial.init_from_calibration(torch.load(args.statistics, map_location='cpu'))
    assert trained.head_dim == 128 and trained.heads == 12 and trained.is_wide and trained.content_pairs == 32
    assert trained.mask_q.all() and trained.mask_k.all()
    assert (trained.selected_pairs('nope').sum(-1) == 32).all()
    assert trained.layer_active.all() and trained.step_active.all()

    # This is the runtime's own rope implementation, constructed on CPU without
    # loading a transformer. Its axis dimensions are 44 / 42 / 42 coordinates.
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    rope = WanRotaryPosEmbed(128, (1, 2, 2), max_seq_len=32, theta=10000.0)
    freqs = torch.cat([10000.0 ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
                       for dim in (44, 42, 42)])
    assert freqs.shape == (64,)
    # Cross-check angular frequencies against native cosine/sine at position 1.
    native = torch.complex(rope.freqs_cos[1, ::2], rope.freqs_sin[1, ::2])
    torch.testing.assert_close(native, torch.exp(1j * freqs).to(native.dtype), rtol=1e-6, atol=1e-7)
    args.out.mkdir(parents=True, exist_ok=True)
    scope = ('Widened-head position channel: scaled additive QK logit contribution (1/sqrt(128)) of the 128-dim '
             'constant Q/K vectors under RoPE, summing all rotary pairs belonging to this axis (t 22, h 21, w 21 pairs); '
             'the 32-pair unrotated content channel is excluded. All 12 heads reported separately for each layer/step/branch. '
             'No token, prompt, step, head or branch averages. Delta is query minus key patch-grid position. '
             'Each table contains its own axis contribution; the total bias is the sum of all three axis tables. '
             'Initial constants pool all-token pre-RoPE post-RMSNorm means equally over 64 calibration prompts; '
             'trained constants come from the final EMA. Layer indices zero-based; denoising steps one-based.')
    index = []
    plots = []
    for layer in (5, 15, 25):
        for step_one_based in (2, 10, 18):
            for branch in (0, 1):
                step = step_one_based - 1
                a = initial.position_bias_tables(layer, step, branch, freqs)
                b = trained.position_bias_tables(layer, step, branch, freqs)
                tables = {}
                for axis, radius in (('t', 11), ('h', 29), ('w', 29)):
                    delta = list(range(-radius, radius + 1))
                    x = torch.stack([a[axis](d) for d in delta], dim=1)
                    y = torch.stack([b[axis](d) for d in delta], dim=1)
                    assert x.shape == y.shape == (12, len(delta))
                    assert x.isfinite().all() and y.isfinite().all()
                    tables[axis] = {'delta': delta, 'head_indices': list(range(12)),
                                    'initial_by_head': x.tolist(), 'trained_ema_by_head': y.tolist()}
                name = f'layer-{layer:02d}_step-{step_one_based:02d}_branch-{branch}.json'
                write_json(args.out / name, {'layer_zero_based': layer, 'step_one_based': step_one_based,
                           'step_zero_based': step, 'branch': branch, 'branch_name': ('conditional', 'unconditional')[branch],
                           'scope': scope, 'tables': tables})
                index.append(name)
                if step_one_based == 10 and branch == 0:
                    plots.append((layer, tables))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row, (layer, tables) in enumerate(plots):
        for col, axis in enumerate(('t', 'h', 'w')):
            data = tables[axis]
            ax = axes[row, col]
            # One individual head is shown, without averaging; JSON keeps all 12.
            ax.plot(data['delta'], data['initial_by_head'][0], label='Calibration', linestyle='--')
            ax.plot(data['delta'], data['trained_ema_by_head'][0], label='Trained EMA')
            ax.set_title(f'Layer {layer}, head 0, {axis} axis')
            ax.set_xlabel('Query − key grid position')
            ax.set_ylabel('Additive scaled QK logit')
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Widened-head position bias (all 64 pairs): denoising step 10, conditional branch\n'
                 'Individual head 0; selected pairs summed within each axis')
    fig.savefig(args.out / 'trained-vs-calibration.png', dpi=150)
    plt.close(fig)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    drift = {}
    for side in ('q', 'k'):
        mask = getattr(trained, 'mask_' + side) & trained.layer_active[:, None, None, None, None] & trained.step_active[None, :, None, None, None]
        x, y = getattr(initial, 'mu_' + side).detach(), getattr(trained, 'mu_' + side).detach()
        drift[side] = {'active_selected_coordinate_count': int(mask.sum()),
                       'relative_l2_drift_active_selected_coordinates': float((y[mask] - x[mask]).norm() / x[mask].norm()),
                       'scope': 'All 14400 layer/step/branch/head cells pooled (nothing exempt); all 128 coordinates per cell, Euclidean parameter-vector ratio of trained EMA minus calibration means over calibration means. No token averaging; each parameter is token-independent.'}
    write_json(args.out / 'index.json', {'scope': scope, 'tables': index, 'rope_theta': 10000,
               'frequencies_by_pair': freqs.tolist(), 'axis_pair_ranges_inclusive': {'t': [0, 21], 'h': [22, 42], 'w': [43, 63]},
               'trained_center_sha256': sha(center_path), 'calibration_sha256': sha(args.statistics), 'drift': drift})
    print('Wrote', len(index), 'tables and comparison figure', flush=True)


if __name__ == '__main__':
    main()
