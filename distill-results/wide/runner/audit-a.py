import sys, json, torch
sys.path.insert(0, 'scripts')
from learned_center import LearnedCenter
new = torch.load('out/distill-selection-v4/selection.pt', map_location='cpu')
old = torch.load('out/distill-selection/selection.pt', map_location='cpu')
out = {}
s_new, s_old = new['variants']['shared'], old['variants']['shared']
out['shared_masks_tensor_equal_to_original'] = all(torch.equal(s_new[k], s_old[k]) for k in ('mask_q', 'mask_k'))
w = new['variants']['wide']
assert w['wide'] is True
means = w['mask_q'].sum(-1); content = w['mask_nope'].sum(-1)
out['wide_flag'] = bool(w['wide'])
out['wide_means_pairs_per_cell'] = {'min': int(means.min()), 'max': int(means.max())}
out['wide_k_equals_q_means'] = bool(torch.equal(w['mask_q'], w['mask_k']))
out['wide_content_pairs_per_cell'] = {'min': int(content.min()), 'max': int(content.max())}
out['cells'] = int(content.numel())
c = LearnedCenter.from_checkpoint(torch.load('out/distill-selection-v4/static-wide/learned_center.pt', map_location='cpu'))
out['static_wide'] = {'is_wide': c.is_wide, 'content_pairs': c.content_pairs, 'layers_active': int(c.layer_active.sum()), 'of_layers': c.layers,
    'steps_active': int(c.step_active.sum()), 'of_steps': c.steps, 'content_index_shape': list(c.content_index.shape),
    'content_index_matches_mask': bool(torch.equal(torch.zeros_like(c.mask_nope).scatter_(-1, c.content_index, True), c.mask_nope))}
# content pairs should be the highest-damage pairs: min damage in content >= max damage outside, per cell
d = new['damage_q'] + new['damage_k']
inside = torch.where(w['mask_nope'], d, torch.full_like(d, float('inf'))).min(-1).values
outside = torch.where(~w['mask_nope'], d, torch.full_like(d, -float('inf'))).max(-1).values
out['content_is_top32_by_damage_q_plus_k_every_cell'] = bool((inside >= outside).all())
assert out['shared_masks_tensor_equal_to_original'] and out['wide_means_pairs_per_cell'] == {'min': 64, 'max': 64}
assert out['wide_content_pairs_per_cell'] == {'min': 32, 'max': 32} and c.is_wide and c.content_pairs == 32
assert c.layer_active.all() and c.step_active.all() and out['static_wide']['content_index_matches_mask']
out['scope'] = 'Per layer-step-branch-head cell (30 x 20 x 2 x 12 = 14400 cells), pair-level masks (64 rotary pairs per head); damage = selection criterion pooled over the 64 calibration prompts; no exemptions.'
json.dump(out, open('out/distill-wide-run/selection-audit.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
print(json.dumps(json.load(open('out/distill-selection-v4/selection-report.json'))['wide'], indent=1))
