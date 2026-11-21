import sys, torch
sys.path.insert(0, 'scripts')
from learned_center import LearnedCenter
for path in sys.argv[1:]:
    saved = torch.load(path, map_location='cpu')
    for mode in ('mean', 'delete'):
        old = LearnedCenter(layers=30, steps=20, heads=12, head_dim=128, mode=mode)
        old.load_state_dict(saved['state_dict'])
        new = LearnedCenter.from_checkpoint(saved, mode=mode)
        a, b = old.state_dict(), new.state_dict()
        assert a.keys() == b.keys(), (a.keys() ^ b.keys())
        assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]
        assert old.mode == new.mode and old.is_wide == new.is_wide and old.uses_nope == new.uses_nope
    print('IDENTICAL', path, 'wide', new.is_wide, 'nope', new.uses_nope, 'layers_active', int(new.layer_active.sum()), 'steps_active', int(new.step_active.sum()))
