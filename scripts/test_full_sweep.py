"""Numerical checks against explicit full-vector attention calculations."""
import math
import unittest

import torch

from full_sweep_metrics import analyze, axis_lines, axis_logits, pearson_rows, rotate


class MetricsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.grid = (2,3,4)
        self.h, self.n, self.d = 2,24,12
        self.q = torch.randn(self.h,self.n,self.d)+1
        self.k = torch.randn(self.h,self.n,self.d)+.7
        coords = torch.stack(torch.meshgrid(*(torch.arange(s) for s in self.grid),indexing="ij"),-1).reshape(-1,3)
        self.phase = torch.cat([coords[:,i:i+1]*torch.tensor([[.25,.7]]) for i in range(3)],-1)
        self.c,self.s = self.phase.cos(),self.phase.sin()

    def test_rotation_and_lines(self):
        qr = rotate(self.q,self.c,self.s)
        kr = rotate(self.k,self.c,self.s)
        torch.testing.assert_close(qr.norm(dim=-1),self.q.norm(dim=-1))
        full = qr@kr.transpose(-1,-2)/math.sqrt(self.d)
        ids=torch.arange(self.n).reshape(1,self.n,1)
        for axis in range(3):
            line_ids=axis_lines(ids,self.grid,axis)[0,:,:,0]
            ql,kl=axis_lines(qr,self.grid,axis),axis_lines(kr,self.grid,axis)
            actual=ql@kl.transpose(-1,-2)/math.sqrt(self.d)
            for line,idx in enumerate(line_ids):
                torch.testing.assert_close(actual[:,line],full[:,idx[:,None],idx[None,:]])

    def test_all_pairs_and_origin(self):
        result=analyze(self.q,self.k,self.c,self.s,self.grid,["t"]*2+["h"]*2+["w"]*2)
        for axis,name in enumerate(("t","h","w")):
            item=result["axes"][name]
            assert sum(item["pairs_per_offset"])==self.n*self.grid[axis]
            assert item["total_queries_per_head"]==self.n
            assert all(int(d)//2==axis for d in item["dominant_pair"])
            qr,kr=rotate(self.q,self.c,self.s),rotate(self.k,self.c,self.s)
            cqr=rotate(self.q.mean(1)[:,None].expand_as(self.q),self.c,self.s)
            ckr=rotate(self.k.mean(1)[:,None].expand_as(self.k),self.c,self.s)
            ql,kl=axis_lines(qr,self.grid,axis),axis_lines(kr,self.grid,axis)
            cql,ckl=axis_lines(cqr,self.grid,axis),axis_lines(ckr,self.grid,axis)
            actual=ql@kl.transpose(-1,-2)/math.sqrt(self.d)
            predicted=cql@ckl.transpose(-1,-2)/math.sqrt(self.d)
            brute=[]
            for h in range(self.h):
                rows=[]
                for line in range(ql.shape[1]):
                    for query in range(self.grid[axis]):
                        rows.append(torch.corrcoef(torch.stack([actual[h,line,query],predicted[h,line,query]]))[0,1])
                brute.append(torch.stack(rows).mean())
            torch.testing.assert_close(torch.tensor(item["per_query_r_mean"]),torch.stack(brute),atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(torch.tensor(item["origin_query_measured"]),actual[:,0,0])

    def test_constant_centers_and_undefined(self):
        q=self.q.mean(1)[:,None].expand_as(self.q)
        k=self.k.mean(1)[:,None].expand_as(self.k)
        result=analyze(q,k,self.c,self.s,self.grid,["t"]*2+["h"]*2+["w"]*2)
        for axis in result["axes"].values():
            torch.testing.assert_close(torch.tensor(axis["per_query_r_mean"]),torch.ones(self.h),atol=1e-5,rtol=1e-5)
        assert torch.isnan(pearson_rows(torch.ones(2,4),torch.ones(2,4))).all()


if __name__=="__main__":
    unittest.main()
