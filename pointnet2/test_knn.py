import pointnet2
from pointnet2.utils import pointnet2_utils as pn2utils # import KNN_query
import torch
import numpy as np
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


a = torch.rand((12, 10240, 3), dtype=torch.float, device="cuda")
b = torch.rand((12, 1024, 3), dtype=torch.float, device="cuda")
t_start = time.time()
idxs = pn2utils.furthest_point_sample(b, 16)
b_ref = pn2utils.gather_operation(b.transpose(1,2).contiguous(), idxs).transpose(1,2) #[B, 16, 3]
assert(b_ref.size()[1] == 16)
radius = torch.min(torch.norm(a.unsqueeze(dim=2) - b_ref.unsqueeze(dim=1), dim=-1), dim=-1)[0] #[B,N]
idxs = pn2utils.KNN_query(8, radius, a, b).float()
idxs = torch.sort(idxs)[0]
print(idxs.size())
print("time: ", time.time() - t_start)


dists = torch.norm(a.unsqueeze(dim=2) - b.unsqueeze(dim=1), dim=-1) #[B, N, M]
idxs_top = torch.topk(dists, 8, sorted=True, largest=False)[1].float()

diff = torch.norm(idxs - idxs_top)

print("results: ", diff.item())
print(idxs[0, :8], idxs_top[0, :8])


