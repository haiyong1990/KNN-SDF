import os
import numpy as np
import torch
from im2mesh.utils import libmcubes

def make_3d_grid(bb_min, bb_max, shape, device="cuda"):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0], device=device)
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1], device=device)
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2], device=device)
    # print("xyz: ",shape, pxs*32)

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def points2svoxel(points_in, minv=-1.0, maxv=1.0, res=128, device="cuda"):
    # NOTE: corner in corner_list may have duplicates.
    points_in = torch.tensor(points_in, dtype=torch.float32, device=device)
    points_nearby = []
    th = (maxv - minv)/res
    pointsf = make_3d_grid((minv,)*3, (maxv,)*3, (res,)*3, device)
    Nf = pointsf.size()[0]
    Ni = 10240
    for ii in range(int(ceil(Nf/1.0/Ni))):
        istart = ii * Ni
        iend = min((ii+1)*Ni, Nf)
        dist = torch.norm(pointsf[istart:iend].unsqueeze(dim=1) - points_in.unsqueeze(dim=0), dim=-1)
        dist = torch.min(dist, dim=1)[0]
        points_select = torch.index_select(pointsf[istart:iend], torch.nonzero(dist < th)) # [N1, 3]
        points_nearby.append(points_select)
    coord_list = torch.cat(points_nearby, dim=0) # [N,3]

    offset = [ [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1] ]
    offset = torch.tensor(offset, dtype=torch.float, device=device) * th/2.0 # [8, 3]
    corner_list = coord_list.unsqueeze(dim=1) + offset # [N, 8, 3]

    sdf = [1.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0]
    sdf = torch.tensor(sdf, dtype=torch.float, device=device)
    sdf_list = sdf.expand(corner_list.size()[:2]) # [N,8]
    return coord_list, corner_list, sdf_list


root_path = "../../../data/ShapeNet/02691156/"
device = "cuda"
for f in os.path.listdir(root_path):
    model_path = root_path + f
    if os.path.isdir(model_path):
        print("Processing " + model_path)
        file_path = os.path.join(model_path, "pointcloud.npz")
        pointcloud_dict = np.load(file_path)
        pcd = pointcloud_dict['points'].astype(np.float32) # [N,3]
        normals = pointcloud_dict['normals'].astype(np.float32) # [N,3]

        res = 128
        coord_list, corner_list, sdf_list = points2svoxel(pcd, minv=-0.5, maxv=0.5, res=res, device=device)
        dx, dy, dz = 1.0/res, 1.0/res, 1.0/res
        low = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        upper = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        vertices, triangles = libmcubes.marching_cubes_voxels(
                corner_list, sdf_list, coord_list,
                low, upper, dx, dy, dz, isovalue)
    break



