import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import im2mesh.common as common
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False, normalize=True):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if len(voxels.shape) == 2 and voxels.shape[0]>0:
        v_raw = voxels
        if normalize:
            # voxels = (voxels - np.min(voxels, axis=0))/np.maximum(np.max(voxels, axis=0) - np.min(voxels, axis=0), 1.0)
            np.clip(voxels+0.5, 0.0, 1.0)
        else:
            voxels = np.clip(voxels, 0.0, 1.0)
        stride = 32
        coord2index = lambda x: np.clip(np.floor(x*stride), 0, stride-1).astype(np.int32)
        indices = coord2index(voxels[:,0])*stride*stride \
            + coord2index(voxels[:,1])*stride + coord2index(voxels[:,2])
        voxels = np.zeros((stride*stride*stride,), dtype=bool)
        # print("unique: ", len(np.unique(indices)), v_raw[:5])
        voxels[indices] = True
        voxels = np.reshape(voxels, (stride, stride, stride))

    if len(voxels.shape) == 3:
        voxels = voxels.transpose(2, 0, 1)
        # ax.voxels(voxels, edgecolor='k')
        ax.voxels(voxels, facecolors=[0.0, 1.0, 0.0, 1.0], edgecolors=[0.0, 0.0, 0.0, 0.3])
    else:
        print("Error shape in voxel visualization: ", voxels.shape, out_file.split("/")[-1])
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def cuboid_data(o, size=(1,1,1), theta=0):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float) # [6, 4, 3]
    # scale
    S = np.array(size).astype(float) # [3,]
    X *= S
    X -= S/2 # [6,4,3]
    # rotate
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1],
                  ], dtype=np.float) # [3,3]
    X = np.matmul(X.reshape((6*4,3)), R.transpose(1,0)).reshape(6,4,3) # [6, 4, 3]
    X += S/2
    X += np.array(o)
    return X

def plotCubeAt(positions, sizes, thetas=None, c='r', alpha=1.0, **kwargs):
    if thetas is None:
        thetas = np.zeros((positions.shape[0], 1), dtype=np.float)
    g = []
    for p,s,t in zip(positions,sizes,thetas):
        g.append( cuboid_data(p, size=s, theta=t) )
    return Poly3DCollection(np.concatenate(g), facecolors=c, alpha=alpha, **kwargs)

def visualize_sparse_voxels(voxels, corners, radius, out_file=None):
    r''' Visualizes voxel data.

    Args:
        voxels (np.ndarray): voxel data, [N,3]
        corners (np.ndarray): corner for each voxel, [N, 8, 3]
        out_file (string): output file
    '''
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    N = voxels.shape[0]
    sizes = np.ones_like(voxels)*radius
    # cubes = plotCubeAt(voxels, sizes, c=[0.0, 1.0, 0.0, 0.4])
    # ax.add_collection3d(cubes)

    ax.scatter(voxels[:, 2:3], voxels[:, 0:1], voxels[:, 1:2], c='g', marker='o', s=1.0)
    corners = corners.reshape((-1, 3))
    ax.scatter(corners[:, 2:3], corners[:, 0:1], corners[:, 1:2], c='r', marker='o', s=4.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, color="y"):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=2.0, c=color)
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def visualize_pointcloud_layer(points, preds, out_file=None):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ratios = [[0.0, 0.2], [0.2,0.4], [0.4,0.5], [0.5, 0.6]]
    clrs = ["g", "c", "r", "b"]
    preds = torch.sigmoid(preds)
    for ii,r in enumerate(ratios):
        mask = ((preds>=1.0-r[1]) & (preds<1.0-r[0]))
        points_sel = points[torch.nonzero(mask).reshape(-1), :].data.cpu().numpy()
        ax.scatter(points_sel[:, 2], points_sel[:, 0], points_sel[:, 1], s=2.0, c=clrs[ii])
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



def visualize_pointcloud_diff(pred_eq_gnd, pred_ge_gnd, gnd_ge_pred, out_file):
    r''' Visualizes point cloud data.

    Args:
        pred_eq_gnd: [N,3]
        pred_ge_gnd: [M,3]
        gnd_ge_pred: [K,3]
    '''
    # Create plot
    color_array = np.array([
        [0.0, 0.0, 0.0, 0.8], # intersection, green
        [0.0, 1.0, 0.0, 0.8], # gnd, yellow
        [1.0, 0.0, 0.0, 0.8], # pred, red
        ])
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(pred_eq_gnd[:, 2], pred_eq_gnd[:, 0], pred_eq_gnd[:, 1], s=2.0, c=color_array[0:1])
    ax.scatter(gnd_ge_pred[:, 2], gnd_ge_pred[:, 0], gnd_ge_pred[:, 1], s=2.0, c=color_array[1:2])
    ax.scatter(pred_ge_gnd[:, 2], pred_ge_gnd[:, 0], pred_ge_gnd[:, 1], s=2.0, c=color_array[2:3])

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def visualise_projection(
        self, points, world_mat, camera_mat, img, output_file='out.png'):
    r''' Visualizes the transformation and projection to image plane.

        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.

    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    '''
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot(
        (pimg2[:, 0] + 1)*image.shape[1]/2,
        (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
    plt.savefig(output_file, bbox_inches='tight')


def visualize_pcd_w_voxel(points, voxels, out_file):
    res = voxels.shape[-1]
    # drawing pcd
    points = (np.asarray(points)+0.5) * res
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=0.5, c="tab:orange", alpha=0.8)

    # drawing voxels
    voxels = np.asarray(voxels)
    assert(len(voxels.shape) == 3)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, facecolors=[0.0, 1.0, 0.0, 0.6], edgecolors=[0.0, 0.0, 0.0, 0.8])
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    #  ax.view_init(elev=30, azim=45)
    #  plt.savefig(out_file)

    for ii in range(1):
        angle = 45 + ii * 90
        ax.view_init(elev=30, azim=angle)
        plt.savefig(out_file + "-%d.png"%ii, bbox_inches='tight')

    ax.view_init(elev=-30, azim=45)
    plt.savefig(out_file + "-5.png", bbox_inches='tight')
    plt.close(fig)


def visualize_pcd_w_voxel_diff(points, voxel_pcd, voxel_gnd, out_file):
    res = voxel_gnd.shape[-1]
    # drawing pcd

    points = (np.asarray(points)+0.5) * res
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=0.5, c="tab:orange", alpha=0.8)

    # drawing voxel
    voxel_inter = np.logical_and(voxel_gnd>0.5, voxel_pcd>0.5)
    voxel_diff_gnd = (voxel_gnd - voxel_inter.astype(np.float32)) > 0.5
    voxel_diff_pcd = (voxel_pcd - voxel_inter.astype(np.float32)) > 0.5

    voxel_inter = voxel_inter.transpose(2, 0, 1)
    voxel_diff_gnd = voxel_diff_gnd.transpose(2, 0, 1)
    voxel_diff_pcd = voxel_diff_pcd.transpose(2, 0, 1)
    ax.voxels(voxel_inter, facecolors=[0.0, 1.0, 0.0, 0.8], edgecolors=[0.0, 0.0, 0.0, 0.8])
    ax.voxels(voxel_diff_gnd, facecolors=[1.0, 1.0, 0.0, 0.8], edgecolors=[0.0, 0.0, 0.0, 0.8])
    ax.voxels(voxel_diff_pcd, facecolors=[1.0, 0.0, 1.0, 0.8], edgecolors=[0.0, 0.0, 0.0, 0.8])

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    #  ax.view_init(elev=30, azim=45)
    #  plt.savefig(out_file)

    for ii in range(1):
        angle = 45 + ii * 90
        ax.view_init(elev=30, azim=angle)
        plt.savefig(out_file + "-%d.png"%ii, bbox_inches='tight')

    # ax.view_init(elev=-30, azim=45)
    # plt.savefig(out_file + "-5.png")
    plt.close(fig)


def visualize_voxel_diff(voxel_pred, voxel_gnd, out_file):
    # drawing voxel
    voxel_inter = np.logical_and(voxel_gnd>0.5, voxel_pred>0.5)
    voxel_diff_gnd = (voxel_gnd - voxel_inter.astype(np.float32)) > 0.5
    voxel_diff_pred = (voxel_pred - voxel_inter.astype(np.float32)) > 0.5

    color_array = np.array([
        [0.0, 1.0, 0.0, 0.8], # intersection, green
        [1.0, 1.0, 0.0, 0.8], # gnd, yellow
        [1.0, 0.0, 1.0, 0.8], # pred, red
        ])
    voxel_inter = voxel_inter.transpose(2, 0, 1)
    voxel_diff_gnd = voxel_diff_gnd.transpose(2, 0, 1)
    voxel_diff_pred = voxel_diff_pred.transpose(2, 0, 1)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.voxels(voxel_inter, facecolors=color_array[0], edgecolors=[0.0, 0.0, 0.0, 0.8])
    ax.voxels(voxel_diff_gnd, facecolors=color_array[1], edgecolors=[0.0, 0.0, 0.0, 0.8])
    ax.voxels(voxel_diff_pred, facecolors=color_array[2], edgecolors=[0.0, 0.0, 0.0, 0.8])

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    plt.savefig(out_file, bbox_inches='tight')


