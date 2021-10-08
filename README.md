This project is the implementation for the paper "Neighborhood-based Neural Implicit Reconstruction from Point Clouds, 3DV2021".

### Setup the Environments.

    conda create env -f environment.yaml
    conda activate knn_sdf
    python setup.py build_ext --inplace


### Dataset Preparation.
Pls. refer to 
[OccupancyNetwork](https://github.com/autonomousvision/occupancy_networks.git). 

### Train the Model.

    python main configs/pointcloud/xxx.yaml

### Test the Model.

    python generate.py configs/pointcloud/xxx.yaml
    python eval_meshes.py configs/pointcloud/xxx.yaml

### References:
We implement the project based on [OccupancyNetwork](https://github.com/autonomousvision/occupancy_networks.git) and [PointNet2](https://github.com/erikwijmans/Pointnet2_PyTorch). Please consider to cite them if you use this repository.


