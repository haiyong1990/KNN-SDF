try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
import glob
import os


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    #extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_compile_args=['-std=c++11', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'im2mesh/utils/libmcubes/mcubes.pyx',
        'im2mesh/utils/libmcubes/pywrapper.cpp',
        'im2mesh/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)


# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)



#### setup chamferdist
package_name = 'chamferdist'
version = '0.3.0'
requirements = [
    'Cython',
    'torch>1.1.0',
]
long_description = 'A pytorch module to compute Chamfer distance \
                    between two point sets (pointclouds).'

setup(
    name='chamferdist',
    version=version,
    description='Pytorch Chamfer distance',
    long_description=long_description,
    requirements=requirements,
    ext_modules=[
        CUDAExtension('chamferdistcuda', [
            'chamferdist/chamfer_cuda.cpp',
            'chamferdist/chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


### build pointnet2
try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True
import pointnet2

_ext_src_root = "pointnet2/_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["etw_pytorch_utils==1.1.1", "h5py", "pprint", "enum34", "future"]

setup(
    name="pointnet2",
    version=pointnet2.__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


## build pvconv
_src_path = "pvcnn/modules/functional/"
_ext_sources=[os.path.join(_src_path,'src', f) for f in [
    'ball_query/ball_query.cpp',
    'ball_query/ball_query_gpu.cu',
    'grouping/grouping.cpp',
    'grouping/grouping_gpu.cu',
    'interpolate/neighbor_interpolate.cpp',
    'interpolate/neighbor_interpolate_gpu.cu',
    'interpolate/trilinear_devox.cpp',
    'interpolate/trilinear_devox_gpu.cu',
    'interpolate/nn_devox.cpp',
    'interpolate/nn_devox_gpu.cu',
    'sampling/sampling.cpp',
    'sampling/sampling_gpu.cu',
    'voxelization/vox.cpp',
    'voxelization/vox_gpu.cu',
    'bindings.cpp',
]]
_ext_headers=[os.path.join(_src_path,'src', f) for f in [
    'ball_query/',
    'grouping/',
    'interpolate/',
    'interpolate/',
    'sampling/',
    'voxelization/',
    './',
]]
requirements = [
    'Cython',
    'torch>1.1.0',
]

setup(
    name='_pvcnn_backend',
    version='v1.0',
    author="pvconv",
    packages=find_packages(),
#    extra_cflags=['-O3', '-std=c++11'],
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pvcnn.modules.functional._pvcnn_backend",
            sources=_ext_sources,
            include_dirs = _ext_headers,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++11"],
                "nvcc": ["-O3", "-std=c++11"],
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

