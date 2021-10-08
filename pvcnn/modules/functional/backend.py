from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

from torch.utils.cpp_extension import load
## see setup.py
from pvcnn.modules.functional import _pvcnn_backend as _backend

# _src_path = os.path.dirname(os.path.abspath(__file__))
# _backend = load(name='_pvcnn_backend',
#                 extra_cflags=['-O3', '-std=c++11'],
#                 sources=[os.path.join(_src_path,'src', f) for f in [
#                     'ball_query/ball_query.cpp',
#                     'ball_query/ball_query.cu',
#                     'grouping/grouping.cpp',
#                     'grouping/grouping.cu',
#                     'interpolate/neighbor_interpolate.cpp',
#                     'interpolate/neighbor_interpolate.cu',
#                     'interpolate/trilinear_devox.cpp',
#                     'interpolate/trilinear_devox.cu',
#                     'sampling/sampling.cpp',
#                     'sampling/sampling.cu',
#                     'voxelization/vox.cpp',
#                     'voxelization/vox.cu',
#                     'bindings.cpp',
#                 ]])


#  ## compile it offline
#  try:
#          import builtins
#  except:
#          import __builtin__ as builtins
#
#          _ext_src_root = "modules/functional/_ext-src"
#          _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
#              "{}/src/*.cu".format(_ext_src_root)
#          )
#          _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
#
#          requirements = ["etw_pytorch_utils==1.1.1", "h5py", "pprint", "enum34", "future"]
#
#          setup(
#              name="_backend",
#              version=pointnet2.__version__,
#              author="xx",
#              packages=find_packages(),
#              install_requires=requirements,
#              ext_modules=[
#                      CUDAExtension(
#                                  name="pointnet2._ext",
#                                  sources=_ext_sources,
#                                  extra_compile_args={
#                                                  "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
#                                                  "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
#                                              },
#                              )
#                  ],
#              cmdclass={"build_ext": BuildExtension},
#          )
#

__all__ = ['_backend']
