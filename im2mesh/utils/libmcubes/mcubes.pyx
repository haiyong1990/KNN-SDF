
# distutils: language = c++
# cython: embedsignature = True

# from libcpp.vector cimport vector
import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()

cdef extern from "pywrapper.h":
    cdef object c_marching_cubes "marching_cubes"(np.ndarray, double) except +
    cdef object c_marching_cubes2 "marching_cubes2"(np.ndarray, double) except +
    cdef object c_marching_cubes3 "marching_cubes3"(np.ndarray, double) except +
    cdef object c_marching_cubes_voxels "marching_cubes_voxels"(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, double) except +
    cdef object c_marching_cubes_func "marching_cubes_func"(tuple, tuple, int, int, int, object, double) except +

def marching_cubes(np.ndarray volume, float isovalue):
    
    verts, faces = c_marching_cubes(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes2(np.ndarray volume, float isovalue):

    verts, faces = c_marching_cubes2(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes3(np.ndarray volume, float isovalue):

    verts, faces = c_marching_cubes3(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes_func(tuple lower, tuple upper, int numx, int numy, int numz, object f, double isovalue):
    
    verts, faces = c_marching_cubes_func(lower, upper, numx, numy, numz, f, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes_voxels(np.ndarray idx_list, np.ndarray val_list, np.ndarray coord_list, np.ndarray low, np.ndarray upper, float dx, float dy, float dz, float isovalue):
    ### # idx_list: [N,8], val_list: [M,1], coord_list: [N,3]
    # idx_list: [N,8,3], val_list: [N,8], coord_list: [N,3]
    verts, faces = c_marching_cubes_voxels(idx_list, val_list, coord_list, low, upper, dx, dy, dz, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces


