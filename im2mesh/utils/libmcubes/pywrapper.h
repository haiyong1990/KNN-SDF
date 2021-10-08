
#ifndef _PYWRAPPER_H
#define _PYWRAPPER_H

#include <Python.h>
#include "pyarraymodule.h"

#include <vector>

PyObject* marching_cubes(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes2(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes3(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f, double isovalue);

PyObject* marching_cubes_voxels(
        PyArrayObject* idx_arr,
        PyArrayObject* val_arr,
        PyArrayObject* coord_arr,
        PyArrayObject* lower, PyArrayObject* upper,
        float dx, float dy, float dz, double isovalue);

#endif // _PYWRAPPER_H
