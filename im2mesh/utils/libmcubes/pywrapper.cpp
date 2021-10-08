
#include "pywrapper.h"

#include "marchingcubes.h"

#include <stdexcept>

struct PythonToCFunc
{
    PyObject* func;
    PythonToCFunc(PyObject* func) {this->func = func;}
    double operator()(double x, double y, double z)
    {
        PyObject* res = PyObject_CallFunction(func, "(d,d,d)", x, y, z); // py::extract<double>(func(x,y,z));
        if(res == NULL)
            return 0.0;
        
        double result = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return result;
    }
};

PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f, double isovalue)
{
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    // Copy the lower and upper coordinates to a C array.
    double lower_[3];
    double upper_[3];
    for(int i=0; i<3; ++i)
    {
        PyObject* l = PySequence_GetItem(lower, i);
        if(l == NULL)
            throw std::runtime_error("error");
        PyObject* u = PySequence_GetItem(upper, i);
        if(u == NULL)
        {
            Py_DECREF(l);
            throw std::runtime_error("error");
        }
        
        lower_[i] = PyFloat_AsDouble(l);
        upper_[i] = PyFloat_AsDouble(u);
        
        Py_DECREF(l);
        Py_DECREF(u);
        if(lower_[i]==-1.0 || upper_[i]==-1.0)
        {
            if(PyErr_Occurred())
                throw std::runtime_error("error");
        }
    }
    
    // Marching cubes.
    mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, PythonToCFunc(f), isovalue, vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    return res;
}

struct PyArrayToCFunc
{
    PyArrayObject* arr;
    PyArrayToCFunc(PyArrayObject* arr) {this->arr = arr;}
    double operator()(int x, int y, int z)
    {
        npy_intp c[3] = {x,y,z};
        return PyArray_SafeGet<double>(arr, c);
    }
};

PyObject* marching_cubes(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    
    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    double lower[3] = {0,0,0};
    double upper[3] = {double(shape[0]-1), double(shape[1]-1), double(shape[2]-1)};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    // Marching cubes.
    mc::marching_cubes<double>(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}

PyObject* marching_cubes2(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");

    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    double lower[3] = {0,0,0};
    double upper[3] = {double(shape[0]-1), double(shape[1]-1), double(shape[2]-1)};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    // Marching cubes.
    mc::marching_cubes2<double>(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);

    return res;
}

PyObject* marching_cubes3(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");

    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    double lower[3] = {0,0,0};
    double upper[3] = {double(shape[0]-1), double(shape[1]-1), double(shape[2]-1)};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    // Marching cubes.
    mc::marching_cubes3<double>(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);

    return res;
}

struct NpArrayIndexCoords
{
    PyArrayObject* m_coord_arr;
    NpArrayIndexCoords(PyArrayObject* coord_arr) 
    {
        this->m_coord_arr = coord_arr;
    }

    void operator()(int idx, double retv[3])
    {
        for(int i=0; i<3; i++)
        {
            npy_intp inds[2] = {idx, i};
            retv[i] = PyArray_SafeGet<float>(m_coord_arr, inds);
        }
    }
};


struct NpArrayIndexCorners
{
    PyArrayObject* m_corner_arr;
    NpArrayIndexCorners(PyArrayObject* corner_arr) 
    {
        this->m_corner_arr = corner_arr;
    }

    void operator()(int idx, double retv[8])
    {
        for(int i=0; i<8; i++)
        {
            npy_intp inds[2] = {idx, i};
            retv[i] = PyArray_SafeGet<float>(m_corner_arr, inds);
        }
    }
};


//WARNING: CORNERS may have duplicateds.
PyObject* marching_cubes_voxels(
        PyArrayObject* idx_arr,
        PyArrayObject* val_arr,
        PyArrayObject* coord_arr,
        PyArrayObject* lower, PyArrayObject* upper,
        float dx, float dy, float dz, double isovalue)
{
    // val_arr: [N, 8], coord_arr: [N, 3] 
    if(PyArray_DIMS(val_arr)[1] != 8 && PyArray_DIMS(coord_arr)[1] != 3)
        throw std::runtime_error("Error dimension: idx(N,8), coord(N,3)");

    // Prepare data.
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    // Copy the lower and upper coordinates to a C array.
    double lower_[3];
    double upper_[3];
    for(int i=0; i<3; ++i)
    {

        npy_intp inds[1] = {i};
        lower_[i] = PyArray_SafeGet<float>(lower, inds);
        upper_[i] = PyArray_SafeGet<float>(upper, inds);
    }
    int numx = int((upper_[0] - lower_[0])/dx);
    int numy = int((upper_[1] - lower_[1])/dx);
    int numz = int((upper_[2] - lower_[2])/dx);
    npy_intp* shape = PyArray_DIMS(val_arr);
    int numv = int(shape[0]);

    // Marching cubes.
    mc::marching_cubes_voxels<double>(
            lower_, numx, numy, numz, numv,
            dx, dy, dz, 
            NpArrayIndexCorners(val_arr), 
            NpArrayIndexCoords(coord_arr), 
            isovalue, vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);

    return res;
}


