#include <exception>
#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

void compute_PKJK(py::array_t<double> I, py::array_t<double> D, py::array_t<double> J, py::array_t<double> K)
{
    // Get the array objects.
    py::buffer_info I_info = I.request();
    py::buffer_info D_info = D.request();
    py::buffer_info J_info = J.request();
    py::buffer_info K_info = K.request();    

    if(I_info.ndim != 4) throw std::runtime_error("I is not a rank-4 tensor!");
    if(D_info.ndim != 2) throw std::runtime_error("D is not a matrix!");
    if(J_info.ndim != 2) throw std::runtime_error("J is not a matrix!");
    if(K_info.ndim != 2) throw std::runtime_error("K is not a matrix!");

    const double * I_data = static_cast<double *>(I_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);
    double * J_data = static_cast<double *>(J_info.ptr);
    double * K_data = static_cast<double *>(K_info.ptr);

    // Get the dimensions.
    size_t dim = D_info.shape[0];
    size_t dim2 = dim*dim;
    size_t dim3 = dim*dim2;

    // Formation of J and K.
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t p = 0; p < dim; p++)
    {
        for(size_t q = 0; q <= p; q++)
        {
            double Jvalue = 0.;
            double Kvalue = 0.;
            for(size_t r = 0; r < dim; r++)
            {
                #pragma omp simd
                for(size_t s = 0; s < r; s++)
                {
                    Jvalue += 2.*I_data[p * dim3 + q * dim2 + r * dim + s] * D_data[r * dim + s];
                }
                Jvalue += I_data[p * dim3 + q * dim2 + r * dim + r] * D_data[r * dim + r];
                #pragma omp simd
                for(size_t i = 0; i < dim; i++)
                {
                    Kvalue += I_data[p * dim3 + r * dim2 + q * dim + i] * D_data[r * dim + i];
                }
            }
            J_data[p * dim + q] = Jvalue;
            J_data[q * dim + p] = Jvalue;
            K_data[p * dim + q]	= Kvalue;
            K_data[q * dim + p]	= Kvalue;            
        }
    }
}

void compute_DFJK(py::array_t<double> I, py::array_t<double> D, py::array_t<double> J, py::array_t<double> K)
{
    // Get the array objects.
    py::buffer_info I_info = I.request();
    py::buffer_info D_info = D.request();
    py::buffer_info J_info = J.request();
    py::buffer_info K_info = K.request();    

    if(I_info.ndim != 3) throw std::runtime_error("I is not a rank-3 tensor!");
    if(D_info.ndim != 2) throw std::runtime_error("D is not a matrix!");
    if(J_info.ndim != 2) throw std::runtime_error("J is not a matrix!");
    if(K_info.ndim != 2) throw std::runtime_error("K is not a matrix!");

    const double * I_data = static_cast<double *>(I_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);
    double * J_data = static_cast<double *>(J_info.ptr);
    double * K_data = static_cast<double *>(K_info.ptr);

    size_t dimD = D_info.shape[0];
    size_t dimIg = I_info.shape[0];

    // Intermediates.
    std::vector<double> kaiP_data(dimIg);
    std::vector<double> zeta_data(dimIg * dimD * dimD);

    // Formation of J.
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t p = 0; p < dimIg; p++)
    {
        double value = 0.;
        for(size_t l = 0; l < dimD; l++)
        {
            #pragma omp simd
            for(size_t s = 0; s < dimD; s++)
            {
                value += I_data[p * dimD * dimD + l * dimD + s]* D_data[l * dimD + s];
            }
        }
        kaiP_data[p] = value;
    }
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t l = 0; l < dimD; l++)
    {
        for(size_t s = l; s < dimD; s++)
        {
            double value = 0.;
            for(size_t p = 0; p < dimIg; p++)
            {
                value += I_data[p * dimD * dimD + l * dimD + s] * kaiP_data[p];
            }
            J_data[l * dimD + s] = value;
            J_data[s * dimD + l] = value;
        }
    }

    // Formation of K.	
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t p = 0; p < dimIg; p++)
    {
        for(size_t l = 0; l < dimD; l++)
        {
            for(size_t m = 0; m < dimD; m++)
            {
                double value = 0.;
                for(size_t s = 0; s < dimD; s++)
                {
                    value += I_data[p * dimD * dimD + l * dimD + s] * D_data[m * dimD + s];
                }
                zeta_data[p * dimD * dimD + l * dimD + m] = value;
            }
        }
    }
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t l = 0; l < dimD; l++)
    {
        for(size_t s = 0; s <= l; s++)
        {
            double value = 0.;
            for(size_t p = 0; p < dimIg; p++)
            {
                for(size_t r = 0; r < dimD; r++)
                {
                    value += I_data[p * dimD * dimD + l * dimD + r] * zeta_data[p * dimD * dimD + s * dimD + r];
                }
            }
            K_data[l * dimD + s] = value;
            K_data[s * dimD + l] = value;
        }
    }
}

PYBIND11_PLUGIN(core) {
    py::module m("core", "pybind11 core plugin");

    m.def("compute_PKJK", &compute_PKJK, "A function that can compute the PK J and K matrices.");
    m.def("compute_DFJK", &compute_PKJK, "A function that can compute the J and K matrices using density-fitting.");

    return m.ptr();
}
