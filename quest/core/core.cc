#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <exception>

namespace py = pybind11;

// In mc_functions.cpp
double system_energy(py::array_t<double> x_coords, py::array_t<double> y_coords, py::array_t<double> z_coords,
                     double box_length, double cutoff2, double epsilon);

double pair_energy(int par_num, py::array_t<double> x_coords, py::array_t<double> y_coords,
                   py::array_t<double> z_coords, double box_length, double cutoff2, double epsilon);

py::array_t<double> rdf(double delta_r, py::array_t<double> gr, py::array_t<double> x_coords,
                        py::array_t<double> y_coords, py::array_t<double> z_coords, double box_length, double cutoff2);

// In JK_functions.cpp
void compute_PKJK(py::array_t<double> I, py::array_t<double> D, py::array_t<double> J, py::array_t<double> K);

void compute_DFJK(py::array_t<double> I, py::array_t<double> D, py::array_t<double> J, py::array_t<double> K);


PYBIND11_PLUGIN(core) {
    py::module m("core", "pybind11 QuESt 'core' plugin");

    m.def("compute_PKJK", &compute_PKJK, "A function that can compute the PK J and K matrices.");
    m.def("compute_DFJK", &compute_DFJK, "A function that can compute the J and K matrices using density-fitting.");
    m.def("system_energy", &system_energy, "A function that calculates the total energies");
    m.def("pair_energy", &pair_energy, "Calculates a single atom's interaction with all molecules");
    m.def("rdf", &rdf, "Calculates the RDF for a given system");

    return m.ptr();
}
