#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <iostream>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h>
#include <sstream>

namespace py = pybind11;

double system_energy(py::array_t<double> x_coords, py::array_t<double> y_coords, py::array_t<double> z_coords,
                     double box_length, double cutoff2, double epsilon)

{
    py::buffer_info x_coords_info = x_coords.request();
    py::buffer_info y_coords_info = y_coords.request();
    py::buffer_info z_coords_info = z_coords.request();

    int num_particles = x_coords_info.shape[0];

    // Maybe add checks to make sure arrays are same size

    const double *x_data = static_cast<double *>(x_coords_info.ptr);
    const double *y_data = static_cast<double *>(y_coords_info.ptr);
    const double *z_data = static_cast<double *>(z_coords_info.ptr);

    double e_total = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+ : e_total)
    for (int i = 0; i < num_particles; i++) {
        double i_x = x_data[3 * i];
        double i_y = y_data[3 * i];
        double i_z = z_data[3 * i];

        for (int j = i + 1; j < num_particles; j++) {
            double j_x = x_data[3 * j];
            double j_y = y_data[3 * j];
            double j_z = z_data[3 * j];

            double rijx = j_x - i_x;
            double rijy = j_y - i_y;
            double rijz = j_z - i_z;

            rijx = rijx - box_length * round(rijx / box_length);
            rijy = rijy - box_length * round(rijy / box_length);
            rijz = rijz - box_length * round(rijz / box_length);

            double rij2 = pow(rijx, 2) + pow(rijy, 2) + pow(rijz, 2);

            if (rij2 < cutoff2) {
                double sig_by_r6 = pow((1 / rij2), 3);
                double sig_by_r12 = pow(sig_by_r6, 2);
                double lj_pot = 4.0 * (sig_by_r12 - sig_by_r6) * epsilon;
                e_total += lj_pot;
            }
        }
    }

    return e_total;
}

double pair_energy(int par_num, py::array_t<double> x_coords, py::array_t<double> y_coords,
                   py::array_t<double> z_coords, double box_length, double cutoff2, double epsilon)

{
    py::buffer_info x_coords_info = x_coords.request();
    py::buffer_info y_coords_info = y_coords.request();
    py::buffer_info z_coords_info = z_coords.request();

    int num_particles = x_coords_info.shape[0];

    // Maybe add checks to make sure arrays are same size

    const double *x_data = static_cast<double *>(x_coords_info.ptr);
    const double *y_data = static_cast<double *>(y_coords_info.ptr);
    const double *z_data = static_cast<double *>(z_coords_info.ptr);

    double e_pair = 0.0;
    double i_x = x_data[3 * par_num];
    double i_y = y_data[3 * par_num];
    double i_z = z_data[3 * par_num];

#pragma omp parallel for schedule(static) reduction(+ : e_pair)
    for (int j = 0; j < num_particles; j++) {
        if (par_num != j) {
            double j_x = x_data[3 * j];
            double j_y = y_data[3 * j];
            double j_z = z_data[3 * j];

            double rijx = j_x - i_x;
            double rijy = j_y - i_y;
            double rijz = j_z - i_z;

            rijx = rijx - box_length * round(rijx / box_length);
            rijy = rijy - box_length * round(rijy / box_length);
            rijz = rijz - box_length * round(rijz / box_length);

            double rij2 = pow(rijx, 2) + pow(rijy, 2) + pow(rijz, 2);

            if (rij2 < cutoff2) {
                double sig_by_r6 = pow((1 / rij2), 3);
                double sig_by_r12 = pow(sig_by_r6, 2);
                double lj_pot = 4.0 * (sig_by_r12 - sig_by_r6);
                e_pair += lj_pot;
            }
        }
    }

    return e_pair;
}

py::array_t<double> rdf(double delta_r, py::array_t<double> gr, py::array_t<double> x_coords,
                        py::array_t<double> y_coords, py::array_t<double> z_coords, double box_length, double cutoff2)

{
    py::buffer_info x_coords_info = x_coords.request();
    py::buffer_info y_coords_info = y_coords.request();
    py::buffer_info z_coords_info = z_coords.request();
    py::buffer_info gr_info = gr.request();

    int num_particles = x_coords_info.shape[0];
    int bins = gr_info.shape[0];
    //  std :: cout << gr << std :: endl;
    double half_box_length2 = pow(box_length / 2., 2);
    double rij;

    if (x_coords_info.ndim != 1) throw std::runtime_error("x_coords is not an one dimensional array");
    if (y_coords_info.ndim != 1) throw std::runtime_error("y_coords is not an one dimensional array");
    if (z_coords_info.ndim != 1) throw std::runtime_error("z_coords is not an one dimensional array");
    if (x_coords_info.shape[0] != y_coords_info.shape[0] or x_coords_info.shape[0] != z_coords_info.shape[0])
        throw std::runtime_error("not same length");

    const double *x_data = static_cast<double *>(x_coords_info.ptr);
    const double *y_data = static_cast<double *>(y_coords_info.ptr);
    const double *z_data = static_cast<double *>(z_coords_info.ptr);
    double *gr_data = static_cast<double *>(gr_info.ptr);

    //#pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < num_particles; m++) {
        double i_x = x_data[3 * m];
        double i_y = y_data[3 * m];
        double i_z = z_data[3 * m];

        for (int n = m + 1; n < num_particles; n++) {
            double j_x = x_data[3 * n];
            double j_y = y_data[3 * n];
            double j_z = z_data[3 * n];

            double rijx = j_x - i_x;
            double rijy = j_y - i_y;
            double rijz = j_z - i_z;

            rijx = rijx - box_length * round(rijx / box_length);
            rijy = rijy - box_length * round(rijy / box_length);
            rijz = rijz - box_length * round(rijz / box_length);

            double rij2 = pow(rijx, 2) + pow(rijy, 2) + pow(rijz, 2);
            if (rij2 < half_box_length2) {
                double rij = sqrt(rij2);
                int bin_number = (int)(rij / delta_r);
                //#pragma omp critical
                //{
                gr_data[bin_number] += 2;

                //  }
            }
        }
    }

    return gr;
}
