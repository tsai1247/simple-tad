#include "lib/baum_welch.hpp"
#include "lib/coord.hpp"
#include "lib/di.hpp"
#include "lib/viterbi.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#define DISCREATE_THRESHOLD 0.4

namespace py = pybind11;

PYBIND11_MODULE(simple_tad, m) {
    m.doc() = "A C++ implementation of calculating TADs.";

    m.def(
        "calculate_tad_coords", [](py::array_t<float> data_array, std::size_t edge_size, std::size_t bin_size) {
            py::buffer_info data_buf = data_array.request();
            float* data = static_cast<float*>(data_buf.ptr);

            auto di = calculate_di_AVX2(data, edge_size, bin_size);
            int* di_discrete = new int[edge_size];

            for (std::size_t i = 0; i < edge_size; i++) {
                if (di[i] > DISCREATE_THRESHOLD) {
                    di_discrete[i] = 1;
                } else if (di[i] < -DISCREATE_THRESHOLD) {
                    di_discrete[i] = 2;
                } else {
                    di_discrete[i] = 0;
                }
            }

            float* initial = new float[3] { 0.4, 0.3, 0.3 };
            float* transition = new float[3 * 3] {
                0.7, 0.2, 0.1,
                0.1, 0.6, 0.3,
                0.2, 0.3, 0.5
            };
            float* emission = new float[3 * 3] {
                0.7, 0.2, 0.1,
                0.1, 0.6, 0.3,
                0.2, 0.1, 0.7
            };

            baum_welch(di_discrete, edge_size, initial, transition, emission, 3, 3); // side effect: update initial, transition, emission

            auto states = scalar::viterbi(di_discrete, edge_size, initial, transition, emission);

            auto coords = calculate_coord(reinterpret_cast<BiasState*>(states), edge_size);

            delete[] di;
            delete[] di_discrete;

            delete[] initial;
            delete[] transition;
            delete[] emission;

            delete[] states;

            return coords;
        },
        py::arg("data"), py::arg("edge_size"), py::arg("bin_size"), "Calculate TADs from Hi-C data.");
}
