#include "lib/di.hpp"
#include "lib/baum_welch.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

#define DISCREATE_THRESHOLD 0.4

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    auto& [data, edge_size] = read_hi_c_data("./data/GM12878_MboI_Diag_chr6.csv", 5000, 305000, 170085000, 825000, 170605000);
    auto end_read_data = std::chrono::high_resolution_clock::now();

    auto di = calculate_di_AVX2(data, edge_size, 5000);
    auto end_calculate_di = std::chrono::high_resolution_clock::now();

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

    std::cout << "Start descrete Baum-Welch algorithm..." << std::endl;

    auto start_em = std::chrono::high_resolution_clock::now();
    
    baum_welch(di_discrete, edge_size, initial, transition, emission, 3);
    auto end_em = std::chrono::high_resolution_clock::now();

    std::cout << "Estimated initial probability:" << std::endl;
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << std::setw(10) << initial[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Estimated transition matrix:" << std::endl;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(10) << transition[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Estimated emission matrix:" << std::endl;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(10) << emission[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "---" << std::endl;

    delete[] data;
    delete[] di;
    delete[] di_discrete;

    delete[] initial;
    delete[] transition;
    delete[] emission;

    std::cout << "Read data: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_read_data - start).count() << "ns" << std::endl;
    std::cout << "Calculate di: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_calculate_di - end_read_data).count() << "ns" << std::endl;
    std::cout << "Baum-Welch: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_em - start_em).count() << "ns" << std::endl;

    return 0;
}