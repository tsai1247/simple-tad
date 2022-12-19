#include "lib/di.hpp"
#include "lib/baum_welch.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    auto& [data, edge_size] = read_hi_c_data("./data/GM12878_MboI_Diag_chr6.csv", 5000, 305000, 170085000, 825000, 170605000);
    auto end_read_data = std::chrono::high_resolution_clock::now();
    auto di = calculate_di_AVX2(data, edge_size, 5000);
    auto end_calculate_di = std::chrono::high_resolution_clock::now();

    // std::cout << "di: " << std::endl;
    // for (std::size_t i = 0; i < edge_size; ++i) {
    //     if (di[i] != 0)
    //         std::cout << "di[" << i << "]: " << di[i] << std::endl;
    // }

    std::cout << "Read data: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_read_data - start).count() << "ns" << std::endl;
    std::cout << "Calculate di: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_calculate_di - end_read_data).count() << "ns" << std::endl;

    std::vector<float> di_vec(edge_size, 0);
    for (std::size_t i = 0; i < edge_size; ++i) {
        if (di[i] != 0)
            di_vec[i] = di[i];
    }

    std::vector<int> di_discrete;
    std::transform(di_vec.begin(), di_vec.end(), std::back_inserter(di_discrete),
        [](const auto& idx) {
            if (idx >= 0.4) return 1;
            else if (idx <= - 0.4) return 2;
            else return 0;
        });

    std::cout << "Start descrete Baum-Welch algorithm..." << std::endl;
    auto start_em = std::chrono::high_resolution_clock::now();
    baum_welch(di_discrete);
    auto end_em = std::chrono::high_resolution_clock::now();

    std::cout << "Baum-Welch: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_em - start_em).count() << std::endl;

    delete[] data;
    delete[] di;

    return 0;
}