#include "lib/di.hpp"
#include "lib/baum_welch_loglik.hpp"
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
    // for (auto d : di) {
    //     std::cout << d << std::endl;
    // }

    std::cout << "Read data: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_read_data - start).count() << "ns" << std::endl;
    std::cout << "Calculate di: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_calculate_di - end_read_data).count() << "ns" << std::endl;

    std::vector<int> di_discrete(0, di.size());
    std::transform(di.begin(), di.end(), di_discrete.begin(),
        [](auto& idx) {
            if (idx >= 0.4) return 1;
            else if (idx <= - 0.4) return 2;
            else return 0;
        });
    baum_welch(di_discrete);

    return 0;
}