#include "di.hpp"
#include <algorithm>
#include <iostream>
#include <string>

int main() {
    auto data = read_hi_c_data("./data/GM12878_MboI_Diag_chr6.csv");

    std::cout << "Number of data points: " << data.size() << std::endl;

    // print head 10 rows
    for (int i = 0; i < std::min((int) data.size(), 10); i++) {
        std::cout << data[i].bin1 << " " << data[i].bin2
                  << " " << data[i].rescaled_intensity << " " << data[i].diag_offset
                  << " " << data[i].dist << std::endl;
    }

    return 0;
}