#include "di.hpp"
#include <algorithm>
#include <iostream>
#include <string>

int main() {
    auto data = read_hi_c_data("./data/GM12878_MboI_Diag_chr6.csv", 5000, 305000, 170085000, 825000, 170605000);

    return 0;
}