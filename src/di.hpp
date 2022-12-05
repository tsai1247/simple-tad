#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "hi-c.h"

std::vector<HiC> read_hi_c_data(std::string filename) {
    std::fstream file;
    file.open(filename, std::ios::in);

    // check if file is open
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::vector<HiC> hic_data;

    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string chr;
        int bin1;
        int bin2;
        double rescaled_intensity;
        int diag_offset;
        int dist;

        ss >> chr >> bin1 >> bin2 >> rescaled_intensity >> diag_offset >> dist;

        hic_data.push_back({chr, bin1, bin2, rescaled_intensity, diag_offset, dist});
    }
    return hic_data;
}