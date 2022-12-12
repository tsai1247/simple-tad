#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "constants.h"

const std::tuple<double*, std::size_t>
read_hi_c_data( const std::string& filename,\
                const std::size_t& bin_size,\
                const std::size_t& bin1_min,\
                const std::size_t& bin1_max,\
                const std::size_t& bin2_min,\
                const std::size_t& bin2_max) {
    std::fstream file;
    file.open(filename, std::ios::in);

    // check if file is open
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::size_t edge_size = std::max((bin1_max - bin1_min), (bin2_max - bin2_min)) / bin_size + 1;

    double* data = new double[edge_size * edge_size];
    std::string line;

    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string chr;
        std::size_t bin1;
        std::size_t bin2;
        double rescaled_intensity;
        std::size_t diag_offset;
        std::size_t dist;

        std::getline(ss, chr, ',');
        ss >> bin1;
        ss.ignore();
        ss >> bin2;
        ss.ignore();
        ss >> rescaled_intensity;
        ss.ignore();
        ss >> diag_offset;
        ss.ignore();
        ss >> dist;

        if (bin1 >= bin1_min && bin1 <= bin1_max && bin2 >= bin2_min && bin2 <= bin2_max) {
            std::size_t row = (bin1 - bin1_min) / bin_size;
            std::size_t col = (bin2 - bin2_min) / bin_size;
            data[row * edge_size + col] = rescaled_intensity;
        } else {
            std::cout << "chr: " << chr << " bin1: " << bin1 << " bin2: " << bin2 << " rescaled_intensity: " << rescaled_intensity << " diag_offset: " << diag_offset << " dist: " << dist << std::endl;
        }
    }
    return std::make_tuple(data, edge_size);
}

std::vector<double> calculate_di(const double* contact_matrix, const std::size_t& edge_size, const std::size_t& bin_size) {
    std::size_t range = SIGNIFICANT_BINS / bin_size;
    std::vector<double> di(edge_size, 0);

    for (std::size_t locus_index = 0; locus_index < edge_size; locus_index++) {
        double A;
        double B;
        if (locus_index < range) {
            // edge case
            A = std::accumulate(contact_matrix + locus_index * edge_size, contact_matrix + locus_index * edge_size + locus_index, 0.0);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + locus_index * edge_size + locus_index + range + 1, 0.0);
        } else if (locus_index >= edge_size - range) {
            // edge case
            A = std::accumulate(contact_matrix + locus_index * edge_size + locus_index - range, contact_matrix + locus_index * edge_size + locus_index, 0.0);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + (locus_index + 1) * edge_size, 0.0);
        } else {
            // normal case
            A = std::accumulate(contact_matrix + locus_index * edge_size + locus_index - range, contact_matrix + locus_index * edge_size + locus_index, 0.0);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + locus_index * edge_size + locus_index + range + 1, 0.0);
        }

        double E = (A + B) / 2;

        if (A == 0 && B == 0) {
            di[locus_index] = 0;
        } else {
            try {
                di[locus_index] = ((B - A) / (std::abs(B - A))) * ((((A - E) * (A - E)) / E) + (((B - E) * (B - E)) / E));
            } catch (std::exception& e) {
                di[locus_index] = 0;
            }
        }
    }
    return di;
}
