#include "constants.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

const std::tuple<float*, std::size_t> read_hi_c_data(const std::string& filename, const std::size_t& bin_size, const std::size_t& bin1_min, const std::size_t& bin1_max, const std::size_t& bin2_min, const std::size_t& bin2_max) {
    std::fstream file;
    file.open(filename, std::ios::in);

    // check if file is open
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::size_t edge_size = std::max((bin1_max - bin1_min), (bin2_max - bin2_min)) / bin_size + 1;

    float* data = new float[edge_size * edge_size]();
    std::string line;

    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string chr;
        std::size_t bin1;
        std::size_t bin2;
        float rescaled_intensity;
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

float accumulate_AVX2(const float* data, std::size_t size) {
    float sum = 0;

    std::size_t remain = size % 8;
    size -= remain;

    __m256 sum_vec = _mm256_setzero_ps();
    __m256 data_vec;

    for (std::size_t i = 0; i < size; i += 8) {
        data_vec = _mm256_loadu_ps(data + i);
        sum_vec = _mm256_add_ps(sum_vec, data_vec);
    }

    __m256 swap = _mm256_permute2f128_ps(sum_vec, sum_vec, 0x01);
    sum_vec = _mm256_add_ps(sum_vec, swap);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    _mm256_storeu_ps(&sum, sum_vec);

    for (std::size_t i = size; i < size + remain; ++i) {
        sum += data[i];
    }

    return sum;
}

float* calculate_di_AVX2(const float* contact_matrix, const std::size_t& edge_size, const std::size_t& bin_size) {
    std::size_t range = SIGNIFICANT_BINS / bin_size;
    float* di = new float[edge_size]();

    for (std::size_t locus_index = 0; locus_index < edge_size; ++locus_index) {
        float A;
        float B;
        if (locus_index < range) {
            // edge case
            A = accumulate_AVX2(contact_matrix + locus_index * edge_size, locus_index);
            B = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index + 1, range);
        } else if (locus_index >= edge_size - range) {
            // edge case
            A = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index - range, range);
            B = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index + 1, edge_size - locus_index - 1);
        } else {
            // normal case
            A = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index - range, range);
            B = accumulate_AVX2(contact_matrix + locus_index * edge_size + locus_index + 1, range);
        }

        float E = (A + B) / 2;

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

    for (std::size_t i = 0; i < edge_size; ++i) {
        if (std::isnan(di[i])) {
            di[i] = 0;
        }
    }

    return di;
}
