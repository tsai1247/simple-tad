#include "benchmark/benchmark.h"
#include "viterbi.hpp"
#include "viterbi_simd.hpp"
#include "di.hpp"
#include <algorithm>
#include <iostream>

double* calculate_di_SCALAR(const double* contact_matrix, const std::size_t& edge_size, const std::size_t& range) {
    double* di = new double[edge_size]();

    for (std::size_t locus_index = 0; locus_index < edge_size; ++locus_index) {
        double A;
        double B;

        if (locus_index < range) {
            // edge case
            A = std::accumulate(contact_matrix + locus_index * edge_size, contact_matrix + locus_index * edge_size + locus_index, 0.0f);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + locus_index * edge_size + locus_index + range + 1, 0.0f);
        } else if (locus_index >= edge_size - range) {
            // edge case
            A = std::accumulate(contact_matrix + locus_index * edge_size + locus_index - range, contact_matrix + locus_index * edge_size + locus_index, 0.0f);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + (locus_index + 1) * edge_size, 0.0f);
        } else {
            // normal case
            A = std::accumulate(contact_matrix + locus_index * edge_size + locus_index - range, contact_matrix + locus_index * edge_size + locus_index, 0.0f);
            B = std::accumulate(contact_matrix + locus_index * edge_size + locus_index + 1, contact_matrix + locus_index * edge_size + locus_index + range + 1, 0.0f);
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

static void BM_calculate_di_SCALAR(benchmark::State& state) {
    std::size_t edge_size = state.range(0);

    std::vector<double> data(edge_size * edge_size, 0);
    // fill random positive doubles
    std::generate(data.begin(), data.end(), []() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); });

    for (auto _ : state) {
        calculate_di_SCALAR(data.data(), edge_size, 2);
    }

    state.counters["Throughput"] = benchmark::Counter(state.iterations() * edge_size * edge_size * sizeof(double) / 8, benchmark::Counter::kIsRate);
}

static void BM_calculate_di_AVX2(benchmark::State& state) {
    std::size_t edge_size = state.range(0);

    std::vector<double> data(edge_size * edge_size, 0);
    // fill random positive doubles
    std::generate(data.begin(), data.end(), []() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); });

    for (auto _ : state) {
        calculate_di_AVX2(data.data(), edge_size, 40);
    }

    state.counters["Throughput"] = benchmark::Counter(state.iterations() * edge_size * edge_size * sizeof(double) / 8, benchmark::Counter::kIsRate);
}

double* random_probability_list(std::size_t& size)
{
    double* data = new double[size]();
    std::generate(data, data + size, []() { return static_cast<double>(rand()%1000) / static_cast<double>(rand()%1000) ; });
    double sum = 0;
    for(std::size_t i=0; i<size; i++)
        sum += data[i];

    for(std::size_t i=0; i<size; i++)
        data[i] /= sum;
    return data;
}

static void BM_calculate_viterbi_SCALAR(benchmark::State& state) {
    std::size_t edge_size = state.range(0);

    int* observations = new int[edge_size]();
    std::generate(observations, observations + edge_size, []() { return static_cast<int>(rand()%3); });
    
    std::size_t num_observations = edge_size;    
    
    std::size_t num_hiddenstate = 3;
    
    double* initial = random_probability_list(num_hiddenstate);
    
    double* transition = new double[num_hiddenstate*num_hiddenstate]();
    for(std::size_t i=0; i<num_hiddenstate; i++)
    {
        double* tmp = random_probability_list(num_hiddenstate);
        for(std::size_t j=0; j<num_hiddenstate; j++)
            transition[i*3+j] = tmp[j];
    }

    std::size_t num_emission = 3;
    double* emission = new double[num_hiddenstate*num_emission]();
    for(std::size_t i=0; i<num_hiddenstate; i++)
    {
        double* tmp = random_probability_list(num_emission);
        for(std::size_t j=0; j<num_emission; j++)
            emission[i*num_emission+j] = tmp[j];
    }

    for (auto _ : state) {
        scalar::viterbi(observations, num_observations, initial, transition, emission, num_emission, num_hiddenstate);
    }

    state.counters["Throughput"] = benchmark::Counter(state.iterations() * edge_size * sizeof(int) / 8, benchmark::Counter::kIsRate);
}

static void BM_calculate_viterbi_SIMD(benchmark::State& state) {
    std::size_t edge_size = state.range(0);

    int* observations = new int[edge_size]();
    std::generate(observations, observations + edge_size, []() { return static_cast<int>(rand()%3); });
    
    std::size_t num_observations = edge_size;
    
    std::size_t num_hiddenstate = 3;
    
    double* initial = random_probability_list(num_hiddenstate);
    
    double* transition = new double[num_hiddenstate*num_hiddenstate]();
    for(std::size_t i=0; i<num_hiddenstate; i++)
    {
        double* tmp = random_probability_list(num_hiddenstate);
        for(std::size_t j=0; j<num_hiddenstate; j++)
            transition[i*3+j] = tmp[j];
    }

    std::size_t num_emission = 3;
    double* emission = new double[num_hiddenstate*num_emission]();
    for(std::size_t i=0; i<num_hiddenstate; i++)
    {
        double* tmp = random_probability_list(num_emission);
        for(std::size_t j=0; j<num_emission; j++)
            emission[i*num_emission+j] = tmp[j];
    }

    for (auto _ : state) {
        vectorized::viterbi(observations, num_observations, initial, transition, emission, num_emission, num_hiddenstate);
    }

    state.counters["Throughput"] = benchmark::Counter(state.iterations() * edge_size * sizeof(int) / 8, benchmark::Counter::kIsRate);
}

BENCHMARK(BM_calculate_di_SCALAR)->Arg(33957);
BENCHMARK(BM_calculate_di_AVX2)->Arg(33957);

BENCHMARK(BM_calculate_viterbi_SCALAR)->Arg(33957);
BENCHMARK(BM_calculate_viterbi_SIMD)->Arg(33957);

BENCHMARK_MAIN();