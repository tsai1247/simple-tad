#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace scalar {

float emission_func(const float* const& emission, const int& di, const int& state) {
    return emission[state * 3 + di];
}

int* viterbi(const int* const& observations, const std::size_t& num_observation, const float* const& initial, const float* const& transition, const float* const& emission) {
    int* hidden_states = new int[num_observation]();
    float* viterbi = new float[3* num_observation]();

    float* transition_log1p = new float[3*3]();
    for(std::size_t i = 0; i < 3*3; ++i) {
        transition_log1p[i] = std::log1p(transition[i]);
    }

    // initialize viterbi
    for (std::size_t i = 0; i < 3; ++i) {
        viterbi[i * num_observation] = std::log1p(initial[i]) + std::log1p(emission_func(emission, observations[0], i));
    }

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {   // current state
            float max = -INFINITY;
            for (std::size_t j = 0; j < 3; ++j) {   // previous state
                float temp = viterbi[j * num_observation + t - 1] + transition_log1p[j * 3 + i];
                if (temp > max) {
                    max = temp;
                }
            }
            viterbi[i * num_observation + t] = max + std::log1p(emission_func(emission, observations[t], i));
        }
    }

    // find the most probable state
    float max = -INFINITY;
    for (std::size_t i = 0; i < 3; ++i) {
        if (viterbi[i * num_observation + num_observation - 1] > max) {
            max = viterbi[i * num_observation + num_observation - 1];
            hidden_states[num_observation - 1] = i;
        }
    }

    // backtrace
    for (int t = num_observation - 2; t >= 0; --t) {
        float max = -INFINITY;
        for (std::size_t j = 0; j < 3; ++j) {   // previous state
            float temp = viterbi[j * num_observation + t] + transition_log1p[j * 3 + hidden_states[t + 1]];
            if (temp > max) {
                max = temp;
                hidden_states[t] = j;
            }
        }
    }

    delete[] viterbi;

    return hidden_states;
}
}