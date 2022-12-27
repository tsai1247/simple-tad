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
    float* viterbi = new float[num_observation * 3]();

    // initialize viterbi
    for (std::size_t i = 0; i < 3; ++i) {
        viterbi[i] = initial[i] * emission_func(emission, observations[0], i);
    }

    // run viterbi for t > 0
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < 3; ++i) {
            float max_prob = 0;
            for (std::size_t j = 0; j < 3; ++j) {
                float prob = viterbi[(t - 1) * 3 + j] * transition[j * 3 + i];
                if (prob > max_prob) {
                    max_prob = prob;
                }
            }
            viterbi[t * 3 + i] = max_prob * emission_func(emission, observations[t], i);
        }
    }

    // find the most probable state and its backtrack
    float max_prob = 0;
    int max_prob_index = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        if (viterbi[(num_observation - 1) * 3 + i] > max_prob) {
            max_prob = viterbi[(num_observation - 1) * 3 + i];
            max_prob_index = i;
        }
    }

    hidden_states[num_observation - 1] = max_prob_index;
    for (int t = num_observation - 2; t >= 0; --t) {
        float max_prob = 0;
        int max_prob_index = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            float prob = viterbi[t * 3 + i] * transition[i * 3 + hidden_states[t + 1]];
            if (prob > max_prob) {
                max_prob = prob;
                max_prob_index = i;
            }
        }
        hidden_states[t] = max_prob_index;
    }

    delete[] viterbi;
    return hidden_states;
}
}