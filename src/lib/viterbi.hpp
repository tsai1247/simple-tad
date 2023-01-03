#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace scalar {

float emission_func(const float* const& emission, const int& di, const int& state) {
    return emission[state * 3 + di];
}

/*
observations: int[num_observation].  The preprocessed di values(0, 1, or 2).
initial: float[3].  The probabilities for init state.
transition: float[3*3].  The probabilities for transition.  transition[i*3+j] presents "from state i to state j".
emission: float[3*3] now.  The discrete probabilities for emission.  emission[state*3+di] presents "emit di from state",
*/
int* viterbi(const int* const& observations, const std::size_t& num_observation, const float* const& initial, const float* const& transition, const float* const& emission) {
    int* hidden_states = new int[num_observation]();    // result of the function.
    float* viterbi = new float[3* num_observation]();   // record probability for each node.
    int* prev_state = new int[3* num_observation]();    // record previous state for current node.

    // calculate log1p for each transition probability.
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
                    prev_state[i * num_observation + t] = j;
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
        hidden_states[t] = prev_state[hidden_states[t + 1] * num_observation + t + 1];
    }

    delete[] viterbi;
    delete[] prev_state;
    delete[] transition_log1p;

    return hidden_states;
}
}