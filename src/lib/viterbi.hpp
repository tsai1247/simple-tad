#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace scalar {

double emission_func(const double* const& emission, const int& di, const int& state, const std::size_t& num_hiddenstate) {
    return emission[state * num_hiddenstate + di];
}

/*
observations: double[num_observation].  The di values.
initial: double[3].  The probabilities for init state.
transition: double[3*3].  The probabilities for transition.  transition[i*3+j] presents "from state i to state j".
emission: double[3*2] now.  The (average, variance) pairs for gaussion emission function.
*/
int* viterbi(const int* const& observations, const std::size_t& num_observation, const double* const& initial, const double* const& transition, const double* const& emission, const std::size_t& num_emission=3, const std::size_t& num_hiddenstate=3) {
    int* hidden_states = new int[num_observation]();    // result of the function.
    double* viterbi = new double[num_hiddenstate* num_observation]();   // record probability for each node.
    int* prev_state = new int[num_hiddenstate* num_observation]();    // record previous state for current node.

    // calculate log for each transition probability.
    double* transition_log = new double[num_hiddenstate*num_hiddenstate]();
    for(std::size_t i = 0; i < num_hiddenstate*num_hiddenstate; ++i) {
        transition_log[i] = std::log(transition[i]);
    }

    // initialize viterbi
    for (std::size_t i = 0; i < num_hiddenstate; ++i) {
        viterbi[i * num_observation] = std::log(initial[i]) + std::log(emission_func(emission, observations[0], i, num_hiddenstate));
    }

    // run viterbi
    for (std::size_t t = 1; t < num_observation; ++t) {
        for (std::size_t i = 0; i < num_hiddenstate; ++i) {   // current state
            double max = -INFINITY;
            for (std::size_t j = 0; j < num_hiddenstate; ++j) {   // previous state
                double temp = viterbi[j * num_observation + t - 1] + transition_log[j * 3 + i];
                if (temp > max) {
                    max = temp;
                    prev_state[i * num_observation + t] = j;
                }
            }
            viterbi[i * num_observation + t] = max + std::log(emission_func(emission, observations[t], i, num_hiddenstate));
        }
    }

    // find the most probable state
    double max = -INFINITY;
    for (std::size_t i = 0; i < num_hiddenstate; ++i) {
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
    delete[] transition_log;

    return hidden_states;
}
}