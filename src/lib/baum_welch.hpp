#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>

#define TOLERANCE 1e-7
#define MAX_ITERS 1000

float inline log_add(float x, float y) {
    if (y <= x) {
        return x + std::log1p(std::exp(y - x));
    } else {
        return y + std::log1p(std::exp(x - y));
    }
}

float* forward(
    const int* observations,
    const std::size_t num_observations,
    const float* transition,
    const float* emission,
    const float* initial,
    const std::size_t num_states) {
    // init the alpha matrix
    float* alpha = new float[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        alpha[i] = initial[i] + emission[i * num_states + observations[0]];
    }

    // compute the alpha matrix
    for (std::size_t t = 1; t < num_observations; ++t) {
        // now is observation t, and t is from 1 because we skip init observation
        for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
            for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
                float p = alpha[(t - 1) * num_states + prev_state] + transition[prev_state * num_states + curr_state] + emission[curr_state * num_states + observations[t]];
                alpha[t * num_states + curr_state] = log_add(alpha[t * num_states + curr_state], p);
            }
        }
    }

    return alpha;
}

float* backward(
    const int* observations,
    const std::size_t num_observations,
    const float* transition,
    const float* emission,
    const std::size_t num_states) {
    // init the beta matrix
    float* beta = new float[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        beta[(num_observations - 1) * num_states + i] = 0; // log(1)
    }

    // compute the beta matrix
    for (std::size_t t = num_observations - 2;; --t) {
        // now is observation t, and t is from (num_observations - 2) because we already init last observation
        for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
            for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
                float p = beta[(t + 1) * num_states + curr_state] + transition[prev_state * num_states + curr_state] + emission[curr_state * num_states + observations[t + 1]];
                beta[t * num_states + prev_state] = log_add(beta[t * num_states + prev_state], p);
            }
        }

        if (t == 0) {
            break;
        }
    }

    return beta;
}

void baum_welch(
    const int* observations,
    const std::size_t num_observations,
    float* initial,
    float* transition,
    float* emission,
    const std::size_t num_states) {

    // transform initial to log space
    float* log_initial = new float[num_states];
    for (std::size_t i = 0; i < num_states; ++i) {
        log_initial[i] = std::log1p(initial[i]);
    }

    // transform transition matrix and emission matrix to log space
    float* log_transition = new float[num_states * num_states];
    float* log_emission = new float[num_states * num_states];
    for (std::size_t i = 0; i < num_states * num_states; ++i) {
        log_transition[i] = std::log1p(transition[i]);
        log_emission[i] = std::log1p(emission[i]);
    }

    // init the likelihood of the observed data
    float log_likelihood = -INFINITY;

    std::size_t iter = 0;
    while (true) {
        auto alpha = forward(observations, num_observations, log_transition, log_emission, log_initial, num_states);
        auto beta = backward(observations, num_observations, log_transition, log_emission, num_states);

        float new_log_likelihood = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            new_log_likelihood = log_add(new_log_likelihood, alpha[(num_observations - 1) * num_states + i]);
        }

        // check if the log likelihood is converged
        if (std::abs(std::exp(new_log_likelihood) - std::exp(log_likelihood)) < TOLERANCE) {
            std::cout << "Converged at iteration " << iter << std::endl;

            delete[] alpha;
            delete[] beta;
            break;
        }

        // update the log likelihood
        log_likelihood = new_log_likelihood;

        // check if the max iteration is reached
        if (iter >= MAX_ITERS) {
            std::cout << "Max iteration reached." << std::endl;
            
            delete[] alpha;
            delete[] beta;
            break;
        }

        /*  calculate gamma and xi
         *      gamma[t][i]: in t (index of observations) the sum of probability of passing state i
         *      xi[t][i][j]: in t (index of observations) the sum of probability of path from i (index=t) to j (index=t+1, obmit here)
         */
        float* gamma = new float[num_states * num_observations];
        float* xi = new float[num_states * num_states * num_observations];

        std::fill_n(gamma, num_states * num_observations, -INFINITY);
        std::fill_n(xi, num_states * num_states * num_observations, -INFINITY);

        for (std::size_t t = 0; t < num_observations; ++t) {
            float denominator_g = -INFINITY;
            float denominator_x = -INFINITY;

            for (std::size_t i = 0; i < num_states; ++i) {
                denominator_g = log_add(denominator_g, alpha[t * num_states + i] + beta[t * num_states + i] - log_likelihood);

                if (t < num_observations - 1) {
                    for (std::size_t j = 0; j < num_states; ++j) {
                        denominator_x = log_add(denominator_x, alpha[t * num_states + i] + log_transition[i * num_states + j] + log_emission[j * num_states + observations[t + 1]] + beta[(t + 1) * num_states + j] - log_likelihood);
                    }
                }
            }

            for (std::size_t i = 0; i < num_states; ++i) {
                gamma[t * num_states + i] = alpha[t * num_states + i] + beta[t * num_states + i] - denominator_g;

                if (t < num_observations - 1) {
                    for (std::size_t j = 0; j < num_states; ++j) {
                        xi[t * num_states * num_states + i * num_states + j] = alpha[t * num_states + i] + log_transition[i * num_states + j] + log_emission[j * num_states + observations[t + 1]] + beta[(t + 1) * num_states + j] - denominator_x;
                    }
                }
            }
        }

        delete[] alpha;
        delete[] beta;

        // update initial
        float initial_sum = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            log_initial[i] = log_add(log_initial[i], gamma[i]);
            initial_sum = log_add(initial_sum, log_initial[i]);
        }

        // update initial marginalize
        for (std::size_t i = 0; i < num_states; ++i) {
            log_initial[i] -= initial_sum;
        }

        // update transition matrix and emission matrix
        for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
            float gamma_sum = -INFINITY; // sum of all probability that change from prev_state to curr_state before num_observations
            for (std::size_t t = 0; t < num_observations - 1; ++t) {
                gamma_sum = log_add(gamma_sum, gamma[t * num_states + prev_state]);
            }

            // update transition matrix
            for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
                float xi_sum = -INFINITY;
                for (std::size_t t = 0; t < num_observations - 1; ++t) {
                    xi_sum = log_add(xi_sum, xi[t * num_states * num_states + prev_state * num_states + curr_state]);
                }

                log_transition[prev_state * num_states + curr_state] = xi_sum - gamma_sum;
            }

            // update emission matrix
            float* p = new float[num_states]; // sum of all probability that emit k (= observations[t]) from prev_state in num_observations
            std::fill_n(p, num_states, -INFINITY);

            for (std::size_t t = 0; t < num_observations; ++t) {
                p[observations[t]] = log_add(p[observations[t]], gamma[t * num_states + prev_state]);
            }

            for (std::size_t k = 0; k < num_states; ++k) {
                log_emission[prev_state * num_states + k] = p[k] - gamma_sum;
            }

            delete[] p;
        }

        delete[] gamma;
        delete[] xi;

        ++iter;
    }

    // transform initial from log to normal
    for (std::size_t i = 0; i < num_states; ++i) {
        initial[i] = std::exp(log_initial[i]);
    }

    // transform transition matrix and emission matrix from log to normal
    for (std::size_t i = 0; i < num_states * num_states; ++i) {
        transition[i] = std::exp(log_transition[i]);
        emission[i] = std::exp(log_emission[i]);
    }

    delete[] log_initial;
    delete[] log_transition;
    delete[] log_emission;
}