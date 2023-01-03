#define  _USE_MATH_DEFINES
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
// #include <algorithm>

float inline log_add(float x, float y) {
    if (y <= x) {
        return x + std::log1p(std::exp(y - x));
    } else {
        return y + std::log1p(std::exp(x - y));
    }
}

inline float calc_log_pB(const float& obs, const float& mu, const float& sigma) {
    float log_factor = -std::log1p(sigma * std::sqrt(2 * M_PI));
    return log_factor - ((obs-mu)*(obs-mu)) / (2*sigma*sigma);
}

float* forward(
    const float* observations,
    const std::size_t num_observations,
    const float* transition,
    const float* emission,
    const float* initial,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    // init the alpha matrix
    float* alpha = new float[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        alpha[i * num_observations] = initial[i] + calc_log_pB(observations[0], emission[i * num_emissions], emission[i * num_emissions + 1]); //emission[i * num_emissions + observations[0]];
    }

    // compute the alpha matrix
    for (std::size_t t = 1; t < num_observations; ++t) {
        // now is observation t, and t is from 1 because we skip init observation
        for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
            float sum = -INFINITY;
            for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
                sum = log_add(sum, alpha[prev_state * num_observations + t - 1] + transition[prev_state * num_states + curr_state]);
            }
            alpha[curr_state * num_observations + t] = sum + calc_log_pB(observations[t], emission[curr_state * num_emissions], emission[curr_state * num_emissions + 1]); // emission[curr_state * num_emissions + observations[t]];
        }
    }

    return alpha;
}

float* backward(
    const float* observations,
    const std::size_t num_observations,
    const float* transition,
    const float* emission,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    // init the beta matrix
    float* beta = new float[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        beta[i * num_observations + num_observations - 1] = 0;  // log(1)
    }

    // compute the beta matrix
    for (std::size_t t = num_observations - 1; t < num_observations; --t) {
        // now is observation t, and t is from (num_observations - 2) because we already init last observation
        for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
            beta[prev_state * num_observations + t - 1] = transition[prev_state * num_states + 0] + calc_log_pB(observations[t], emission[0 * num_emissions], emission[0 * num_emissions + 1]);
            for (std::size_t curr_state = 1; curr_state < num_states; ++curr_state) {
                beta[prev_state * num_observations + t - 1] = log_add(beta[prev_state * num_observations + t - 1], transition[prev_state * num_states + curr_state] + calc_log_pB(observations[t], emission[curr_state * num_emissions], emission[curr_state * num_emissions + 1]) + beta[curr_state * num_observations + t]); // emission[curr_state * num_emissions + observations[t + 1]]);
            }
        }
    }

    return beta;
}

float* compute_gamma(
    const float* alpha,
    const float* beta,
    const std::size_t num_observations,
    const std::size_t num_states
) {
    float* gamma = new float[num_states * num_observations]();

    for (std::size_t t = 0; t < num_observations; ++t) {
        float sum = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            gamma[i * num_observations + t] = alpha[i * num_observations + t] + beta[i * num_observations + t];
            sum = log_add(sum, gamma[i * num_observations + t]);
        }

        for (std::size_t i = 0; i < num_states; ++i) {
            gamma[i * num_observations + t] -= sum;
        }
    }

    return gamma;
}

float* compute_xi(
    const float* alpha,
    const float* beta,
    const float* observations,
    const std::size_t num_observations,
    const float* transition,
    const float* emission,
    const std::size_t num_states,
    const std::size_t num_emissions
) {
    float* xi = new float[num_states * num_states * num_observations]();

    for (std::size_t t = 0; t < num_observations - 1; ++t) {
        float sum = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_states; ++j) {
                xi[i * num_states * num_observations + j * num_observations + t] = alpha[i * num_observations + t] + transition[i * num_states + j] + calc_log_pB(observations[t + 1], emission[j * num_emissions], emission[j * num_emissions + 1])/*emission[j * num_emissions + observations[t + 1]]*/ + beta[j * num_observations + t + 1];
                sum = log_add(sum, xi[i * num_states * num_observations + j * num_observations + t]);
            }
        }

        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_states; ++j) {
                xi[i * num_states * num_observations + j * num_observations + t] -= sum;
            }
        }
    }

    return xi;
}

void baum_welch(
    const float* observations,
    const std::size_t num_observations,
    float* initial,
    float* transition,
    float* emission,
    const std::size_t num_states,
    const std::size_t num_emissions,
    const float tolerance = 1e-5,
    const std::size_t max_iters = 100
) {
    // transform initial to log space
    float* log_initial = new float[num_states];
    for (std::size_t i = 0; i < num_states; ++i) {
        log_initial[i] = std::log1p(initial[i]);
    }

    // transform transition matrix and emission matrix to log space
    float* log_transition = new float[num_states * num_states];
    for (std::size_t i = 0; i < num_states * num_states; ++i) {
        log_transition[i] = std::log1p(transition[i]);
    }

    std::size_t iter = 0;
    float prev_loglik = INFINITY;
    while (true) {
        float transition_diff = INFINITY;
        float emission_diff = INFINITY;

        auto alpha = forward(observations, num_observations, log_transition, emission, log_initial, num_states, num_emissions);
        auto beta = backward(observations, num_observations, log_transition, emission, num_states, num_emissions);

        // check if the max iteration is reached
        if (iter >= max_iters) {
            std::cout << "Max iteration reached." << std::endl;

            delete[] alpha;
            delete[] beta;
            break;
        }

        float loglik = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            loglik = log_add(loglik, alpha[i * num_observations + num_observations - 1]);
        }
        std::cerr << loglik << std::endl;

        std::cerr << "emissions: " << std::endl;
        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_emissions; ++j) {
                std::cerr << (emission[i * num_emissions + j]) << ' ';
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
        
        // compute gamma and xi
        auto gamma = compute_gamma(alpha, beta, num_observations, num_states);
        auto xi = compute_xi(alpha, beta, observations, num_observations, log_transition, emission, num_states, num_emissions);

        // delete[] alpha;
        // delete[] beta;

        // std::cout << "CMP G X."<<std::endl;

        // update initial
        for (std::size_t i = 0; i < num_states; ++i) {
            log_initial[i] = gamma[i * num_observations];
        }
        
        float* sum_gamma = new float[num_states];

        // update transition matrix
        for (std::size_t i = 0; i < num_states; ++i) {
            sum_gamma[i] = -INFINITY;
            for (std::size_t t = 0; t < num_observations - 1; ++t) {
                sum_gamma[i] = log_add(sum_gamma[i], gamma[i * num_observations + t]);
            }

            for (std::size_t j = 0; j < num_states; ++j) {
                float sum = -INFINITY;
                for (std::size_t t = 0; t < num_observations - 1; ++t) {
                    sum = log_add(sum, xi[i * num_states * num_observations + j * num_observations + t]);
                }
                log_transition[i * num_states + j] = sum - sum_gamma[i];
            }
        }

        // update emission matrix
        for (std::size_t i = 0; i < num_states; ++i)
            sum_gamma[i] = std::exp(sum_gamma[i]);
            
        for (std::size_t i = 0; i < num_states; ++i) {
            // sum_gamma[i] = std::exp(sum_gamma[i]);
            if (sum_gamma[i] == 0) continue;

            emission[i * num_emissions] = 0;
            emission[i * num_emissions + 1] = 0;

            // O - mu can be negative so can't calculate sum as log without safe sum
            for (std::size_t t = 0; t < num_observations; ++t) {
                gamma[i * num_observations + t] = std::exp(gamma[i * num_observations + t]);
                emission[i * num_emissions] += gamma[i * num_observations + t] * observations[t]; // E[Observation]
            }
            emission[i * num_emissions] /= sum_gamma[i]; // E[Observation] = mu

            for (std::size_t t = 0; t < num_observations; ++t)
                emission[i * num_emissions + 1] += gamma[i * num_observations + t] * (observations[t] - emission[i * num_emissions]) * (observations[t] - emission[i * num_emissions]); // E[ (Obs-mu)^2 ]
            
            emission[i * num_emissions + 1] = std::sqrt(emission[i * num_emissions + 1] / sum_gamma[i]); // sigma = sqrt( E[ (Obs-mu)^2 ] )
        }

        delete[] sum_gamma;
        delete[] gamma;
        delete[] xi;

        if (std::fabs(loglik - prev_loglik) < tolerance) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
        prev_loglik = loglik;

        ++iter;
    }

    // transform initial from log to normal
    for (std::size_t i = 0; i < num_states; ++i) {
        initial[i] = std::exp(log_initial[i]);
    }

    // transform transition matrix and emission matrix from log to normal
    for (std::size_t i = 0; i < num_states * num_states; ++i) {
        transition[i] = std::exp(log_transition[i]);
    }

    delete[] log_initial;
    delete[] log_transition;
}