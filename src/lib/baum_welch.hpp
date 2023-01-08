#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>

double inline log_add(double x, double y) {
    if (y <= x) {
        return x + std::log1p(std::exp(y - x));
    } else {
        return y + std::log1p(std::exp(x - y));
    }
}

double* forward(
    const int* observations,
    const std::size_t num_observations,
    const double* transition,
    const double* emission,
    const double* initial,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    // init the alpha matrix
    double* alpha = new double[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        alpha[i * num_observations] = initial[i] + emission[i * num_emissions + observations[0]];
    }

    // compute the alpha matrix
    for (std::size_t t = 1; t < num_observations; ++t) {
        // now is observation t, and t is from 1 because we skip init observation
        for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
            double sum = -INFINITY;
            for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
                sum = log_add(sum, alpha[prev_state * num_observations + t - 1] + transition[prev_state * num_states + curr_state]);
            }
            alpha[curr_state * num_observations + t] = sum + emission[curr_state * num_emissions + observations[t]];
        }
    }

    return alpha;
}

double* backward(
    const int* observations,
    const std::size_t num_observations,
    const double* transition,
    const double* emission,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    // init the beta matrix
    double* beta = new double[num_states * num_observations]();
    for (std::size_t i = 0; i < num_states; ++i) {
        beta[i * num_observations + num_observations - 1] = 0; // log(1)
    }

    // compute the beta matrix
    for (std::size_t t = num_observations - 2;; --t) {
        // now is observation t, and t is from (num_observations - 2) because we already init last observation
        for (std::size_t prev_state = 0; prev_state < num_states; ++prev_state) {
            double sum = -INFINITY;
            for (std::size_t curr_state = 0; curr_state < num_states; ++curr_state) {
                sum = log_add(sum, beta[curr_state * num_observations + t + 1] + transition[prev_state * num_states + curr_state] + emission[curr_state * num_emissions + observations[t + 1]]);
            }
            beta[prev_state * num_observations + t] = sum;
        }

        if (t == 0) {
            break;
        }
    }

    return beta;
}

double* compute_gamma(
    const double* alpha,
    const double* beta,
    const std::size_t num_observations,
    const std::size_t num_states) {
    double* gamma = new double[num_states * num_observations]();

    for (std::size_t t = 0; t < num_observations; ++t) {
        double sum = -INFINITY;
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

double* compute_xi(
    const double* alpha,
    const double* beta,
    const int* observations,
    const std::size_t num_observations,
    const double* transition,
    const double* emission,
    const std::size_t num_states,
    const std::size_t num_emissions) {
    double* xi = new double[num_states * num_states * num_observations]();

    for (std::size_t t = 0; t < num_observations - 1; ++t) {
        double sum = -INFINITY;
        for (std::size_t i = 0; i < num_states; ++i) {
            for (std::size_t j = 0; j < num_states; ++j) {
                xi[i * num_states * num_observations + j * num_observations + t] = alpha[i * num_observations + t] + transition[i * num_states + j] + emission[j * num_emissions + observations[t + 1]] + beta[j * num_observations + t + 1];
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
    const int* observations,
    const std::size_t num_observations,
    double* initial,
    double* transition,
    double* emission,
    const std::size_t num_states,
    const std::size_t num_emissions,
    const double tolerance = 1e-7,
    const std::size_t max_iters = 1000) {
    // transform initial to log space
    double* log_initial = new double[num_states];
    for (std::size_t i = 0; i < num_states; ++i) {
        log_initial[i] = std::log1p(initial[i]);
    }

    // transform transition matrix and emission matrix to log space
    double* log_transition = new double[num_states * num_states];
    for (std::size_t i = 0; i < num_states * num_states; ++i) {
        log_transition[i] = std::log1p(transition[i]);
    }
    double* log_emission = new double[num_states * num_emissions];
    for (std::size_t i = 0; i < num_states * num_emissions; ++i) {
        log_emission[i] = std::log1p(emission[i]);
    }

    std::size_t iter = 0;
    while (true) {
        double transition_diff = 0;
        double emission_diff = 0;
        
        auto alpha = forward(observations, num_observations, log_transition, log_emission, log_initial, num_states, num_emissions);
        auto beta = backward(observations, num_observations, log_transition, log_emission, num_states, num_emissions);

        // check if the max iteration is reached
        if (iter >= max_iters) {
            std::cout << "Max iteration reached." << std::endl;

            delete[] alpha;
            delete[] beta;
            break;
        }

        // compute gamma and xi
        auto gamma = compute_gamma(alpha, beta, num_observations, num_states);
        auto xi = compute_xi(alpha, beta, observations, num_observations, log_transition, log_emission, num_states, num_emissions);

        delete[] alpha;
        delete[] beta;

        // update initial
        for (std::size_t i = 0; i < num_states; ++i) {
            log_initial[i] = gamma[i * num_observations];
        }

        // update transition matrix
        for (std::size_t i = 0; i < num_states; ++i) {
            double denominator = -INFINITY;
            for (std::size_t t = 0; t < num_observations - 1; ++t) {
                denominator = log_add(denominator, gamma[i * num_observations + t]);
            }

            for (std::size_t j = 0; j < num_states; ++j) {
                double numerator = -INFINITY;
                for (std::size_t t = 0; t < num_observations - 1; ++t) {
                    numerator = log_add(numerator, xi[i * num_states * num_observations + j * num_observations + t]);
                }

                transition_diff += std::abs(std::exp(log_transition[i * num_states + j]) - std::exp(numerator - denominator));

                log_transition[i * num_states + j] = numerator - denominator;
            }
        }

        // update emission matrix
        for (std::size_t i = 0; i < num_states; ++i) {
            double denominator = -INFINITY;
            for (std::size_t t = 0; t < num_observations; ++t) {
                denominator = log_add(denominator, gamma[i * num_observations + t]);
            }

            for (std::size_t k = 0; k < num_emissions; ++k) {
                double numerator = -INFINITY;
                for (std::size_t t = 0; t < num_observations; ++t) {
                    if (observations[t] == int(k)) {
                        numerator = log_add(numerator, gamma[i * num_observations + t]);
                    }
                }

                emission_diff += std::abs(std::exp(log_emission[i * num_emissions + k]) - std::exp(numerator - denominator));

                log_emission[i * num_emissions + k] = numerator - denominator;
            }
        }

        delete[] gamma;
        delete[] xi;

        if (transition_diff < tolerance && emission_diff < tolerance) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }

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
    for (std::size_t i = 0; i < num_states * num_emissions; ++i) {
        emission[i] = std::exp(log_emission[i]);
    }

    delete[] log_initial;
    delete[] log_transition;
    delete[] log_emission;
}