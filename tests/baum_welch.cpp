#include <cassert>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <numeric>
#include <vector>
#include <iomanip>
#include <algorithm>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// When add in log likelihood.
double logadd(double x, double y) {
    if (x >= y)
        return x + std::log1p(std::exp(y - x));
    else
        return y + std::log1p(std::exp(x - y));
}

/*  Forward algorithm
 *  Calculate the likelihood of the observed data
 *  Args:
 *      A:  transition matrix
 *      B:  emission matrix
 *      pi: initial state distribution
 *      O:  observed sequence
 *  Outputs:
 *      alpha:  the likelihood of each state
 *  Complexity:
 *      O(T*N*N) ~= O(T)
 */
std::vector<std::vector<double>>
forward(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    const std::vector<double>& pi,
    const std::vector<int>& O) {
    std::size_t N = A.size(); // get the number of hidden states.
    std::size_t T = O.size(); // get the length of observed sequence.

    // init alpha (forward probabilities).
    std::vector<std::vector<double>> alpha(T, std::vector<double>(N));

    // calc the initial forward probabilities.
    for (std::size_t i = 0; i < N; i++)
        alpha[0][i] = pi[i] + B[i][O[0]];

    // Calc the forward probabilities for each step.
    for (std::size_t t = 1; t < T; t++) { // Now is observation t (t is from 1 because we skip init observation).
        for (std::size_t j = 0; j < N; j++) { // Current is state j.
            for (std::size_t i = 0; i < N; i++) { // From state i.
                double p = alpha[t - 1][i] + A[i][j] + B[j][O[t]];
                alpha[t][j] = logadd(alpha[t][j], p); // B[j][O[t]] means the probability of emit O[t] in state j.
            }
        }
    }
    return alpha;
}

/*  Backward algorithm
 *  Calculate the likelihood of the remaining part of the observed data.
 *  Args:
 *      A:  transition matrix
 *      B:  emission matrix
 *      O:  observed sequence
 *  Outputs:
 *      beta:  the likelihood of each state
 * Complexity:
 *      O(T*N*N) ~= O(T)
 */
std::vector<std::vector<double>>
backward(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    const std::vector<int>& O) {
    std::size_t N = A.size(); // get the number of hidden states.
    std::size_t T = O.size(); // get the length of observed sequence.

    // Init the backward probabilities.
    std::vector<std::vector<double>> beta(T, std::vector<double>(N));

    // Calc the initial backward probabilities
    // for (std::size_t i = 0; i < N; i++)
    //     beta[T - 1][i] = 1.0;
    for (std::size_t i = 0; i < N; i++)
        beta[T - 1][i] = 0.0; // 0 = log (1.0)

    // Calc the backward probabilities for each step.
    for (std::size_t t = T - 2; t <= T - 2; t--) { // Now is observation t (t is from T - 2 because we already init last observation).
        for (std::size_t i = 0; i < N; i++) { // From state i.
            for (std::size_t j = 0; j < N; j++) { // Current is state j.
                double p = beta[t + 1][j] + A[i][j] + B[j][O[t + 1]]; // beta[t + 1][j] means likelihood from j state rear step.
                beta[t][i] = logadd(beta[t][i], p); // Likelihood from i state to j state in t step.
            }
        }
    }
    return beta;
}

/*  Baum-Welch algorithm
 *  Uses the forward-backward algorithm to calculate the likelihood of the observed data
 *  given the current estimate of the HMM parameters.
 *  Args:
 *      A:  transition matrix
 *      B:  emission matrix
 *      pi: initial state distribution
 *      O:  observed sequence
 *      tolerance:      default = 1e-5
 *      maxIterations:  default = 1000
 *  Outputs:
 *      void
 */
void baum_welch(std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& B,
    std::vector<double>& pi,
    const std::vector<int>& O,
    double tolerance = 1e-5,
    int maxIterations = 10000) {
    const auto M = B[0].size(); // get the number of emission.
    const auto N = A.size(); // get the number of hidden states.
    const auto T = O.size(); // get the length of observed sequence.

    // Init the likelihood of the observed data.
    double loglik = -INFINITY;

    // Iter until convergence or the maximum number of iterations is reached.
    int iter = 0;
    while (true) {
        // Calc the forward and backward probabilities.
        auto alpha = forward(A, B, pi, O);
        auto beta = backward(A, B, O);

        // Calc the likelihood of the observed data.
        // Use beta[0] will get same result.
        double newloglik = -INFINITY;
        for (std::size_t i = 0; i < N; i++)
            newloglik = logadd(newloglik, alpha[T - 1][i]);

        // std::cerr << "from alpha: " << newloglik << std::endl;

        // Check if converge.
        if (std::abs(std::exp(newloglik) - std::exp(loglik)) < tolerance)
            break;
        loglik = newloglik;
        // std::cout << newloglik << std::endl;

        // Check if reach maxIterations.
        if (iter >= maxIterations)
            break;
        iter++;

        /*  Calculate gamma and xi
         *      gamma[t][i]: In t (idx of observed sequence T) the sum of probability of passing state i.
         *      xi[t][i][j]: In t (idx of observed sequence T) the sum of probability of path from i (idx=t) to j (idx=t+1, obmit here).
         */
        std::vector<std::vector<double>> gamma(T, std::vector<double>(N));
        std::vector<std::vector<std::vector<double>>> xi(T, std::vector<std::vector<double>>(N, std::vector<double>(N)));
        for (std::size_t t = 0; t < T; t++) {

            double denominator_g = -INFINITY, denominator_x = -INFINITY;
            for (std::size_t i = 0; i < N; i++) {
                denominator_g = logadd(denominator_g, (alpha[t][i] + beta[t][i]));
                if (t < T - 1) { // Need to consider t+1
                    for (std::size_t j = 0; j < N; j++)
                        denominator_x = logadd(denominator_x, (alpha[t][i] + A[i][j] + B[j][O[t + 1]] + beta[t + 1][j]));
                }
            }

            std::cerr << t << std::endl;
            for (std::size_t i = 0; i < N; i++) {
                gamma[t][i] = (alpha[t][i] + beta[t][i]) - denominator_g;
                std::cerr << gamma[t][i] << ' ';
                if (t < T - 1) { // Need to consider t+1
                    for (std::size_t j = 0; j < N; j++)
                        xi[t][i][j] = (alpha[t][i] + A[i][j] + B[j][O[t + 1]] + beta[t + 1][j]) - denominator_x;
                }
            }
            std::cerr << std::endl;
        }

        for (std::size_t i = 0; i < N; i++) {
            // Update pi.
            pi[i] = gamma[0][i];

            // Update the transition matrix A.
            for (std::size_t j = 0; j < N; j++) {
                double xi_sum = -INFINITY, gamma_sum = -INFINITY; // Sum of all probability that change from state i to  state j before T.
                for (std::size_t t = 0; t < T - 1; t++) {
                    xi_sum = logadd(xi_sum, xi[t][i][j]);
                    gamma_sum = logadd(gamma_sum, gamma[t][i]);
                }
                A[i][j] = xi_sum - gamma_sum;
            }

            // Update the emission matrix B.
            double denominator_B = -INFINITY; // Sum of all probability that pass state i in T.
            std::vector<double> numerator_B(M, -INFINITY); // Sum of all probability that emmit k (=o[t]) from state i in T.
            for (std::size_t t = 0; t < T; t++) {
                numerator_B[O[t]] = logadd(numerator_B[O[t]], gamma[t][i]);
                denominator_B = logadd(denominator_B, gamma[t][i]);
            }

            for (std::size_t k = 0; k < M; k++)
                B[i][k] = numerator_B[k] - denominator_B;
        }
    }
}

int main() {
    // Define the HMM parameters
    // A:  transition matrix
    std::vector<std::vector<double>> A = {{0.7, 0.2, 0.1},
                                          {0.1, 0.6, 0.3},
                                          {0.2, 0.3, 0.5}};
    
    // B:  emission matrix
    std::vector<std::vector<double>> B = {{0.7, 0.2, 0.1},
                                          {0.1, 0.6, 0.3},
                                          {0.2, 0.1, 0.6}};

    // pi: initial state distribution
    std::vector<double> pi = {0.6, 0.3, 0.3};

    std::transform(pi.begin(), pi.end(), pi.begin(), [](auto& p){return std::log1p(p);});
    for (std::size_t i = 0; i < A.size(); ++i) {
        std::transform(A[i].begin(), A[i].end(), A[i].begin(), [](auto& a){return std::log1p(a);});
        std::transform(B[i].begin(), B[i].end(), B[i].begin(), [](auto& b){return std::log1p(b);});
    }

    // Define the observed sequence
    std::vector<int> O = {0, 1, 2, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0};

    // Estimate the HMM parameters using the Baum-Welch algorithm
    baum_welch(A, B, pi, O);

    std::transform(pi.begin(), pi.end(), pi.begin(), [](auto& p){return std::exp(p);});
    for (std::size_t i = 0; i < A.size(); ++i) {
        std::transform(A[i].begin(), A[i].end(), A[i].begin(), [](auto& a){return std::exp(a);});
        std::transform(B[i].begin(), B[i].end(), B[i].begin(), [](auto& b){return std::exp(b);});
    }

    std::cout << std::endl << "After estimate the HMM parameters using the Baum-Welch algorithm..." << std::endl;

    // Print the estimated parameters
    std::cout << "Estimated transition matrix A:" << std::endl;
    for (const auto& row : A)
    {
        for (const auto& a : row)
            std::cout << std::setiosflags(std::ios::fixed)  << a << " ";
        std::cout << std::endl;
    }

    std::cout << "Estimated emission matrix B:" << std::endl;
    for (const auto& row : B)
    {
        for (const auto& b : row)
            std::cout << std::setiosflags(std::ios::fixed)  << b << " ";
        std::cout << std::endl;
    }

    std::cout << "Estimated initial state distribution pi:" << std::endl;
    for (const auto& p : pi)
        std::cout << std::setiosflags(std::ios::fixed)  << p << " ";
    std::cout << std::endl;

    return 0;
}