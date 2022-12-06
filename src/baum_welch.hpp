#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

/*  Forward algorithm
 *  to calculate the likelihood of the observed data
 *  given the current estimate of the HMM parameters.
 *  Args:
 *      A:  transition matrix
 *      B:  emission matrix
 *      pi: initial state distribution
 *      O:  observed sequence
 *  Outputs:
 *      alpha:  the likelihood of the observed data
 */
std::vector<std::vector<double>> 
forward(const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B,
        const std::vector<double>& pi,
        const std::vector<int>& O)
{
    auto N = A.size(); // get the number of hidden states.
    auto T = O.size(); // get the length of observed sequence.

    // init alpha (forward probabilities).
    std::vector<std::vector<double>> alpha(T, std::vector<double>(N));

    // calc the initial forward probabilities.
    for (size_t i = 0; i < N; i++)
        alpha[0][i] = pi[i] * B[i][O[0]];
    
    // Calc the forward probabilities for each step.
    // Similar to viterbi but sum up all probabilities.
    for (size_t t = 1; t < T; t++) {            // Now is observation t (t is from 1 because we skip init observation).
        for (size_t j = 0; j < N; j++) {        // Current is state j.
            auto sum = 0.0;                     // Sum of all probabilities to state j in t 
            for (size_t i = 0; i < N; i++)      // From state i.
                sum += alpha[t-1][i] * A[i][j]; // Translate from state i to state j.
            alpha[t][j] = sum * B[j][O[t]];     // B[j][O[t]] means the probability of emit O[t] in state j.
        }
    }
    return alpha;
}

/*  Backward algorithm
 *  to calculate the likelihood of the remaining part of the observed data.
 *  Args:
 *      A:  transition matrix
 *      B:  emission matrix
 *      O:  observed sequence
 *  Outputs:
 *      beta:  the likelihood of the observed data
 */
std::vector<std::vector<double>> 
backward(   const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            const std::vector<int>& O)
{
    auto N = A.size(); // get the number of hidden states.
    auto T = O.size(); // get the length of observed sequence.

    // Init the backward probabilities.
    std::vector<std::vector<double>> beta(T, std::vector<double>(N));

    // Calc the initial backward probabilities
    for (size_t i = 0; i < N; i++)
        beta[T - 1][i] = 1.0;

    // Calc the backward probabilities for each step.
    for (size_t t = T - 2; t >= 0; t--) {   // Now is observation t (t is from T - 2 because we already init last observation).
        for (size_t i = 0; i < N; i++) {    // From state i.
            double sum = 0.0;               // Sum of all probabilities to state j in t
            for (size_t j = 0; j < N; j++)  // Current is state j.
                sum += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]; // beta[t + 1][j] means likelihood from j state rear step.
            beta[t][i] = sum;   // Likelihood from i state to j state in t step.
        }
    }
    return beta;
}

void 
baum_welch(  std::vector<std::vector<double>>& A,
            std::vector<std::vector<double>>& B,
            std::vector<double>& pi,
            const std::vector<int>& O,
            double tolerance = 1e-5,
            int maxIterations = 1000);