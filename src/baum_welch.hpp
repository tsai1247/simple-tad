#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

/*  Forward algorithm to
 *  calculate the likelihood of the observed data
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
    auto T = B.size(); // get the length of observed sequence.

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




std::vector<std::vector<double>> 
backward(   const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            const std::vector<int>& O);
void 
baum_welch(  std::vector<std::vector<double>>& A,
            std::vector<std::vector<double>>& B,
            std::vector<double>& pi,
            const std::vector<int>& O,
            double tolerance = 1e-5,
            int maxIterations = 1000);