#include "di.hpp"
#include "viterbi.hpp"
#include "baum_welch.hpp"
#include "gtest/gtest.h"

TEST(tests, di) {
    float* data = new float[25]{
        0.1, 0.3, 0.0, 0.0, 0.0,
        0.2, 0.4, 0.4, 0.0, 0.0,
        0.0, 0.5, 0.6, 0.3, 0.0,
        0.0, 0.0, 0.1, 0.4, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.2
    };

    float* expected_di = new float[5]{
        0.3, 0.0666667, -0.05, 0.0, -0.1
    };

    float* di = calculate_di_AVX2(data, 5, SIGNIFICANT_BINS/2);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(di[i], expected_di[i], 1e-4);
    }

    delete[] data;
    delete[] expected_di;
    delete[] di;
}

double emission_probability(double emit_value, BiasState state) {
    auto sigma = 25, mu = 0;
    if (state == UpstreamBias) {
        mu = 50;
    } else if (state == DownstreamBias) {
        mu = -50;
    } else if (state == NoBias) {
        mu = 0;
    } else {
        throw runtime_error("Error: impossible state");
    }
    double pow_sigma2_2times = 2 * pow(sigma, 2);
    double pow_delta_emitvalue = -pow((emit_value - mu), 2);

    double ret = 1.0 / (sigma * sqrt(2 * PI)) * exp(pow_delta_emitvalue / pow_sigma2_2times);
    return ret;
}

TEST(tests, viterbi_scalar) {
    vector<double> observation = { 50, 12, -3, -20, 1, 5, -18 };

    vector<BiasState> states = { UpstreamBias, DownstreamBias, NoBias };
    map<BiasState, double> start_p = {
        { UpstreamBias, 0.49 },
        { DownstreamBias, 0.02 },
        { NoBias, 0.49 }
    };

    map<BiasState, map<BiasState, double>> transition_p = {
        { UpstreamBias, {
                            { UpstreamBias, 0.7 },
                            { DownstreamBias, 0.1 },
                            { NoBias, 0.1 },
                        } },

        { DownstreamBias, {
                              { UpstreamBias, 0.1 },
                              { DownstreamBias, 0.7 },
                              { NoBias, 0.1 },
                          } },

        { NoBias, {
                      { UpstreamBias, 0.25 },
                      { DownstreamBias, 0.25 },
                      { NoBias, 0.4 },
                  } },
    };

    double (*emission_p)(double, BiasState) = emission_probability;
    auto viterbi_result = viterbi(observation, states, start_p, transition_p, emission_p);

    vector<BiasState> expected_result = { DownstreamBias, UpstreamBias, DownstreamBias, UpstreamBias, DownstreamBias, UpstreamBias, NoBias };
    EXPECT_EQ(viterbi_result, expected_result);
}

TEST(tests, baum_welch_scalar) {
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
    std::vector<double> pi = {0.4, 0.3, 0.3};

    std::transform(pi.begin(), pi.end(), pi.begin(), [](auto& p){return std::log1p(p);});
    for (std::size_t i = 0; i < A.size(); ++i) {
        std::transform(A[i].begin(), A[i].end(), A[i].begin(), [](auto& a){return std::log1p(a);});
        std::transform(B[i].begin(), B[i].end(), B[i].begin(), [](auto& b){return std::log1p(b);});
    }

    // Define the observed sequence
    std::vector<int> O = {0, 1, 2, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0};

    // Estimate the HMM parameters using the Baum-Welch algorithm
    baum_welch(O);

    std::vector<std::vector<double>> A_expected = {{0.530628, 0.182321, 0.095310},
                                                   {0.095310, 0.470003, 0.262364},
                                                   {0.182321, 0.262364, 0.405465}};
    std::vector<std::vector<double>> B_expected = {{0.530628, 0.182321, 0.095310},
                                                   {0.095310, 0.470003, 0.262364},
                                                   {0.182321, 0.095310, 0.470003}};
    std::vector<double> pi_expected = {0.336472, 0.262364, 0.262364};

    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            EXPECT_NEAR(A[i][j], A_expected[i][j], 1e-6);
        }
    }

    for (std::size_t i = 0; i < B.size(); ++i) {
        for (std::size_t j = 0; j < B[i].size(); ++j) {
            EXPECT_NEAR(B[i][j], B_expected[i][j], 1e-6);
        }
    }

    for (std::size_t i = 0; i < pi.size(); ++i) {
        EXPECT_NEAR(pi[i], pi_expected[i], 1e-6);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}