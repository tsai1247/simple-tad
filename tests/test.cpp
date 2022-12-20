#include "di.hpp"
#include "viterbi.hpp"
#include "baum_welch.hpp"
#include "gtest/gtest.h"
#define PI acos(-1)

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

float emission_probability(float emit_value, int state)
{
    auto sigma = 20, mu = 0;
    if(state == UpstreamBias)
    {
        mu = 40;
    }
    else if(state == DownstreamBias)
    {
        mu = -40;
    }
    else if(state == NoBias)
    {
        mu = 0;
    }
    else 
    {
        throw runtime_error("Error: impossible state");
    }
    float pow_sigma2_2times = 2 * pow(sigma, 2);
    float pow_delta_emitvalue = - pow((emit_value - mu), 2);

    float ret = 1.0 / (sigma * sqrt(2*PI)) * exp( pow_delta_emitvalue  / pow_sigma2_2times );
    return ret;
}

TEST(tests, viterbi_raw) {
    // set input
    float observation[] = {  50,   8,  -5, -22,   1,   3,  -20, -50, -12,   6, 
                             11,  50,  50,  50,  20,  18,    7,   1,  -1,  -1, 
                             -2,  -2,  -1,  -4, -12, -39,   -7, -11, -50, -50, 
                            -50, -16, -14, -14, -50, -50,  -50, -50, -50,  10, 
                             40,  50,  10,   2,  18,   1, -1.5,   4,   1, 0.5, 
                             -1, -26};
    auto sizeof_observation = 52;

    float start_p[3] = {0.33, 0.33, 0.33};

    float transition_p[3*3] = {
        0.7,    0.1,    0.2,
        0.1,    0.7,    0.2,
        0.36,   0.36,   0.28
    };

    // call viterbi algorithm
    auto viterbi_result = viterbi(observation, sizeof_observation, start_p, transition_p, emission_probability);

    vector<BiasState> expected_result = { UpstreamBias,     NoBias,         NoBias,         DownstreamBias, NoBias, 
                                          NoBias,           DownstreamBias, DownstreamBias, DownstreamBias, NoBias, 
                                          UpstreamBias,     UpstreamBias,   UpstreamBias,   UpstreamBias,   UpstreamBias, 
                                          UpstreamBias,     NoBias,         NoBias,         NoBias,         NoBias, 
                                          NoBias,           NoBias,         NoBias,         NoBias,         DownstreamBias, 
                                          DownstreamBias,   DownstreamBias, DownstreamBias, DownstreamBias, DownstreamBias, 
                                          DownstreamBias,   DownstreamBias, DownstreamBias, DownstreamBias, DownstreamBias, 
                                          DownstreamBias,   DownstreamBias, DownstreamBias, DownstreamBias, NoBias, 
                                          UpstreamBias,     UpstreamBias,   NoBias,         NoBias,         NoBias, 
                                          NoBias,           NoBias,         NoBias,         NoBias,         NoBias, 
                                          NoBias,           DownstreamBias,  };
    EXPECT_EQ(viterbi_result, expected_result);
}

TEST(tests, baum_welch_scalar) {
    int* observations = new int[16] { 0, 1, 2, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0 };

    float* initial = new float[3] { 0.4, 0.3, 0.3 };
    float* transition = new float[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5
    };
    float* emission = new float[3 * 3] {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.2, 0.1, 0.7
    };

    baum_welch(observations, 16, initial, transition, emission, 3);

    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << std::setiosflags(std::ios::fixed) << initial[i] << " ";
    }
    std::cout << std::endl;

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::cout << std::setiosflags(std::ios::fixed) << transition[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::cout << std::setiosflags(std::ios::fixed) << emission[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] observations;
    delete[] initial;
    delete[] transition;
    delete[] emission;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}