#include "baum_welch.hpp"
#include "di.hpp"
#include "viterbi_simd.hpp"
#include "viterbi.hpp"
#include "gtest/gtest.h"
#define PI acos(-1)

TEST(tests, di) {
    float* data = new float[25] {
        0.1, 0.3, 0.0, 0.0, 0.0,
        0.2, 0.4, 0.4, 0.0, 0.0,
        0.0, 0.5, 0.6, 0.3, 0.0,
        0.0, 0.0, 0.1, 0.4, 0.1,
        0.0, 0.0, 0.0, 0.1, 0.2
    };

    float* expected_di = new float[5] {
        0.3, 0.0666667, -0.05, 0.0, -0.1
    };

    float* di = calculate_di_AVX2(data, 5, SIGNIFICANT_BINS / 2);

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
    if(state == vectorized::UpstreamBias)
    {
        mu = 40;
    }
    else if(state == vectorized::DownstreamBias)
    {
        mu = -40;
    }
    else if(state == vectorized::NoBias)
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

TEST(tests, viterbi_simdpp) {
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
    auto viterbi_result = vectorized::viterbi(observation, sizeof_observation, start_p, transition_p, emission_probability);

    int expected_result[] = { 0,     2,         2,         1, 2, 
                                2,           1, 1, 1, 2, 
                                0,     0,   0,   0,   0, 
                                0,     2,         2,         2,         2, 
                                2,           2,         2,         2,         1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 2, 
                                0,     0,   2,         2,         2, 
                                2,           2,         2,         2,         2, 
                                2,           1,  };

    for(int i=0; i<sizeof_observation; i++)
        EXPECT_EQ(viterbi_result[i], expected_result[i]);
}

TEST(tests, viterbi_raw) {
    // set input
    int observation[] = {  50,   8,  -5, -22,   1,   3,  -20, -50, -12,   6, 
                             11,  50,  50,  50,  20,  18,    7,   1,  -1,  -1, 
                             -2,  -2,  -1,  -4, -12, -39,   -7, -11, -50, -50, 
                            -50, -16, -14, -14, -50, -50,  -50, -50, -50,  10, 
                             40,  50,  10,   2,  18,   1, -1,   4,   1, 0, 
                             -1, -26};
    auto sizeof_observation = 52;

    float start_p[3] = {0.33, 0.33, 0.33};

    float transition_p[3*3] = {
        0.7,    0.1,    0.2,
        0.1,    0.7,    0.2,
        0.36,   0.36,   0.28
    };
    
    float emission_p[3*3] = {
        0.61, 0.14, 0.24,
        0.57, 0.16, 0.26,
        0.58, 0.15, 0.25,
    };

    // call viterbi algorithm
    auto viterbi_result = scalar::viterbi(observation, sizeof_observation, start_p, transition_p, emission_p);

    int expected_result[] = {   1,   1, 1, 1, 1,
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 1, 1, 1, 
                                1,   1, 
                                };
    for(int i=0; i<sizeof_observation; i++)
        EXPECT_EQ(viterbi_result[i], expected_result[i]);
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

    baum_welch(observations, 16, initial, transition, emission, 3, 3, 1e-7, 5);

    float* expected_initial = new float[3] { 0.625550, 0.164645, 0.209805 };
    float* expected_transition = new float[3 * 3] {
        0.417358, 0.316256, 0.264661,
        0.276578, 0.400094, 0.324230,
        0.295554, 0.340282, 0.365140
    };
    float* expected_emission = new float[3 * 3] {
        0.449340, 0.370611, 0.245848,
        0.368948, 0.419476, 0.278882,
        0.377467, 0.412418, 0.277082
    };

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(initial[i], expected_initial[i], 1e-4);
    }
    for (int i = 0; i < 3 * 3; i++) {
        EXPECT_NEAR(transition[i], expected_transition[i], 1e-4);
    }
    for (int i = 0; i < 3 * 3; i++) {
        EXPECT_NEAR(emission[i], expected_emission[i], 1e-4);
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