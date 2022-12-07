#include "../lib/viterbi.hpp"

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

int main() {
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
    for (auto& result : viterbi_result) {
        cout << "'" << result << "', ";
    }
    cout << endl;
}