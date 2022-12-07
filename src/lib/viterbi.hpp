#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace std;
#define PI acos(-1)

enum BiasState {
    UpstreamBias,
    DownstreamBias,
    NoBias,
};

vector<BiasState> viterbi(vector<double> observation, vector<BiasState> states, map<BiasState, double> start_p,
    map<BiasState, map<BiasState, double>> transition_p, double (*emission_p)(double, BiasState)) {
    vector<map<BiasState, double>> V;
    map<BiasState, vector<BiasState>> path;

    // Initialize
    for (auto& st : states) {
        V.push_back(map<BiasState, double>());
        V[0][st] = log(start_p[st]) + log(emission_p(observation[0], st));
        path[st] = vector<BiasState> { st };
    }

    // Run Viterbi when t > 0
    for (std::size_t t = 1; t < observation.size(); t++) {
        V.push_back(map<BiasState, double>());
        map<BiasState, vector<BiasState>> newpath;

        for (auto& cur_st : states) {
            vector<pair<double, BiasState>> paths_to_curr_st;
            for (auto& prev_st : states) {
                auto current_prob = V[t - 1][prev_st] + log(transition_p[prev_st][cur_st] * emission_p(observation[t], cur_st));
                cout << current_prob << ", ";
                paths_to_curr_st.push_back(
                    make_pair(current_prob, prev_st));
            }
            cout << endl;
            sort(paths_to_curr_st.begin(), paths_to_curr_st.end());
            auto best_path = paths_to_curr_st[0];
            V[t][cur_st] = best_path.first;
            auto newpath_curst = path[best_path.second];
            newpath_curst.push_back(cur_st);
            newpath[cur_st] = newpath_curst;
        }
        // No need to keep the old paths
        path = newpath;
    }
    auto prob = 0.0;
    BiasState end_state;
    for (auto& st : states) {
        auto cur_prob = V[V.size() - 1][st];
        if (cur_prob > prob || prob >= 0.0) {
            prob = cur_prob;
            end_state = st;
        }
    }

    vector<BiasState> result = path[end_state];
    return result;
}

/*
input:
    emit_value: 實際噴出值
    state: 當前state
output
    double probability: 當前state噴出emit value的機率
*/
