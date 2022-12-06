#include<map>
#include<vector>
#include<cmath>
#include<algorithm>
#include<iostream>

using namespace std;
#define PI acos(-1)

enum BiasState
{
    UpstreamBias, 
    DownstreamBias, 
    NoBias, 
};

template <typename T>
vector<T> viterbi(vector<double> observation, vector<T> states, map<T, double> start_p, 
    map<T, map<T, double>> transition_p, double (*emission_p)(double, T))
{
    vector<map<T, double>> V;
    map<T, vector<T>> path;

    // Initialize
    for(auto &st : states)
    {
        V.push_back(map<T, double>());
        V[0][st] = log(start_p[st]) + log(emission_p(observation[0], st));
        path[st] = vector<T>{st};
    }   

    // Run Viterbi when t > 0
    for(auto t=1; t<observation.size(); t++)
    {
        V.push_back(map<T, double>());
        map<T, vector<T>> newpath;

        for(auto& cur_st : states)
        {
            vector<pair<double, T>> paths_to_curr_st;
            for(auto& prev_st : states)
            {
                auto current_prob = V[t-1][prev_st] + log(transition_p[prev_st][cur_st] * emission_p(observation[t], cur_st));
                cout<<current_prob<<", ";
                paths_to_curr_st.push_back(
                    make_pair(current_prob, prev_st)
                );
            }
            cout<<endl;
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
    T end_state;
    for(auto& st : states)
    {
        auto cur_prob = V[V.size()-1][st];
        if(cur_prob > prob || prob >= 0.0)
        {
            prob = cur_prob;
            end_state = st;
        }
    }

    vector<T> result = path[end_state];
    return result;
}

/*
input:
    emit_value: 實際噴出值
    state: 當前state
output
    double probability: 當前state噴出emit value的機率
*/


double emission_probability(double emit_value, BiasState state)
{
    auto sigma = 25, mu = 0;
    if(state == UpstreamBias)
    {
        mu = 50;
    }
    else if(state == DownstreamBias)
    {
        mu = -50;
    }
    else if(state == NoBias)
    {
        mu = 0;
    }
    else 
    {
        throw runtime_error("Error: impossible state");
    }
    double pow_sigma2_2times = 2 * pow(sigma, 2);
    double pow_delta_emitvalue = - pow((emit_value - mu), 2);

    double ret = 1.0 / (sigma * sqrt(2*PI)) * exp( pow_delta_emitvalue  / pow_sigma2_2times );
    return ret;
}
