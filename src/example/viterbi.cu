#include "../lib/viterbi.cuh"
#define PI acos(-1)
#include<iostream>

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

int main()
{
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

    // print result
    for(int t=0; t<sizeof_observation; t++)
    {
        if(viterbi_result[t] == UpstreamBias)
        {
            cout<<"P"<<", ";
        }
        else if(viterbi_result[t] == DownstreamBias)
        {
            cout<<"N"<<", ";
        }
        else
        {
            cout<<"-"<<", ";
        }
    }
    cout<<endl;
}