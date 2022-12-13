#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

enum BiasState {
    UpstreamBias,
    DownstreamBias,
    NoBias,
};

/*
input:
    observation: DI array
    sizeof_observation: DI array 的長度
    start_p: 初始機率陣列，通常是array[3]，對應三個狀態的機率
    transition_p: 轉移機率陣列，通常是array[3*3]。transition_p[i*3+j]存放 state i 到 state j 的轉移機率
    emission_p: 噴出機率，會是個function，傳入當前state與當前DI值，回傳的是當前state理論上噴出該值的機率。

output:
    長度與DI array相同的int陣列，內容是每個DI值對應到的bias。
*/
int* viterbi(float* observation, size_t sizeof_observation, 
    float* start_p, float* transition_p, float (*emission_p)(float, int)) {

    // 宣告V變數，V變數用來記錄當前的最佳機率與上一次迭代的最佳機率
    float V[2][3];

    // 宣告path變數，path變數用來儲存當前計算到的「以state i為終點」的最佳路徑，
    int* path[3];
    for(int i=0; i<3; i++)
        path[i] = (int*)malloc(1* sizeof(int));

    // Initialize
    // t == 0 的情況，V是每個state的初始機率與他們對應的的噴出機率；而path中存放的是以各個state為終點的，長度僅為1的路徑。
    for (int i=0; i<3; i++) {
        auto st = static_cast<int>(i);
        V[0][st] = log10(start_p[st]) + log10(emission_p(observation[0], st));
        path[st][0] = st;
    }

    // 迭代 t 為 1~sizeof_oberservation時的情況，每次都會尋找到當前DI值為止，以三個state為終點的最佳路徑
    for (size_t t = 1; t < sizeof_observation; t++) {
        
        // 初始化newpath，每次迭代都會使用newpath取代舊的path
        int* newpath[3];
        for(int i=0; i<3; i++)
            newpath[i] = (int*)malloc(t* sizeof(int));

        // 對每個state計算
        for (int i=0; i<3; i++) {
            auto cur_st = static_cast<int>(i);
            vector<pair<float, int>> paths_to_curr_st;

            // 對當前要計算的cur_st，計算從每個prev_st到cur_st的可能性，只取最高的存入 best_path
            for (int j=0; j<3; j++) {
                auto prev_st = static_cast<int>(j);
                auto current_prob = V[(t-1)%2][prev_st] + log10(transition_p[prev_st*3+cur_st]) + log10(emission_p(observation[t], cur_st));
                paths_to_curr_st.push_back(make_pair(current_prob, prev_st));
            }
            sort(paths_to_curr_st.begin(), paths_to_curr_st.end());
            auto best_path = paths_to_curr_st[paths_to_curr_st.size()-1];
            V[t%2][cur_st] = best_path.first;

            for(int j=0; j<t; j++)
            {
                newpath[cur_st][j] = path[best_path.second][j];
            }
            newpath[cur_st][t] = cur_st;
        }

        // 更新path
        for(int i=0; i<3; i++)
            path[i] = newpath[i];
            
    }

    // V[(sizeof_observation-1)%2]中存的是三個path的最大發生機率
    // 選擇最大的機率對應的path並回傳
    auto prob = 0.0;
    int end_state;
    for (int i=0; i<3; i++) {
        auto st = static_cast<int>(i);
        auto cur_prob = V[(sizeof_observation - 1)%2][st];
        if (cur_prob > prob || prob >= 0.0) {
            prob = cur_prob;
            end_state = st;
        }
    }

    auto result = path[end_state];
    return result;
}
