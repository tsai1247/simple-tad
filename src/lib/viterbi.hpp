#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace scalar {

float emission_func(float* emission_p, int di, int state) {
    return emission_p[state * 3 + di];
}

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
int* viterbi(int* observation, std::size_t sizeof_observation,
    float* start_p, float* transition_p, float* emission_p) {

    // 宣告V變數，V變數用來記錄當前的最佳機率與上一次迭代的最佳機率
    float** V = new float*[2];
    for (int i = 0; i < 2; ++i) {
        V[i] = new float[3]();
    }

    // 宣告path變數，path變數用來儲存當前計算到的「以state i為終點」的最佳路徑，
    int** path = new int*[sizeof_observation];
    for (std::size_t i = 0; i < sizeof_observation; ++i) {
        path[i] = new int[3]();
    }

    // Initialize
    // t == 0 的情況，V是每個state的初始機率與他們對應的的噴出機率；而path中存放的是以各個state為終點的，長度僅為1的路徑。
    for (int state = 0; state < 3; ++state) {
        V[0][state] = std::log10(start_p[state]) + std::log10(emission_func(emission_p, observation[0], state));
        path[state][0] = state;
    }

    // 迭代 t 為 1~sizeof_oberservation時的情況，每次都會尋找到當前DI值為止，以三個state為終點的最佳路徑
    for (std::size_t t = 1; t < sizeof_observation; ++t) {
        // 對每個state計算
        for (int curr_state = 0; curr_state < 3; ++curr_state) {
            std::pair<float, int>* paths_to_curr_st = new std::pair<float, int>[3]();

            // 對當前要計算的curr_st，計算從每個prev_st到curr_st的可能性，只取最高的存入 best_path
            for (int prev_state = 0; prev_state < 3; ++prev_state) {
                auto current_prob = V[(t - 1) % 2][prev_state] + std::log10(transition_p[prev_state * 3 + curr_state]) + std::log10(emission_func(emission_p, observation[t], curr_state));
                paths_to_curr_st[prev_state] = std::make_pair(current_prob, prev_state);
            }
            std::sort(paths_to_curr_st, paths_to_curr_st + 3, [](auto& a, auto& b) { return a.first > b.first; });

            auto best_path = paths_to_curr_st[2];
            V[t % 2][curr_state] = best_path.first;
            path[t][curr_state] = best_path.second;

            delete[] paths_to_curr_st;
        }
    }

    // V[(sizeof_observation-1)%2]中存的是三個path的最大發生機率
    // 選擇最大的機率對應的path並回傳
    auto prob = 0.0;
    int end_state;
    for (int state = 0; state < 3; ++state) {
        auto curr_prob = V[(sizeof_observation - 1) % 2][state];
        if (curr_prob > prob || prob >= 0.0) {
            prob = curr_prob;
            end_state = state;
        }
    }

    // int* ret = (int*)malloc(sizeof_observation * sizeof(int));
    int* ret = new int[sizeof_observation]();
    ret[sizeof_observation - 1] = end_state;
    for (int i = sizeof_observation - 2; i >= 0; i--) {
        ret[i] = path[i + 1][ret[i + 1]];
    }

    for (int i = 0; i < 2; ++i) {
        delete[] V[i];
    }
    delete[] V;

    for (std::size_t i = 0; i < sizeof_observation; ++i) {
        delete[] path[i];
    }
    delete[] path;

    return ret;
}
}