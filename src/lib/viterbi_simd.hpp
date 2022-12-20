#include <cmath>
#include<simdpp/simd.h>
#include<iostream>

using namespace std;
using namespace simdpp;
#define PI acos(-1)

namespace vectorized
{
    const int N = 4;
    enum BiasState {
        UpstreamBias,
        DownstreamBias,
        NoBias,
    };

    int indexof(float value, float32<N> arr)
    {
        int tmp_arr[N] = {0, 1, 2, 0};
        int32<N> tmp = load(tmp_arr);
        
        int32<N> int_arr;
        int_arr = cmp_neq(arr, value);
        int_arr = add(int_arr, 1);
        int_arr = mul_lo(int_arr, tmp);
        int ret = reduce_add(int_arr);
        return ret;
    }

    void init_path(float32<N>* V, float32<N> start_p, uint8<N>* path, float32<N> emission_p)
    {
        V[0] = add(start_p, emission_p);
        float path_arr[3] = {0, 1, 2};
        path[0] = load(path_arr);
    }

    void get_path_until_t(float32<N> transition_p[N], float emission_p[N], 
        float32<N>* V, uint8<N>* path, int t)
    {
        float cur_prob[N];
        int prev_state[N];

        // 對當前要計算的 state i ，使用simd計算三個state來到此state的機率，並取其最大值
        for (int i=0; i<N; i++) {
            // 取得當前state最佳的機率
            float32<N> current_prob = V[t-1];
            current_prob = add(current_prob, transition_p[i]);  // strange
            current_prob = add(current_prob, emission_p[i]);
            cur_prob[i] = reduce_max(current_prob);
            
            // 取得最佳機率發生時的前一個state
            prev_state[i] = indexof(cur_prob[i], current_prob);
        }

        V[t] = load(cur_prob);
        path[t] = load(prev_state);
    }

    void printfloat(float32<N> arr)
    {
        float ori_arr[N];
        store(ori_arr, arr);
        cout<<"[";
        for(int i=0; i<N; i++)
        {
            cout<<ori_arr[i]<<", ";
        }
        cout<<"]"<<endl;
    }
    void printint(int32<N> arr)
    {
        int ori_arr[N];
        store(ori_arr, arr);
        cout<<"[";
        for(int i=0; i<N; i++)
        {
            cout<<ori_arr[i]<<", ";
        }
        cout<<"]"<<endl;
    }


    /*
    input:
        observation: DI array
        sizeof_observation: DI array 的長度
        start_p: 初始機率陣列，通常是array[3]對應三個狀態的機率
        transition_p: 轉移機率陣列，通常是array[3*3]，較特別的是，array[i*3+j]存放的是state j到state i的機率。
        emission_p: 噴出機率，會是個function，傳入當前state與當前DI值，回傳的是當前state理論上噴出該值的機率。

    output:
        長度與DI array相同的int陣列，內容是每個DI值對應到的bias。
    */
    int* viterbi(float* observation, size_t sizeof_observation, 
        float* start_p, float* transition_p, float (*emission_func)(float, int)) {
        // 前處理，將 start_p 與 transition_p 取 log
        float* new_start_p = (float*)malloc(N * sizeof(float));
        for(int i=0; i<N-1; i++)
        {
            new_start_p[i] = log10(start_p[i]);
        }
        new_start_p[N-1] = log10(1e-9);

        float* new_transition_p = (float*)malloc(N*N * sizeof(float));
        for(int i=N*N-1; i >= N*(N-1); i--)
        {
            new_transition_p[i] = log10(1e-9);
        }

        for(int i=N*(N-1)-1; i>=0; i--)
        {
            if((i+1)%N == 0)
                new_transition_p[i] = log10(1e-9);
            else
                new_transition_p[i] = log10(transition_p[i - i/N]);
        }

        // 將機率轉換成simdpp vector type
        float32<N> simd_start_p = load(new_start_p);
        float32<N>* simd_transition_p = (float32<N>*)calloc(N, sizeof(float32<N>));
        for(int i=0; i<N; i++)
        {
            simd_transition_p[i] = load(new_transition_p + i*N);
        }

        // 宣告V變數，V變數用來記錄當前的最佳機率與上一次迭代的最佳機率
        float32<N>* V = (float32<N>*)calloc(sizeof_observation, sizeof(float32<N>));

        // 宣告path變數，path變數用來儲存當前計算到的「以state i為終點」的最佳路徑，
        uint8<N>* path = (uint8<N>*)calloc(sizeof_observation, sizeof(uint8<N>));
        
        // Initialize
        // t == 0 的情況，V是每個state的初始機率與他們對應的的噴出機率；而path中存放的是以各個state為終點的，長度僅為1的路徑。
        float emission_p[N];
        float32<N> simd_emission_p;
        for(int i=0; i<N-1; i++)
        {
            emission_p[i] = log10(emission_func(observation[0], i));
        }
        emission_p[N-1] = log10(1e-9);
        simd_emission_p = load(emission_p);
        init_path(V, simd_start_p, path, simd_emission_p);

        // 迭代 t 為 1~sizeof_oberservation時的情況，每次都會尋找到當前DI值為止，以三個state為終點的最佳路徑
        for (size_t t = 1; t < sizeof_observation; t++) {
            // 對每個state計算
            for(int i=0; i<N-1; i++)
            {
                emission_p[i] = log10(emission_func(observation[t], i));
            }
            emission_p[N-1] = log10(1e-9);
            simd_emission_p = load(emission_p);
            get_path_until_t(simd_transition_p, emission_p, V, path, t);
        }

        // V[(sizeof_observation-1)%2]中存的是三個path的最大發生機率
        // 選擇最大的機率對應的path並回傳
        float max_end_prob = reduce_max(V[sizeof_observation-1]);
        int end_state = indexof(max_end_prob, V[sizeof_observation-1]);

        int* ret = (int*)malloc(sizeof_observation* sizeof(int));
        int cur_state_arr[3];
        ret[sizeof_observation-1] = end_state;
        int cur_state = end_state;
        for(int i=sizeof_observation-2; i>=0; i--)
        {
            store(cur_state_arr, path[i+1]);
            ret[i] = cur_state_arr[cur_state];
            cur_state = ret[i];
        }

        return ret;
    }
}