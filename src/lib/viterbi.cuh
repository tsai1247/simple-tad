#include <cmath>

using namespace std;
#define PI acos(-1)

const int N = 3;

enum BiasState {
    UpstreamBias,
    DownstreamBias,
    NoBias,
};

__global__ void init_path(float* V, float* start_p, int* path, float* emission_p)
{
    int i = threadIdx.x;
    V[i] = log10(start_p[i]) + log10(emission_p[i]);
    path[i] = i;
}

__global__ void get_path_until_t(float* transition_p, float* emission_p, float* V, int* path, int* newpath, int t)
{
    auto i = threadIdx.x;
    float prob_to_curr_st[3];
    int path_to_curr_st[3];
    
    // 對當前要計算的cur_st，計算從每個prev_st到cur_st的可能性，只取最高的存入 best_path
    for (int j=0; j<3; j++) {
        auto current_prob = V[(t-1)%2*3+j];
        current_prob += log10(transition_p[j*3+i]);
        current_prob += log10(emission_p[i]);
        prob_to_curr_st[j] = current_prob;
        path_to_curr_st[j] = j;
    
    }

    float best_prob = prob_to_curr_st[0];
    int best_path = path_to_curr_st[0];
    for(int j=1; j<3; j++)
    {
        if(best_prob < prob_to_curr_st[j])
        {
            best_prob = prob_to_curr_st[j];
            best_path = path_to_curr_st[j];
        }
    }
    V[t%2*3+i] = best_prob;

    for(int j=0; j<t; j++)
    {
        newpath[i*(t+1)+j] = path[best_path*t+j];
    }
    newpath[i*(t+1)+t] = i;
    
}

template <typename T>
T* copy(T* host_arr, size_t size)
{
    T* device_arr;
    cudaMalloc(&device_arr, size*sizeof(T));
    cudaMemcpy(device_arr, host_arr, size * sizeof(T), cudaMemcpyHostToDevice);
    return device_arr;
}

/*
input:
    observation: DI array
    sizeof_observation: DI array 的長度
    start_p: 初始機率陣列，通常是array[3]對應三個狀態的機率
    transition_p: 轉移機率陣列，通常是array[3*3]，存放每個state到各個state的轉移機率
    emission_p: 噴出機率，會是個function，傳入當前state與當前DI值，回傳的是當前state理論上噴出該值的機率。

output:
    長度與DI array相同的int陣列，內容是每個DI值對應到的bias。
*/
int* viterbi(float* observation, size_t sizeof_observation, 
    float* start_p, float* transition_p, float (*emission_func)(float, int)) {
    // 把參數傳遞至gpu
    auto cuda_observation = copy(observation, sizeof_observation);
    auto cuda_start_p = copy(start_p, 3);
    auto cuda_transition_p = copy(transition_p, 9);

    // 宣告V變數，V變數用來記錄當前的最佳機率與上一次迭代的最佳機率
    float* V;
    cudaMalloc(&V, 2*3*sizeof(float));

    // 宣告path變數，path變數用來儲存當前計算到的「以state i為終點」的最佳路徑，
    int* path;   // path[3][t] -> path[3*t]
    cudaMalloc(&path, 3*1*sizeof(int));
    
    // Initialize
    // t == 0 的情況，V是每個state的初始機率與他們對應的的噴出機率；而path中存放的是以各個state為終點的，長度僅為1的路徑。
    
    float emission_p[3];
    for(int i=0; i<3; i++)
    {
        emission_p[i] = emission_func(observation[0], i);
    }
    float* cuda_emission_p = copy(emission_p, 3);

    init_path<<<1, N>>>(V, cuda_start_p, path, cuda_emission_p);

    // 迭代 t 為 1~sizeof_oberservation時的情況，每次都會尋找到當前DI值為止，以三個state為終點的最佳路徑
    for (size_t t = 1; t < sizeof_observation; t++) {
        // 初始化newpath，每次迭代都會使用newpath取代舊的path
        int* newpath;   // path[3][t] -> path[3*t]
        cudaMalloc(&newpath, 3*(t+1)*sizeof(int));

        for(int i=0; i<3; i++)
        {
            emission_p[i] = emission_func(observation[t], i);
        }
        cuda_emission_p = copy(emission_p, 3);
        // 對每個state計算
        get_path_until_t<<<1, N>>>(cuda_transition_p, cuda_emission_p, V, path, newpath, t);
        
        // 更新path
        for(int i=0; i<3; i++)
        {
            cudaFree(path);
            cudaMalloc(&path, 3*(t+1)*sizeof(int));
            cudaMemcpy(path, newpath, 3*(t+1)*sizeof(int), cudaMemcpyDeviceToDevice);
        }   
    }

    // V[(sizeof_observation-1)%2]中存的是三個path的最大發生機率
    // 選擇最大的機率對應的path並回傳
    auto prob = 0.0;
    int end_state;
    for (int i=0; i<3; i++) {
        auto st = static_cast<int>(i);
        float cur_prob;
        cudaMemcpy(&cur_prob, &(V[(sizeof_observation - 1)%2*3+st]), 1*sizeof(float), cudaMemcpyDeviceToHost);
        if (cur_prob > prob || prob >= 0.0) {
            prob = cur_prob;
            end_state = st;
        }
    }

    int* result = (int*)malloc(sizeof_observation * sizeof(int));
    cudaMemcpy(result, &(path[end_state*sizeof_observation]), sizeof_observation*sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}
