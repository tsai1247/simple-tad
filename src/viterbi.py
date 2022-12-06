import numpy as np
from math import log10 as log
states = ('Upstream Bias', 'DownStream Bias', 'No Bias')
 
observations = (50, 12, -3, -20, 1, 5, -18)
 
start_probability = {'Upstream Bias': 0.49, 'DownStream Bias': 0.49, 'No Bias': 0.02}
 
transition_probability = {
   'Upstream Bias' : {'Upstream Bias': 0.7, 'DownStream Bias': 0.2, 'No Bias': 0.1},
   'DownStream Bias' : {'Upstream Bias': 0.2, 'DownStream Bias': 0.7, 'No Bias': 0.1},
   'No Bias' : {'Upstream Bias': 0.42, 'DownStream Bias': 0.42, 'No Bias': 0.16},
   }
 
def emission_probability(bins, key):
    sigma = 25
    if key == 'Upstream Bias':
        mu = 50
    if key == 'DownStream Bias':
        mu = -50
    if key == 'No Bias':
        mu = 0
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )

# emission_probability = {
#    'Upstream Bias' : {'+2': 0.1585, '+1': 0.68, '0': 0.135, '-1': 0.0235, '-2': 0.0},
#    'DownStream Bias' : {'-2': 0.1585, '-1': 0.68, '0': 0.135, '+1': 0.0235, '+2': 0.0},
#    'No Bias' : {'+2': 0.025, '+1': 0.135, '0': 0.68, '-1': 0.135, '-2': 0.025},
#    }

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize
    for st in states:
        V[0][st] = log(start_p[st]) + log(emit_p(obs[0], st))
        path[st] = [st]

    # Run Viterbi when t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}

        for curr_st in states:
            paths_to_curr_st = []
            for prev_st in states:
                paths_to_curr_st.append((V[t-1][prev_st] + log(trans_p[prev_st][curr_st] * emit_p(obs[t], curr_st)), prev_st))
            curr_prob, prev_state = max(paths_to_curr_st)
            V[t][curr_st] = curr_prob
            newpath[curr_st] = path[prev_state] + [curr_st]

        # No need to keep the old paths
        path = newpath
    
    print("Observation", observations)
    for line in dptable(V):
        print(line)
    prob, end_state = max([(V[-1][st], st) for st in states])
    return prob, path[end_state]

def dptable(V):
    # Print a table of steps from dictionary
    yield ' ' * 4 + '    '.join(states)
    for t in range(len(V)):
        yield '{}   '.format(t) + '    '.join(['{:.4f}'.format(V[t][state]) for state in V[0]])

def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)

if __name__ == '__main__':
    print(example())