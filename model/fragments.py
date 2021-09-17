import torch
import numpy as np
import math

def calculate_fragments(seq_len, num_fragments):
    '''
    The sequence must be divided into "num_fragments" fragments.
    Since seq_len/num_fragments will not be a perfect division,
    we need to take both floor and ceiling values, then try to fit
    them in a way such that the sum of all fragments is equal
    to the size of the sequence.
    '''

    fragment_size = seq_len/num_fragments
    fragment_floor = math.floor(fragment_size)
    fragment_ceil = math.ceil(fragment_size)
    i_part, d_part = divmod(fragment_size, 1)

    frag_jump = np.zeros(num_fragments)

    upper = np.round(d_part*num_fragments).astype(int)
    lower = num_fragments - upper

    for i in range(lower):
        frag_jump[i] = fragment_floor
    for i in range(upper):
        frag_jump[lower + i] = fragment_ceil
    
    # Roll the scores, so that the larger fragments fall at 
    # the center of the sequence. Should not make a difference.
    frag_jump = np.roll(frag_jump, -int(num_fragments*(1 - d_part)/2))

    return frag_jump.astype(int)

def compute_fragments(seq_len, action_state_size):

    frag_jump = calculate_fragments(seq_len, action_state_size)

    # "action_fragments" contains the starting and ending frame of each action fragment
    action_fragments = torch.zeros((action_state_size, 2), dtype=torch.int64)

    # action_fragments[i, 0] is starting point
    # action_fragments[i, 1] is ending point
    for i in range(action_state_size - 1):
        action_fragments[i, 1] = torch.tensor(sum(frag_jump[0:i + 1]) - 1)
        action_fragments[i + 1, 0] = torch.tensor(sum(frag_jump[0:i + 1]))
    action_fragments[action_state_size - 1, 1] = torch.tensor(sum(frag_jump) - 1)

    return action_fragments

if __name__ == '__main__':
    pass