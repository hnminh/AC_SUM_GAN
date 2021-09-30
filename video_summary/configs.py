import argparse
from pathlib import Path
import pprint
import math
import json

save_dir = Path('output_feature')

class Config(object):
    def __init__(self, **kwargs):
        '''
        Configuration class: set kwargs as class attributes with setattr
        '''

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.termination_point = math.floor(0.15*self.action_state_size)
        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='TVSum'):
        self.log_dir = save_dir.joinpath(video_type, 'logs/split' + str(self.split_index))
        self.score_dir = save_dir.joinpath(video_type, 'scores/split' + str(self.split_index))
        self.save_dir = save_dir.joinpath(video_type, 'models/split' + str(self.split_index))
        
    def __repr__(self):
        '''
        Pretty-print configurations in alphabetical order
        '''

        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def str2bool(v):
    '''
    string to boolean
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(parse=False, **optional_kwargs):
    '''
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    '''

    # Load default parameters from file
    with open('video_summary/default_params.json', 'r') as f:
        default_params = json.load(f)

    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default=default_params['mode'])
    parser.add_argument('--verbose', type=str2bool, default=default_params['verbose'])
    parser.add_argument('--video_type', type=str, default=default_params['video_type'])

    # Model
    parser.add_argument('--input_size', type=int, default=default_params['input_size'])
    parser.add_argument('--hidden_size', type=int, default=default_params['hidden_size'])
    parser.add_argument('--num_layers', type=int, default=default_params['num_layers'])
    parser.add_argument('--regularization_factor', type=float, default=default_params['regularization_factor'])
    parser.add_argument('--entropy_coef', type=float, default=default_params['entropy_coef'])

    # Train
    parser.add_argument('--n_epochs', type=int, default=default_params['n_epochs'])
    parser.add_argument('--batch_size', type=int, default=default_params['batch_size'])
    parser.add_argument('--clip', type=float, default=default_params['clip'])
    parser.add_argument('--lr', type=float, default=default_params['lr'])
    parser.add_argument('--discriminator_lr', type=float, default=default_params['discriminator_lr'])
    parser.add_argument('--split_index', type=int, default=default_params['split_index'])
    parser.add_argument('--action_state_size', type=int, default=default_params['action_state_size'])
    
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)