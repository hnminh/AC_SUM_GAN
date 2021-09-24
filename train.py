from model.configs import get_config
from model.data_loader import get_loader, get_loader_custom_video_data
from model.solver import Solver

from feature_extraction.generate_dataset import GenerateDataset
import sys

if __name__ == '__main__':

    # if custom video dataset is chosen, then parse
    if len(sys.argv) > 1:
        # parse
        video_path = sys.argv[1].strip()
        save_path = 'data/custom/custom_video.h5'

        # feature extraction
        gen_data = GenerateDataset(video_path, save_path)
        gen_data.generate_dataset()

        # init training config
        config = get_config(mode='train', video_type='custom_video')
        test_config = get_config(mode='test', video_type='custom_video')
        print(config)
        print(test_config)

        # init data loader
        train_loader = get_loader_custom_video_data(config.mode, save_path, config.action_state_size)
        test_loader = get_loader_custom_video_data(test_config.mode, save_path, test_config.action_state_size)

        # train
        solver = Solver(config, train_loader, test_loader)
        solver.build()
        solver.train()

    else: 

        # init configs
        config = get_config(mode='train')
        test_config = get_config(mode='test')
        print(config)
        print(test_config)

        # init data loader
        train_loader = get_loader('tvsum', config.mode, config.split_index, config.action_state_size)
        test_loader = get_loader('tvsum', test_config.mode, test_config.split_index, test_config.action_state_size)
        
        # train
        solver = Solver(config, train_loader, test_loader)
        solver.build()
        solver.evaluate(-1)  # evaluates the summaries generated using the initial random weights of the network
        solver.train()