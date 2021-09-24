from video_generation.generator import generate_video
from model.solver import Solver
from model.data_loader import get_loader_custom_video_data
from model.configs import get_config
from feature_extraction.generate_dataset import GenerateDataset
import sys

if __name__ == '__main__':
    # args should be passed in
    if len(sys.argv) > 1:
        video_path = sys.argv[1].strip()
        save_path = 'data/custom/custom_video.h5'

        # feature extraction
        gen_data = GenerateDataset(video_path, save_path)
        gen_data.generate_dataset()

        # init test config
        config = get_config(mode='test', video_type='custom_video')
        print(config)

        # init data loader
        train_loader = None
        test_loader = get_loader_custom_video_data(config.mode, save_path, config.action_state_size)

        # evaluation
        solver = Solver(config, train_loader, test_loader)
        solver.build()
        solver.loadfrom_checkpoint('exp1/TVSum/models/split4/epoch-84.pkl')
        solver.evaluate(-1)

        # generate video
        score_path = 'exp1/custom_video/results/split' + str(config.split_index) + '/custom_video_-1.json'
        generate_video(score_path, save_path, video_path)