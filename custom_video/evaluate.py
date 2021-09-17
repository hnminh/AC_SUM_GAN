import sys
sys.path.insert(0, '')

from model.data_loader import CustomVideoData
from model.configs import get_config
from model.solver import Solver

if __name__ == '__main__':
    config = get_config(mode='eval', video_type='custom_video')
    custom_video_loader = CustomVideoData('data/custom/custom_video.h5', config.action_state_size)

    solver = Solver(config, None, custom_video_loader)

    solver.build()
    solver.loadfrom_checkpoint('exp1/TVSum/models/split4/epoch-84.pkl')
    solver.evaluate(-1)