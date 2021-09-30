import os
from tqdm import tqdm
from feature_extraction.networks import ResNet
import numpy as np
import h5py
import decord

class GenerateDataset:
    def __init__(self, video_path, save_path):
        self.resnet = ResNet()
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.h5_file = h5py.File(save_path, 'w')

        self.set_video_list(video_path)

    def set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = sorted(os.listdir(video_path))
            self.video_list = [x for x in self.video_list if '.mp4' in x]
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx + 1)] = {}
            self.h5_file.create_group('video_{}'.format(idx + 1))

    def extract_feature(self, frame):
        '''
        extract frame feature by passing it through pre-trained ResNet
        '''

        res_pool5 = self.resnet(frame)
        frame_feat = res_pool5.cpu().data.numpy().flatten()

        return frame_feat

    def get_change_points(self, video_path):
        '''
        extract indices of keyframes using decord
        then construct segments
        '''

        vr = decord.VideoReader(video_path)

        fps = int(vr.get_avg_fps())
        n_frames = len(vr)

        key_indices = vr.get_key_indices()

        # To remove similar key frames
        prev = int()
        key_indices_reduced = []
        for v in key_indices:
            if v - prev > fps:
                prev = v
                key_indices_reduced.append(v)

        # add starting and ending frames before constructing segments
        key_indices_reduced = [0] + key_indices_reduced + [n_frames]

        temp_change_points = []
        for idx in range(len(key_indices_reduced) - 1):
            # from current change point to the previous frame of the next change point
            # last frame in list is number of frames
            segment = [key_indices_reduced[idx], key_indices_reduced[idx + 1] - 1]
            
            temp_change_points.append(segment)
        
        change_points = np.array(list(temp_change_points))

        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frames = change_points[change_points_idx][1] - change_points[change_points_idx][0] + 1
            temp_n_frame_per_seg.append(n_frames)

        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return change_points, n_frame_per_seg

    def generate_dataset(self):
        '''
        convert from video file (mp4) to h5 file with the right format
        '''

        for video_idx, video_filename in enumerate(tqdm(self.video_list, desc='Feature Extract', ncols=80, leave=True)):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_name = os.path.basename(video_path)

            vr = decord.VideoReader(video_path, width=224, height=224)  # for passing through resnet

            fps = vr.get_avg_fps()
            n_frames = len(vr)

            frame_list = []
            picks = []
            video_feat = None

            change_points, n_frame_per_seg = self.get_change_points(video_path)

            # each change point is a small sequence of similar frames,
            # we just need to take 1 frame from that segment
            # for representing the main features of the whole segment
            for segment in change_points:
                mid = (segment[0] + segment[1])//2
                frame = vr[mid].asnumpy()

                frame_feat = self.extract_feature(frame)

                picks.append(mid)

                if video_feat is None:
                    video_feat = frame_feat
                else:
                    video_feat = np.vstack((video_feat, frame_feat))

            self.h5_file['video_{}'.format(video_idx + 1)]['features'] = list(video_feat)
            self.h5_file['video_{}'.format(video_idx + 1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg
            self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_name

        self.h5_file.close()

if __name__ == '__main__':
    pass