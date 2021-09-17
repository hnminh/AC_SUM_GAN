import sys
sys.path.insert(0, '')

import os
import json
import numpy as np
import h5py
import decord
import cv2
import moviepy.editor as mpe
from pydub import AudioSegment
from tqdm import tqdm

from evaluation.generate_summary import generate_summary

SCORE_PATH = 'exp1/custom_video/results/split0/custom_video_-1.json'
DATASET_PATH = 'data/custom/custom_video.h5'

all_scores = []
with open(SCORE_PATH) as f:
    data = json.loads(f.read())
    keys = list(data.keys())

    for video_name in keys:
        scores = np.asarray(data[video_name])
        all_scores.append(scores)

all_shot_bound, all_nframes, all_positions = [], [], []
with h5py.File(DATASET_PATH, 'r') as hdf:        
    for video_key in keys:
        video_index = video_key[6:]
        
        sb = np.array( hdf.get('video_' + video_index + '/change_points') )
        n_frames = np.array( hdf.get('video_' + video_index + '/n_frames') )
        positions = np.array( hdf.get('video_' + video_index + '/picks') )

        all_shot_bound.append(sb)
        all_nframes.append(n_frames)
        all_positions.append(positions)

all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

with h5py.File(DATASET_PATH, 'r') as hdf:
    for video_key, summary in zip(keys, all_summaries):
    # for video_key, summary in tqdm(zip(keys, all_summaries), total=len(keys)):

        # extract metadata
        video_name = hdf[video_key + '/video_name'][()]
        audio_name = video_name[:-4] + '.mp3'

        # tqdm.write(video_name)

        video_reader = decord.VideoReader('custom_video/original/' + video_name)
        audio_reader = AudioSegment.from_file('custom_video/original/' + video_name, 'mp4')

        fps = video_reader.get_avg_fps()
        (frame_height, frame_width, _) = video_reader[0].asnumpy().shape

        # add 5 seconds of video beginning and end into summary
        summary[:int(fps*5)] = 1
        summary[-int(fps*5):] = 1

        # extract frame numbers
        frame_numbers = list(np.argwhere(summary == 1).reshape(1, -1).squeeze(0))

        vid_writer = cv2.VideoWriter(
            'custom_video/summarized/' + video_name,
            cv2.VideoWriter_fourcc(*'MP4V'),
            fps,
            (frame_width, frame_height)
        )
        summarized_audio = None

        # generate summary video and audio
        for idx in frame_numbers:
            # write video to file
            frame = video_reader[idx]
            vid_writer.write(cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))

            # get audio idx in seconds
            audio_start_idx, audio_end_idx = video_reader.get_frame_timestamp(idx)
            # seconds to miliseconds
            audio_start_idx = round(audio_start_idx*1000)
            audio_end_idx = round(audio_end_idx*1000)

            # concat audio segment
            if summarized_audio is None:
                summarized_audio = audio_reader[audio_start_idx:audio_end_idx]
            else:
                summarized_audio += audio_reader[audio_start_idx:audio_end_idx]

        # write audio to file
        summarized_audio.export('custom_video/summarized/' + audio_name, format='mp3')

        vid_writer.release()

        # combine video and audio
        input_video = mpe.VideoFileClip('custom_video/summarized/' + video_name)
        input_audio = mpe.AudioFileClip('custom_video/summarized/' + audio_name)

        output_video = input_video.set_audio(input_audio)
        output_video.write_videofile(
            'custom_video/summarized/fin_' + video_name,
            codec='libx264',
            audio_codec='aac',
        )

        os.remove('custom_video/summarized/' + video_name)
        os.remove('custom_video/summarized/' + audio_name)