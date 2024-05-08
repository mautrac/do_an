import numpy as np
import torch
import pickle

from reid.make_model import make_model
from run_scmt import run_video
import cv2
from write_videos import write_video

cam_folder = 'c011'

reid_model = make_model(1000, pretrained_path='./reid/resnet101_ibn_a_2.pth')
reid_model.to('cuda')
reid_model.eval()

results = run_video(f'./input/{cam_folder}', './reid/resnet101_ibn_a_2.pth',
                    save_video_name=False,
                    detection_thres=0.3,
                    iou_thres=0.7,
                    track_thres=0.6,
                    match_thres=0.5,
                    alpha_fuse=0.5,
                    frame_rate=40)

with open('./output_pkl/results.pkl', 'wb') as f:
    pickle.dump(results, f)

labels = []
for result in results:
    labels.append([result[1], result[0], result[2], result[3], result[4], result[5]])
labels = np.array(labels, dtype=int)

video = cv2.VideoCapture(f'./input/{cam_folder}/vdo.avi')

write_video(f'{cam_folder}_out.avi', 10, video, labels)
