import numpy as np
import torch
import pickle
from run_scmt import run_video
import cv2
from write_videos import write_video

results = run_video('./input/c010', './reid/resnet101_ibn_a_2.pth',
                    save_video_name=False,
                    detection_thres=0.4,
                    iou_thres=0.7,
                    track_thres=0.6,
                    match_thres=0.4,
                    alpha_fuse=0.5)

with open('./output_pkl/results.pkl', 'wb') as f:
    pickle.dump(results, f)

labels = []
for result in results:
    labels.append([result[1], result[0], result[2], result[3], result[4], result[5]])
labels = np.array(labels, dtype=int)

video = cv2.VideoCapture('./input/c010/vdo.avi')

write_video('c010_out.avi', 10, video, labels)
