import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
def write_video(video_name, frame_rate, origin_video, labels):
    length = int(origin_video.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(origin_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(origin_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    breakpoint()
    j = 0
    for i in tqdm(range(length)):
        flag, orig_img = origin_video.read()
        #orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        if j < len(labels):
            temp = labels[j][1]
        else:
            temp = -1

        working_frame = i + 1
        while j < len(labels) and working_frame == labels[j][1]:
            tid = labels[j][0]
            tlwh = labels[j][2:6]
            #xyxy = (tlwh[1], tlwh[0], tlwh[1] + tlwh[2], tlwh[0] + tlwh[3])
            xyxy = (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])
            #print(tlwh)
            #breakpoint()
            org = (xyxy[0], xyxy[1] - 10)
            cv2.rectangle(orig_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]) ,(0, 0, 255), thickness=2,\
                          lineType=cv2.LINE_8)
            orig_img = cv2.putText(orig_img, f'ID: {tid}', org, font, fontScale, color, thickness, cv2.LINE_AA)

            j += 1
        out_video.write(orig_img)

    out_video.release()
#
# names = ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
# annotation_file = np.loadtxt('ICA/output2.txt', delimiter=' ', dtype=int)
# #annotation_file = np.loadtxt('ground_truth_train.txt/ground_truth_train.txt', delimiter=' ', dtype=int)
#
# annotation_file = pd.DataFrame(annotation_file, columns=names)
# camera_ids = [10, 11, 12, 13, 14, 15]
# idx = annotation_file['CameraId'].isin(camera_ids)
# annotation_file = annotation_file[idx]
#
#
# grouped = annotation_file.groupby('CameraId')
#
# for name, group in grouped:
#     video = cv2.VideoCapture('videos/' + f'vdo{name}' + '.avi')
#     sorted_group = group.sort_values('FrameId')
#     print(f"CameraId: {name}")
#
#     frame_rate = video.get(cv2.CAP_PROP_FPS)
#     w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     write_video(f'output{name}.avi', frame_rate, video, sorted_group.values[:, 1:])
