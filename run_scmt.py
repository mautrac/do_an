from ultralytics import YOLO
import torch
import cv2
import os
from tqdm.auto import tqdm

from torchvision.transforms import v2

import numpy as np
from ByteTrack.src.fm_tracker.byte_tracker import BYTETracker


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


img_size = [480, 864]
obj_size = [256, 256]

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=obj_size, antialias=True),
    #v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_numpy(tensor_):
    if device == 'cpu':
        return tensor_.detach().numpy()
    return tensor_.detach().cpu().numpy()

class_IDS = [2, 3, 5, 7]


def process_video(detection_model, reid_model, working_path, save_video=False, \
                  verbose=False, detection_thres=0.2, iou_thres=0.7, \
                  track_thres=0.6, match_thres=0.4, frame_rate=10, \
                  limit=-1, alpha_fuse=0.8, **kwargs):
    video_path = os.path.join(working_path, 'vdo.avi')
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_cnt = 0
    str_path = working_path.split('/')
    cid = int(str_path[-1][1:])
    roi = np.ones((h, w), dtype=np.uint8)

    if os.path.exists(os.path.join(working_path, 'roi.jpg')):
        roi = cv2.imread(os.path.join(working_path, 'roi.jpg'))

    # dieu chinh param cho bytetrack
    # BYTETracker(track_thresh, match_thresh, frame_rate=seq_info["frame_rate"])
    tracker = BYTETracker(track_thres, match_thres, frame_rate, alpha_fuse)

    tracking_results = []

    if save_video:
        out_video = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (w, h))

    if limit != -1:
        length = limit

    print('processing video...')
    for ix in tqdm(range(length), desc=" inner loop"):
        if 'flag' in kwargs:
            if kwargs['flag'] == False:
                break
        frame_cnt += 1
        flag, orig_img = video.read()
        img2 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img2 = img2 * (roi > 1)

        with torch.no_grad():
            results = detection_model.predict(img2, classes=class_IDS, conf=detection_thres, iou=iou_thres,verbose=verbose)
            detections = []
            batch = []

            for det_num, bb in enumerate(results[0].boxes):
                xyxyn = get_numpy(bb.xyxyn[0])

                xyxyn[0], xyxyn[2] = xyxyn[0] * w, xyxyn[2] * w
                xyxyn[1], xyxyn[3] = xyxyn[1] * h, xyxyn[3] * h
                xyxyn = xyxyn.astype(int)

                # tlwh = [xyxyn[0], xyxyn[1], xyxyn[2] - xyxyn[0], xyxyn[3]-xyxyn[1]]
                detection = list(xyxyn)
                detection.append(get_numpy(bb.conf)[0])
                detections.append(detection)

                obj = img2[xyxyn[1]:xyxyn[3], xyxyn[0]:xyxyn[2]]

                batch.append(transforms(obj))

                if save_video:
                    cv2.rectangle(orig_img, (xyxyn[0], xyxyn[1]), (xyxyn[2], xyxyn[3]), (0, 0, 255), thickness=2,
                                  lineType=cv2.LINE_8)
            ############
            feats = []
            if len(batch) == 0:
                continue

            input_tensor = torch.stack(batch, 0).to(device)
            feat = reid_model(input_tensor)
            # feat = torch.nn.functional.normalize(feat, dim=2)
            feat = torch.nn.functional.normalize(feat, dim=1)
            # feat = feat.mean(dim=1)
            feats = get_numpy(feat)

            online_targets = tracker.update(np.array(detections), np.array(feats), cid, use_embedding=True)

            # fair tracker
            #online_targets = tracker.update(np.array(detections), np.array(feats), frame_cnt)

            # Store results.
            for t in online_targets:
                tlwh, tid, score = t.det_tlwh, t.track_id, t.score
                feature = t.features[-1]

                if save_video:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (int(tlwh[0]), int(tlwh[1] - 30))
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(orig_img, f'ID: {tid}', org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)

                if tlwh[2] * tlwh[3] > 50:
                    tracking_results.append([
                        frame_cnt, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score, feature
                    ])
            if save_video:
                out_video.write(orig_img)

        ########
    del tracker
    return tracking_results
    # break


def run_video(working_path, reid_model, save_video_name=None, **kwargs):
    """
        This function is used to process a video for object tracking and returns the tracking results.

        Parameters:
        working_path (str): The path where the video file is located.
        reid_model_path (str): The path where the re-identification model is located.
        save_video_name (str): The name of the video file to be saved after processing.
        **kwargs: Arbitrary keyword arguments for future extension.

        Returns:
        list: Returns a list of tracking results. Each result is a list containing frame id, track id,
        top-left width and height of bounding box, score, and feature vector.

        Note:
        This function internally calls the `process_video` function with predefined parameters for object detection
        and tracking. The parameters are set as follows:
        verbose=False, detection_thres=0.2, iou_thres=0.7, track_thres=0.6, match_thres=0.4, frame_rate=10, limit=-1, alpha_fuse=0.8
        """
    detection_thres = 0.3
    iou_thres = 0.7
    track_thres = 0.5
    match_thres = 0.4
    frame_rate = 10
    limit = -1
    alpha_fuse = 0.8

    if 'detection_thres' in kwargs:
        detection_thres = kwargs['detection_thres']
    if 'iou_thres' in kwargs:
        iou_thres = kwargs['iou_thres']
    if 'track_thres' in kwargs:
        track_thres = kwargs['track_thres']
    if 'match_thres' in kwargs:
        match_thres = kwargs['match_thres']
    if 'frame_rate' in kwargs:
        frame_rate = kwargs['frame_rate']
    if 'limit' in kwargs:
        limit = kwargs['limit']
    if 'alpha_fuse' in kwargs:
        alpha_fuse = kwargs['alpha_fuse']

    detection_model = YOLO('yolov8x.pt')
    detection_model.to(device)
    detection_model.model.eval()
    if 'flag' in kwargs:
        flag = kwargs['flag']
    else:
        flag = True
    tracking_results = process_video(detection_model, reid_model, working_path, save_video_name, verbose=False,
                                     detection_thres=detection_thres,
                                     iou_thres=iou_thres,
                                     track_thres=track_thres,
                                     match_thres=match_thres,
                                     frame_rate=frame_rate,
                                     limit=limit,
                                     alpha_fuse=alpha_fuse,
                                     flag=flag)

    return tracking_results

