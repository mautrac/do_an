import queue

import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from functools import reduce

import time
import threading


def resize_with_max_size(img, max_size):
    h, w = img.shape[:2]
    max_w, max_h = max_size
    # downscale
    if h > max_h or w > max_w:
        r = min(max_w / w, max_h / h)
        new_w = int(w * r)
        new_h = int(h * r)
        return cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)

    # upscale
    r = max(max_w / w, max_h / h)
    new_w = int(w * r)
    new_h = int(h * r)
    img = cv2.resize(img, (new_w, new_h))

    return img

class BigVideo:
    def __init__(self, grid, videos, out_video, out_size) -> None:
        ws = {}
        hs = {}
        cids = list(videos.keys())
        cids.sort()

        for cid, video in videos.items():
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ws[cid] = w
            hs[cid] = h

        coords = {}

        print(cids)
        cell_x = out_size[0] / grid[1]
        cell_y = out_size[1] / grid[0]
        for y in range(grid[0]):
            for x in range(grid[1]):
                id = y * grid[1] + x
                cid = cids[id]
                coords[cid] = (int(cell_x * x), int(cell_y * y))

        self.coords = coords
        self.ws = ws
        self.hs = hs
        self.cell_size = (cell_x, cell_y)
        self.out_size = out_size


        self.out_video = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'XVID'), 10, out_size)
        self.cur_frame = 0

        print(f'out_size: {self.out_size[0]}x{self.out_size[1]}')
        self.orig_frame = np.zeros((out_size[1], out_size[0], 3))

        self.imgs = {}
        self.write_thread = None

    def write(self, imgs):
        self.imgs = imgs
        # ???ms
        frame = self.orig_frame
        for id, img in imgs.items():
            if img is None:
                continue
            x, y = self.coords[id]
            w, h = self.ws[id], self.hs[id]
            #breakpoint()
            img = resize_with_max_size(img, self.cell_size)
            img[:40,:220] = 0
            img = cv2.putText(img, f'cam: {id}', (0, 35), font, 1.5, (0, 0, 255), thickness, cv2.LINE_AA)
            new_h, new_w = img.shape[:2]
            frame[y: y + new_h, x: x + new_w] = img
            # print(f'{x} {y} {w} {h}')
            # cv2.imwrite(f'img_{id}.jpg', img)

        # 100ms
        #frame = cv2.resize(frame, self.out_video, interpolation=cv2.INTER_AREA)
        # cv2.imwrite(f'images/img_{self.cur_frame}.jpg', frame)
        frame = np.uint8(frame)


        # if self.write_thread is not None:
        #     self.write_thread.join()
        #
        # if frame.size > 1:
        #     self.write_thread = threading.Thread(None, target=self.out_video.write, args=[frame, ])
        #     self.write_thread.start()

        # 20ms
        self.cur_frame += 1
        self.out_video.write(frame)



        # breakpoint()

    def close(self):
        self.out_video.release()


def interpolate_frame(img1, img2):
    optical_flow = cv2.calcOpticalFlowFarneback(img1, img2, None, pyr_scale=0.5, levels=3, winsize=200,
                                                iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

    num_frames = 10
    h, w = optical_flow.shape[:2]
    for frame_num in range(1, num_frames + 1):
        alpha = frame_num / num_frames
        flow = alpha * optical_flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        interpolated_frame = cv2.remap(img2, flow, None, cv2.INTER_LINEAR)


def processing(cid, label, cur_idx, img, cur_frame):
    i = int(cur_idx[cid])
    while label.iloc[i]['frame'] < cur_frame[cid]:
        i += 1
        if i >= len(label):
            break

    while i < len(label) and label.iloc[i]['frame'] == cur_frame[cid]:
        # breakpoint()
        tid = label.iloc[i]['track']
        tlwh = label.iloc[i].values[3:7]
        xyxy = (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])

        org = (xyxy[0], xyxy[1] - 10)

        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), thickness=2, \
                      lineType=cv2.LINE_8)
        img = cv2.putText(img, f'ID: {tid}', org, font, fontScale, color, thickness, cv2.LINE_AA)

        i += 1
        if i >= len(label):
            break
        # breakpoint()

    return cid, img, i


if __name__ == '__main__':
    time_stamp = {15:1, 14:1, 13: 115, 12: 51, 11: 11, 10: 1}

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    camera_ids = [10, 11, 12, 13]

    path = 'input/videos'
    videos = {}

    for folder in os.listdir(path):
        cid = int(folder[-2:])
        if cid in camera_ids:
            v = cv2.VideoCapture(path + '/' + folder + '/' + 'vdo.avi')
            v.set(cv2.CAP_PROP_POS_FRAMES, time_stamp[cid] - 1)
            videos[cid] = v

    data = np.loadtxt('output_ica.txt', delimiter=' ', dtype=int)
    names = ['cam', 'track', 'frame', 'top', 'left', 'width', 'height', 'x', 'y']
    df = pd.DataFrame(data, columns=names)

    idx = df['cam'].isin(camera_ids)
    df = df[idx]

    grouped = df.groupby('cam')
    labels = {}
    for cid, group in grouped:
        sorted_group = group.sort_values('frame')
        labels[cid] = sorted_group

    cur_frame = {}
    cur_idx = {}
    for cid in camera_ids:
        #index = labels[cid]['frame'] == time_stamp[cid]
        cur_idx[cid] = 1
        #breakpoint()
        cur_frame[cid] = time_stamp[cid]

    # (2560 + 2560, 1920 + 1920) = (5120, 3840)
    # (5120, 3840) / 2 = (2560, 1920)
    big_video = BigVideo((2, 2), videos=videos, out_video='big_video2.avi', out_size=(2560, 1920))
    pbar = tqdm(total=2500)


    def wrap_cv2_read(cid, video):
        flag, frame = video.read()
        q.put((cid, flag, frame))

    q = queue.Queue()
    imgs = {}
    frame_cnt = 0

    while len(videos):
        drops = []
        # 40ms on threads
        threads = []
        for cid, video in videos.items():
            thread = threading.Thread(None, target=wrap_cv2_read, args=[cid, video])
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        while not q.empty():
            cid, flag, frame = q.get()
            if not flag:
                drops.append(cid)
            imgs[cid] = frame

        # 80ms on single thread
        # for cid, video in videos.items():
        #     flag, frame = video.read()
        #     # print(frame.shape)
        #     imgs[cid] = frame
        #     if not flag:
        #         drops.append(cid)

        for cid in drops:
            videos.pop(cid)
            labels.pop(cid)

        # 10ms
        for cid, label in labels.items():
            i, im, ix = processing(cid, label, cur_idx, imgs[cid], cur_frame)
            imgs[i] = im
            cur_idx[i] = ix
            cur_frame[cid] += 1

        # ???ms
        big_video.write(imgs)
        pbar.update(1)
        frame_cnt += 1

        if frame_cnt == 5000:
            break

    big_video.close()




