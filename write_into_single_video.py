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

class BigVideo:
    def __init__(self, grid, videos, out_video) -> None:
        ws = {}
        hs = {}
        cids = list(videos.keys())
        for cid, video in videos.items():
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ws[cid] = w
            hs[cid] = h

        w_total = 0
        h_total = 0
        offset_x = 0
        offset_y = 0
        coords = {}

        print(cids)

        for y in range(grid[0]):
            max_h = 0
            max_w = 0
            offset_x = 0
            for x in range(grid[1]):
                id = y * grid[1] + x
                cid = cids[id]

                coords[cid] = (offset_x, offset_y)
                offset_x += ws[cid] + 10

                max_w += ws[cid] + 10
                if max_h < hs[cid]:
                    max_h = hs[cid]
            if max_w > w_total:
                w_total = max_w

            h_total += max_h + 20
            offset_y += max_h + 20

        self.coords = coords
        self.h_total = h_total
        self.w_total = w_total
        self.ws = ws
        self.hs = hs

        factor = 2
        while w_total // factor > 2560 or h_total // factor > 1440:
            factor += 1
        self.factor = factor

        self.new_size = (w_total // factor, h_total // factor)
        self.out_video = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'), 10, self.new_size)
        self.cur_frame = 0

        print(f'{h_total} {w_total} {self.new_size[0]} {self.new_size[1]}')
        self.orig_frame = np.zeros((self.h_total, self.w_total, 3))

    def write(self, imgs):

        # 180ms
        frame = self.orig_frame
        for id, img in imgs.items():
            x, y = self.coords[id]
            w, h = self.ws[id], self.hs[id]
            frame[y: y + h, x: x + w] = img
            # print(f'{x} {y} {w} {h}')
            # cv2.imwrite(f'img_{id}.jpg', img)


        # 100ms
        frame = cv2.resize(frame, self.new_size)
        # cv2.imwrite(f'images/img_{self.cur_frame}.jpg', frame)
        frame = np.uint8(frame)

        # 20ms
        self.cur_frame += 1
        self.out_video.write(frame)
        # breakpoint()

    def close(self):
        self.out_video.release()



def processing(cid, label, cur_idx, img, cur_frame):
    i = int(cur_idx[cid])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    while label.iloc[i]['frame'] == cur_frame:
        # breakpoint()
        tid = label.iloc[i]['track']
        tlwh = label.iloc[i].values[3:7]
        xyxy = (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])

        org = (xyxy[0], xyxy[1] - 10)

        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), thickness=2, \
                      lineType=cv2.LINE_8)
        img = cv2.putText(img, f'ID: {tid}', org, font, fontScale, color, thickness, cv2.LINE_AA)

        cur_idx[cid] += 1
        i += 1
        if i >= len(label):
            break
        # breakpoint()

    return cid, img, cur_idx[cid]


if __name__ == '__main__':
    path = 'input/videos'
    videos = {}

    for folder in os.listdir(path):
        cid = int(folder[-2:])
        v = cv2.VideoCapture(path + '/' + folder + '/' + 'vdo.avi')
        videos[cid] = v

    data = np.loadtxt('output_ica.txt', delimiter=' ', dtype=int)
    names = ['cam', 'track', 'frame', 'top', 'left', 'width', 'height', 'x', 'y']
    df = pd.DataFrame(data, columns=names)

    camera_ids = [10, 11, 12, 13, 14, 15]
    idx = df['cam'].isin(camera_ids)
    df = df[idx]

    grouped = df.groupby('cam')
    labels = {}
    for cid, group in grouped:
        sorted_group = group.sort_values('frame')
        labels[cid] = sorted_group

    cur_frame = 0
    cur_idx = {}
    for cid in camera_ids:
        cur_idx[cid] = 1



    big_video = BigVideo((2, 3), videos=videos, out_video='big_video2.mp4')
    pbar = tqdm(total=2500)

    cur_frame = 1

    def wrap_cv2_read(cid, video):
        flag, frame = video.read()
        q.put((cid, flag, frame))

    q = queue.Queue()

    while len(videos):
        imgs = {}
        drops = []
        # 40ms on thread
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

        # ???ms
        big_video.write(imgs)
        pbar.update(1)
        cur_frame += 1

        # if cur_frame == 500:
        #     break

    big_video.close()




