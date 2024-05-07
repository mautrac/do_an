
import pickle
import numpy as np
import os
import cv2

time_stamp = {
    10: int(8.715 * 10),
    11: int(8.457 * 10),
    12: int(5.879 * 10),
    13: 0,
    14: int(5.042 * 10),
    15: int(8.492 * 8)
}

adjacent_matrix = [
    [-1, 1, -1, -1, -1, -1]
    [2, -1, 1, -1, -1, -1]
    [-1, 2, -1, 1, -1, -1]
    [-1, -1, 2, -1, 1, 4]
    [-1, -1, -1, 2, -1, -1]
    [-1, -1, -1, 2, 1, -1]
]

moving_time_threshold = {
    {10, 11}: 1 * 10,
    {11, 12}: 1 * 10
}

with open('../temp.pkl', 'rb') as f:
    cam_dict_results = pickle.load(f)



mask_matrix = {}
_path = '../mask_zone'
for d in os.listdir(_path):
    mask = cv2.imread(_path + '/' + d, cv2.IMREAD_GRAYSCALE)
    s = d.split('.')[0][-2:]
    s = int(s)
    mask = np.where(mask > 5, 0, mask)
    mask_matrix[s] = mask

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3, 2, figsize=(20,20))
# for i, (key, value) in enumerate(mask_matrix.items()):
#     axes[i // 2, i % 2].set_title(f"Cam {key}")
#     axes[i // 2, i % 2].imshow(value)
#
# fig.show()

class Tracklet(object):
    def __init__(self, cam_id, x, y, fr_id):
        self.cam_id = cam_id
        self.st_frame = fr_id
        self.en_frame = -1
        self.st_id = -1  # the "in port" id when the track appears in the image at the start
        self.en_id = -1  # the "out port" id when the track disappears in the image at the end
        self.select_st_id(x, y)

    def select_st_id(self, x, y):
        self.st_id = mask_matrix[self.cam_id][x][y]

    def select_en_id(self, x, y):
        self.en_id = mask_matrix[self.cam_id][x][y]

    def add_element(self, x, y, fr_id):
        self.en_frame = fr_id
        self.select_en_id(x, y)


cam_dict_tracklet = {}
for cam_id, tracks in cam_dict_results.items():
    cam_dict_tracklet.setdefault(cam_id, {})
    for track in tracks:
        t, l, w, h = track[2:6]
        yc = int(t + w // 2)
        xc = int(l + h // 2)
        if track[1] not in cam_dict_tracklet[cam_id].keys():
            cam_dict_tracklet[cam_id].setdefault(track[1], Tracklet(cam_id, xc, yc, track[0]))
        else:
            cam_dict_tracklet[cam_id][int(track[1])].add_element(xc, yc, track[0])


for k in cam_dict_tracklet.keys():
    print(len(cam_dict_tracklet[k]))

records = 0
for k in cam_dict_results.keys():
    records += len(cam_dict_results[k])

box_cam_id_arr, box_id_arr, box_feat_arr = np.zeros(records, dtype=int), np.zeros(records, dtype=int), np.zeros((records, 2048))
box_st_frame, box_en_frame, box_st_zone, box_en_zone = np.zeros(records, dtype=int), np.zeros(records, dtype=int), np.zeros(records, dtype=int),np.zeros(records, dtype=int)

idx = 0
for key, tracks in cam_dict_results.items():
    for track in tracks:
        box_cam_id_arr[idx] = key
        box_id_arr[idx] = track[1]

        #print(cam_dict_tracklet[key][track[1]].en_frame)
        box_st_frame[idx] = cam_dict_tracklet[key][track[1]].st_frame
        box_en_frame[idx] = cam_dict_tracklet[key][track[1]].en_frame

        box_st_zone[idx] = cam_dict_tracklet[key][track[1]].st_id
        box_en_zone[idx] = cam_dict_tracklet[key][track[1]].en_id

        box_feat_arr[idx] = track[7]
        idx += 1


def fusion(cam_id_1, cam_id_2):
    c1_idx = box_cam_id_arr == cam_id_1
    c2_idx = box_cam_id_arr == cam_id_2




#cam_arr, track_arr, in_dir_arr, out_dir_arr, in_time_arr, out_time_arr, feat_arr,


with open('../c041_dets_feat.pkl', 'rb') as f:
    test_ = pickle.load(f)

