import os.path

import numpy as np
from PIL import Image
import pickle
from scipy.spatial import distance

from reid import make_model
from reid.resnet_50_vehiclenet import Resnet50
from torch.cuda import is_available
from torchvision.transforms import v2
import torch

import cv2
import os
import matplotlib.pyplot as plt


obj_size = [256, 256]

device = 'cpu'
if is_available():
    device = 'cuda'
device = 'cpu'

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=obj_size, antialias=True),
    # v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_numpy(tensor_):
    if device == 'cpu':
        return tensor_.detach().numpy()
    return tensor_.detach().cpu().numpy()


reid_model = None
results = None
result_idx = None

def search_vehicle(image: Image.Image, distance_threshold=0.5):
    # load pkl in output_pkl folder
    if not os.path.exists('output_pkl/matching_result_101.pkl'):
        return 'Missing files', 'Results file is missing'

    global results
    if results is None:
        with open('output_pkl/matching_result_101.pkl', 'rb') as f:
            results = pickle.load(f)

    global reid_model
    if reid_model is None:
        reid_model = make_model.make_model(1000, pretrained_path='./reid/resnet101_ibn_a_2.pth')
        reid_model.eval()
        reid_model.to(device)

        # reid_model = Resnet50('./reid/net_last.pth', device=device)
        # reid_model.eval()
        # reid_model.to(device)

    # resize image
    image = transforms(image).unsqueeze(0)
    # get feature of image
    with torch.no_grad():
        f = reid_model(image.to(device))

    f = torch.nn.functional.normalize(f, dim=1)
    f = get_numpy(f)

    # results = [track_cam_id_arr, track_id_arr, matcher.global_id_arr, track_st_zone, track_en_zone,
    #                     track_st_frame, track_en_frame, feat_dict, track_bboxes, track_scores]

    cam_arr, id_arr, globel_id_arr, feat_dict = results[0], results[1], results[2], results[7]
    bbox_arr = results[8]
    st_time_arr, en_time_arr = results[5], results[6]

    closet_distance = 1000
    index = -1
    for i in range(len(id_arr)):
        cid = cam_arr[i]
        tid = id_arr[i]
        boxes_distance = distance.cdist(f, feat_dict[cid][tid], 'euclid')
        mean_distance = np.mean(boxes_distance)
        if mean_distance < closet_distance:
            closet_distance = mean_distance
            index = i

    id = globel_id_arr[index]
    global result_idx
    result_idx = []

    for i in range(len(globel_id_arr)):
        if globel_id_arr[i] == id:
            result_idx.append(i)

    cams = cam_arr[result_idx]
    start_times = st_time_arr[result_idx]

    sorted_ix = np.argsort(start_times)
    cams = cams[sorted_ix]
    start_times = start_times[sorted_ix]

    # breakpoint()
    if closet_distance > distance_threshold:
        return -1
    return closet_distance, id, cams, start_times



def annotate(img, tlwh, id):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    xyxy = (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])
    xyxy = [int(i) for i in xyxy]
    #breakpoint()
    org = (xyxy[0], xyxy[1] - 10)

    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), thickness=2, \
                  lineType=cv2.LINE_8)
    #img = cv2.putText(img, f'ID: {id}', org, font, fontScale, color, thickness, cv2.LINE_AA)

    return img


def get_vehicle_images(cids, track_id, start_times, tlwh):
    images = {}
    for cid in cids:
        start_time = start_times[cid]
        video = cv2.VideoCapture(f'input/videos/c0{cid:02d}/vdo.avi')
        #fps = video.get(cv2.CAP_PROP_FPS)
        #frame = int(start_time * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_time)
        ret, frame = video.read()
        frame = annotate(frame, tlwh[cid], track_id)
        images[cid] = frame

    return images


def visualize_images_with_plt():
    global results
    global result_idx
    cam_arr, id_arr, globel_id_arr = results[0], results[1], results[2]
    bbox_arr = results[8]
    track_scores = results[9]
    track_frames = results[10]
    st_time_arr = results[5]

    track_id = globel_id_arr[result_idx[0]]

    track_start_time = {}
    bboxes = {}
    track_cam = []
    for i in result_idx:
        cam = cam_arr[i]
        track_cam.append(cam)
        length = len(bbox_arr[i])
        max_score = 0
        offset = 0
        for j in range(0, length):
            if track_scores[i][j] > max_score:
                max_score = track_scores[i][j]
                offset = j

        track_start_time[cam] = track_frames[i][offset]
        bboxes[cam] = bbox_arr[i][offset + 1]
        print(f'cam: {cam}  {len(bbox_arr[i])} {offset}')


    imgs = get_vehicle_images(track_cam, track_id, track_start_time, bboxes)

    # Visualize with matplotlib
    rows = len(track_cam) // 3 + 1
    cols = 3 if len(track_cam) > 3 else len(track_cam)

    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    if len(track_cam) == 1:
        cid, img = list(imgs.items())[0]
        axs.imshow(cv2.cvtColor(imgs[track_cam[0]], cv2.COLOR_BGR2RGB))
        t = f'{track_start_time[cid] // 3600}:{track_start_time[cid] // 60}:{track_start_time[cid] % 60}'
        axs.set_title(f'Cam: {cid}\nStart time: {t}')
        axs.axis('off')
        axs.set_axis_off()
    else:
        for ax, (cid, img) in zip(axs.flat, imgs.items()):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            t = f'{track_start_time[cid]//3600}:{track_start_time[cid]//60}:{track_start_time[cid]%60}'
            ax.set_title(f'Cam: {cid}\nTime: {t}')
            ax.axis('off')
            ax.set_axis_off()

        for ax in axs.flat[len(track_cam):]:
            ax.axis('off')
            #ax.imshow(np.ones((256, 256, 3)) * 255)


    fig.show()

#
# image =  Image.open('query_image_2.png').convert('RGB')
# res = search_vehicle(image)
#
# visualize_images_with_plt()