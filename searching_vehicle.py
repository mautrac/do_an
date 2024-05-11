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

obj_size = [256, 256]

device = 'cpu'
if is_available():
    device = 'cuda'


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

reid_model = None

def search_vehicle(image: Image.Image, distance_threshold=0.5):
    # load pkl in output_pkl folder
    if not os.path.exists('output_pkl/matching_result_101.pkl'):
        return 'Missing files', 'Results file is missing'

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

    # matching_results = [track_cam_id_arr, track_id_arr, matcher.global_id_arr, track_st_zone, track_en_zone,
    #                     track_st_frame, track_en_frame, feat_dict]
    cam_arr, id_arr, globel_id_arr, feat_dict = results[0], results[1], results[2], results[7]
    st_time_arr, en_time_arr = results[5], results[6]

    #breakpoint()
    # rerank

    closet_distance = 1000
    index = -1
    for i in range(len(id_arr) ):
        cid = cam_arr[i]
        tid = id_arr[i]
        boxes_distance = distance.cdist(f, feat_dict[cid][tid], 'euclid')
        mean_distance = np.mean(boxes_distance)
        if mean_distance < closet_distance:
            closet_distance = mean_distance
            index = i

    id = globel_id_arr[index]
    ix = [index]
    for i in range(len(globel_id_arr)):
        if globel_id_arr[i] == id:
            ix.append(i)

    cams = cam_arr[ix]
    start_times = st_time_arr[ix]

    sorted_ix = np.argsort(start_times)
    cams = cams[sorted_ix]
    start_times = start_times[sorted_ix]

    #breakpoint()
    if closet_distance > distance_threshold:
        return -1
    return closet_distance, id, cams, start_times



#image =  Image.open('query_image_3.png').convert('RGB')
#res = search_vehicle(image)
