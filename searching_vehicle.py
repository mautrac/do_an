import os.path

import numpy as np
from PIL import Image
import pickle
from scipy.spatial import distance

from reid import make_model

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



def search_vehicle(image: Image.Image, distance_threshold=0.5):
    # load pkl in output_pkl folder
    if not os.path.exists('output_pkl/results_scmt.pkl'):
        return 'Missing files', 'Results file is missing'

    with open('output_pkl/results_scmt.pkl', 'rb') as f:
        results = pickle.load(f)

    reid_model = make_model.make_model(1000, pretrained_path='./reid/resnet101_ibn_a_2.pth')
    reid_model.eval()
    reid_model.to(device)

    # resize image
    image = transforms(image).unsqueeze(0)
    # get feature of image
    with torch.no_grad():
        f = reid_model(image.to(device))

    f = torch.nn.functional.normalize(f, dim=1)
    f = get_numpy(f)

    # get records from results
    n = 0
    for cam_id in results:
        n += len(results[cam_id])

    cam_arr, id_arr, feat_arr = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros((n, 2048), dtype=float)
    i = 0
    for cid in results:
        for row in results[cid]:
            cam_arr[i] = cid
            id_arr[i] = row[1]
            feat_arr[i] = row[7]
            i += 1

    breakpoint()
    # rerank

    dist_mat = distance.cdist(f, feat_arr, 'euclidean')
    del feat_arr

    j = 0
    closet_id = -1
    closet_distance = 1000
    index = -1
    for i in range(n - 1):
        if id_arr[i + 1] != id_arr[i]:
            sum_distance = np.mean(dist_mat[0][j:i+1])
            if sum_distance < closet_distance:
                closet_distance = sum_distance
                closet_id = id_arr[i]
                index = j
            j = i + 1
    breakpoint()
    if closet_distance < distance_threshold:
        return -1, -1, -1
    return closet_distance, closet_id, index



image =  Image.open('query_image_2.png').convert('RGB')


res = search_vehicle(image)
