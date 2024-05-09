import os
import cv2
from PIL import Image
import sys
sys.path.append('../')
def resize_image(image, max_width, max_height, min_width=100, min_height=100):
    w, h = image.size
    i = 2
    new_w, new_h = w, h
    while new_w > max_width or new_h > max_height:
        new_w = w // i
        new_h = h // i
        i += 1
    while new_w < min_width or new_h < min_height:
        new_w = w * i
        new_h = h * i
        i += 1
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


def get_images_for_gallery():
    images = []
    names = []
    for d in os.listdir('input/videos'):
        cam_id = int(d[-2:])
        for f in os.listdir(f'input/videos/{d}'):
            video = cv2.VideoCapture(f'input/videos/{d}/{f}')
            ret, frame = video.read()
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame))
                names.append(f'Camera {cam_id}')
                break
    return images, names


def check_files():
    for d in os.listdir('input/videos'):
        cam_id = int(d[-2:])
        if not os.path.exists(f'input/videos/{d}/vdo.avi'):
            return 'Missing files', f'Camera {cam_id} video is missing files'
        if not os.path.exists(f'input/videos/{d}/mask_zone.jpg'):
            return 'Missing files', f'Camera {cam_id} mask is missing files'

    if not os.path.exists('../input/cam_infor/adjacent_list.txt'):
        return 'Missing files', 'Adjacent list is missing files'
    if not os.path.exists('../input/cam_infor/travel_time_loose_constraint.txt.txt'):
        return 'Missing files', 'Travel time loose is missing files'
    if not os.path.exists('../input/cam_infor/travel_time_hard_constraint.txt.csv'):
        return 'Missing files', 'Travel time hard is missing files'

    return 'Success', 'All files are present'
