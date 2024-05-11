from tqdm import tqdm
import os
from run_scmt import run_video
from reid.make_model import make_model
from reid.resnet_50_vehiclenet import Resnet50
import pickle
from torch.cuda import is_available
import tkinter as tk
from tkinter import ttk
import threading

path = './input/videos'
flag = True
progress = 0
progress_var = None

if is_available():
    device = 'cuda'
else:
    device = 'cpu'

def quit(popup):
    popup.destroy()
    global flag
    flag = False

def ui_thread(popup, dirs):
    tk.Label(popup, text="Processing video").pack(side=tk.TOP, pady=10, padx=10)
    # place(relx=0.5, rely=0.3, anchor=tk.CENTER))
    global progress_var
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=dirs)
    # progress_bar.grid(row=1, column=0)
    progress_bar.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    cancel_button = tk.Button(popup, text="Cancel", command=lambda :quit(popup))
    cancel_button.pack(side=tk.BOTTOM, pady=10)

    popup.pack_slaves()


def process_scmt():
    global progress
    global flag
    global progress_var

    dirs = os.listdir('./input').__len__()

    # start progress bar
    #popup = tk.Toplevel(height=200, width=200)

    progress_step = float(1 / dirs)

    cam_dict_results = {}
    #th1 = threading.Thread(target=ui_thread, args=(popup, dirs))
    #th1.start()

    reid_model = make_model(1000, pretrained_path='./reid/resnet101_ibn_a_2.pth')
    reid_model.to(device)
    reid_model.eval()

    # reid_model = Resnet50('./reid/net_last.pth', device)
    # reid_model.to(device)
    # reid_model.eval()

    for d in os.listdir(path):
        #popup.update()
        #progress += progress_step
        #progress_var.set(progress)
        cam_id = int(d[-2:])

        cam_dict_results[cam_id] = run_video(path + '/' + d, reid_model,
                                             save_video_name=False,
                                             detection_thres=0.4,
                                             iou_thres=0.7,
                                             track_thres=0.6,
                                             match_thres=0.35,
                                             alpha_fuse=0.3,
                                             frame_rate=40,
                                             flag=flag)

    with open('./output_pkl/results_scmt.pkl', 'wb') as f:
        pickle.dump(cam_dict_results, f)
    #popup.destroy()
    #th1.join()

    return True