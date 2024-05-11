import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk
from PIL import Image
from app import utils
import searching_vehicle
from app.components import InputComponent


def validate_distance_threshold(value):
    try:
        value = float(value)
        if value < 0.0:
            return False
        return True
    except ValueError:
        return False

class QueryFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, bg="#e3e1ca", width=800, height=540)

        self.master = master
        self.working_image = None
        self.distance_threshold = tk.DoubleVar()
        self.distance_threshold.set(0.5)

        self.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.place_forget()

        button_width = 15

        self.frame_img = tk.Frame(self, bg="#ffffff", bd=2, width=600, height=540, padx=5, pady=5)
        #self.frame_img.grid(row=0, column=0, columnspan=3, sticky='news', padx=5, pady=10)
        self.frame_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_for_image = tk.Canvas(self.frame_img, bg='white', height=400, width=400, borderwidth=1, highlightthickness=2)
        self.canvas_for_image.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        #self.load_image('../image/no_image.png')
        self.working_image = None


        self.left_frame = tk.Frame(self, bg="#91c1e3", width=200, height=540)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipadx=5, padx=5)

        self.input_label = tk.Label(self.left_frame, text="Input params", font=("bold", 15), bg="#91c1e3")
        self.input_label.pack(side=tk.TOP, pady=10)


        self.distance_threshold_input = InputComponent(self.left_frame, label='Distance threshold',
                                                       validate_function=None)
        self.distance_threshold_input.pack(side=tk.TOP, pady=10)

        self.process_button = tk.Button(self.left_frame, text="Search", command=self.process_image, borderwidth=5, underline=0, width=button_width)
        #self.process_button.grid(row=2, column=1, sticky='news', padx=5, pady=10)
        self.process_button.pack(side=tk.BOTTOM, expand=False, pady=5)
        #self.process_button.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

        self.open_button = tk.Button(self.left_frame, text="Open", command=self.open_file_dialog, borderwidth=5, underline=0, width=button_width)
        #self.open_button.grid(row=1, column=1, sticky='news', padx=5, pady=10)
        self.open_button.pack(side=tk.BOTTOM, expand=False, pady=5)
        #self.open_button.place(relx=0.25, rely=0.5, anchor=tk.CENTER)


        # self.quit_button = tk.Button(self, text="Close", background='#cf2727', command=self.lower_frame, borderwidth=5, underline=0, width=button_width)
        # self.quit_button.grid(row=3, column=1, sticky='news', padx=5, pady=(30, 0))
        # self.quit_button.pack(side=tk.TOP, expand=False, pady=(30, 0))


    def show_frame(self):
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #self.tkraise()

    def hide_frame(self):
        self.pack_forget()

    def load_image(self, file_path):
        if os.path.exists(file_path):
            image = Image.open(file_path)
            self.working_image = image.copy()
            image = utils.resize_image(image, 400, 400)
            w, h = image.size
            photo = ImageTk.PhotoImage(image)

            self.canvas_for_image.image = photo
            self.canvas_for_image.create_image(0, 0, image=self.canvas_for_image.image, anchor='nw')
            self.canvas_for_image.config(width=w, height=h)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.load_image(file_path)

    def process_image(self):
        if not self.working_image:
            messagebox.showwarning("Warning", "Invalid image! Please choose an image.")
            return

        distance_threshold = self.distance_threshold_input.get_value()
        if not validate_distance_threshold(distance_threshold):
            messagebox.showwarning("Warning", "Invalid distance threshold! Please enter a value larger than 0.")
            return

        print("Processing image...")
        results = searching_vehicle.search_vehicle(self.working_image.convert('RGB'), distance_threshold)
        if not isinstance(results, tuple):
            messagebox.showinfo("Results", 'Not found')
            return
        closet_distance, id, cams, start_times = results

        message = (f"ID: {id}\n"
                   f"Cams: {cams}\n"
                   f"Start times: {start_times}\n")
        messagebox.showinfo('Results', message)

