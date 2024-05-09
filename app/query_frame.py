import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk
from PIL import Image
from app import utils


class QueryFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, bg="#e3e1ca", width=800, height=540)

        self.master = master
        self.working_image = None

        self.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.place_forget()

        button_width = 15

        self.frame_img = tk.Frame(self, bg="#ffffff", bd=2, width=800, height=440, padx=5, pady=5)
        #self.frame_img.grid(row=0, column=0, columnspan=3, sticky='news', padx=5, pady=10)
        self.frame_img.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_for_image = tk.Canvas(self.frame_img, bg='white', height=400, width=400, borderwidth=1, highlightthickness=2)
        self.canvas_for_image.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        #self.load_image('../image/no_image.png')
        self.working_image = None

        self.buttons_frame = tk.Frame(self, bg="#91c1e3", width=800, height=100)
        self.buttons_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.open_button = tk.Button(self.buttons_frame, text="Open", command=self.open_file_dialog, borderwidth=5, underline=0, width=button_width)
        #self.open_button.grid(row=1, column=1, sticky='news', padx=5, pady=10)
        #self.open_button.pack(side=tk.LEFT, expand=False, pady=5)
        self.open_button.place(relx=0.25, rely=0.5, anchor=tk.CENTER)

        self.process_button = tk.Button(self.buttons_frame, text="Search", command=self.process_image, borderwidth=5, underline=0, width=button_width)
        #self.process_button.grid(row=2, column=1, sticky='news', padx=5, pady=10)
        #self.process_button.pack(side=tk.TOP, expand=False, pady=5)
        self.process_button.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

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
        print("Processing image...")
        # Do something with the image
