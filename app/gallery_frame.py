import tkinter as tk
from app import utils
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
#import sys
from process_input import process_scmt
from ICA import ica
#sys.path.append('../')


class SingleImageFrame(tk.Frame):
    def __init__(self, master=None, image=None, name=None):
        img = utils.resize_image(image, 350, 350)
        w, h = img.size
        super().__init__(master, padx=10, pady=10, width=w, height=h + 20)
        self.master = master

        self.canvas = tk.Canvas(self, bg='white', height=h, width=w, borderwidth=0, highlightthickness=0)
        photo = ImageTk.PhotoImage(img)
        self.canvas.image = photo
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')
        self.canvas.config(width=w, height=h)
        #self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        #self.canvas.grid(row=0, column=0, padx=5, pady=5)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.label = tk.Label(self, text=name, bg='white')
        #self.label.grid(row=1, column=0, padx=5, pady=5)
        self.label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)


class GalleryFrame(tk.Frame):
    def __init__(self, master=None, images=None, names=None):
        super().__init__(master, bg="#e3e1ca", width=800)
        self.master = master
        self.images = images
        self.names = names
        button_width = 15

        self.canvas = tk.Canvas(self, width=780, height=440, bg="#e3e1ca")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        if images is not None:
            for i in range(len(images)):
                #img = Image.open('../image/no_image.png')
                SingleImageFrame(self.scrollable_frame, images[i], names[i]).grid(row=i // 2, column=i % 2)


    def update(self, images, names):
        self.images = images
        self.names = names
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        for i in range(len(images)):
            SingleImageFrame(self.scrollable_frame, images[i], names[i]). \
             grid(row=i // 2, column=i % 2, padx=5, pady=5)


class CompositeFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, bg="#e3e1ca", width=800, height=540)
        self.master = master
        self.images = None
        self.names = None
        self.gallery = GalleryFrame(self)
        self.gallery.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.gallery.scrollbar.tkraise()

        button_width = 15

        self.buttons_frame = tk.Frame(self, bg="#91c1e3", width=800, height=100)
        self.buttons_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.load_button = tk.Button(self.buttons_frame, text="Load", command=self.load_data, borderwidth=5, underline=0, width=button_width)
        self.load_button.place(relx=0.25, rely=0.5, anchor=tk.CENTER)

        self.process_button = tk.Button(self.buttons_frame, text="Process", command=self.process_images, borderwidth=5, underline=0, width=button_width)
        self.process_button.place(relx=0.75, rely=0.5, anchor=tk.CENTER)


    def update(self):
        self.gallery.update(self.images, self.names)

    def show_frame(self):
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #self.tkraise()

    def hide_frame(self):
        self.pack_forget()

    def load_data(self):
        images, names = utils.get_images_for_gallery()
        self.images = images
        self.names = names
        self.update()
        utils.check_files()

    def process_images(self):
        process_scmt()
        ica.run()



# root = tk.Tk()
# root.title("Application")
# root.geometry("960x540")
# root.resizable(width=False, height=False)
#
# gallery_frame = CompositeFrame(root)
# gallery_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# #images, names = utils.get_images_for_gallery()
# #gallery_frame.update(images, names)
# gallery_frame.load_data()
# gallery_frame.update()
#
# root.mainloop()
#
# SingleImageFrame(root, images[0], names[0]).pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)
