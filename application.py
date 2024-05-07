import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def open_file_dialog():
    # Create a new window
    new_window = tk.Toplevel(root)
    new_window.title("Open File")
    # Hide the new window
    new_window.withdraw()
    # Open the file dialog
    file_path = filedialog.askopenfilename()
    # Show the new window
    new_window.deiconify()
    return file_path

# Create the main window
root = tk.Tk()
root.title("Application")
root.geometry("1280x720")

iconSave = ImageTk.PhotoImage(Image.open("image/icons8-save-50 (1).png").resize((40, 40)))
iconCancel = ImageTk.PhotoImage(Image.open("image/icons8-cancel-16.png").resize((40, 40)))


# Create a frame to hold the buttons
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Configure the frame's grid to expand as the window is resized
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

bg = tk.PhotoImage(file="./image/bg.bmp")
bg_label = tk.Label( root, image = bg)
bg_label.place(x=-5, y=-5)


labelName = tk.Label(root, text="Click vào nút để lựa chọn theo dự đoán: ", font=('Arial', 14))
labelName.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

img = ImageTk.PhotoImage(Image.open("image/icons8-exit-16.png").resize((40, 40)))
iconImage = ImageTk.PhotoImage(Image.open("image/icons8-image-100.png").resize((40, 40)))
buttonImage = tk.Button(root, text='Chọn ảnh', width=200, font=('Arial', 14), image=iconImage, compound=tk.LEFT,
                     bg="#51aded", fg="white", command=open_file_dialog)
buttonImage.place(relx=0.5, rely=0.3, anchor=tk.CENTER)




# Start the main loop
root.mainloop()