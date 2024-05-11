import tkinter
from app.query_frame import QueryFrame
from app.scmt_window import CompositeFrame
from tkinter import ttk

root = tkinter.Tk()
root.title("Application")
root.geometry("960x540")
root.resizable(width=False, height=False)
root.config(bg="#e3e1ca")

options_frame = tkinter.Frame(root, bg='#c3c3c3', width=160, height=540)
options_frame.pack(side=tkinter.LEFT)
options_frame.pack_propagate(False)

main_frame = tkinter.Frame(root, bg='#e3e1ca', width=800, height=540, borderwidth=(2), relief=tkinter.SOLID)
main_frame.pack(side=tkinter.RIGHT)
main_frame.pack_propagate(False)


query_frame = QueryFrame(main_frame)
gallery_frame = CompositeFrame(main_frame)
gallery_frame.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
#gallery_frame.load_data()
#gallery_frame.update()
#
# scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=gallery_frame.gallery.canvas.yview)
# gallery_frame.gallery.canvas.configure(yscrollcommand=scrollbar.set)
# scrollbar.pack(side="right", fill="y")

def hide_indicators():
    scmt_indicator.config(bg='#c3c3c3')
    search_indicator.config(bg='#c3c3c3')
    #ica_indicator.config(bg='#c3c3c3')


def show_indicator(indicator):
    hide_indicators()
    indicator.config(bg='#158aff')


scmt_btn = tkinter.Button(options_frame, text="RUN", font=("bold", 15), bd=0, fg='#158aff', bg='#c3c3c3',
                          command=lambda: {show_indicator(scmt_indicator), gallery_frame.show_frame(), query_frame.hide_frame()}, width=10)
scmt_btn.place(x=20, y=50)

scmt_indicator = tkinter.Label(options_frame, text="", bg='#158aff')
scmt_indicator.place(x=10, y=50, width=5, height=35)

# ica_btn = tkinter.Button(options_frame, text="ICA", font=("bold", 15), bd=0, fg='#158aff', bg='#c3c3c3', width=10,
#                          command=lambda: {show_indicator(ica_indicator), gallery_frame.hide_frame(),
#                                           query_frame.hide_frame()})
# ica_btn.place(x=20, y=100)
#
# ica_indicator = tkinter.Label(options_frame, text="", bg='#c3c3c3')
# ica_indicator.place(x=10, y=100, width=5, height=35)


search_btn = tkinter.Button(options_frame, text="Search", font=("bold", 15), bd=0, fg='#158aff', bg='#c3c3c3',
                          command=lambda: {show_indicator(search_indicator), query_frame.show_frame(), gallery_frame.hide_frame()}, width=10)
search_btn.place(x=20, y=150)

search_indicator = tkinter.Label(options_frame, text="", bg='#c3c3c3')
search_indicator.place(x=10, y=150, width=5, height=35)



root.mainloop()
