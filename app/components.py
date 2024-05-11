import tkinter as tk



class InputComponent(tk.Frame):
    def __init__(self, master=None, label='Input', validate_function=None, **kwargs):
        super().__init__(master, **kwargs)
        self.var = tk.DoubleVar()
        self.var.set(0.5)
        self.label = tk.Label(self, text=label, font=("bold", 15))

        if validate_function is None:
            validate_function = lambda: True
        self.valiadate_function = self.register(validate_function)
        self.entry = tk.Entry(self, font=("bold", 15), width=20, textvariable=self.var,
                              validatecommand=validate_function, validate='focus')

        self.label.pack(side=tk.TOP, pady=(0,5), anchor=tk.W)
        self.entry.pack(side=tk.TOP, pady=(0,5), anchor=tk.W)

    def get_value(self):
        return self.var.get()

