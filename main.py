import cv2
import numpy as np
from tkinter import Tk, Canvas, NW, filedialog, StringVar, OptionMenu, Scale, Label, HORIZONTAL, IntVar, Button, Radiobutton
from PIL import Image, ImageTk
from filters import bilateral, median, gaussian, sobel, low_pass_filter, high_pass_filter


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.canvas = Canvas(root, cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.points = []
        self.image = None
        self.tk_image = None
        self.cv_image = None

        self.filter_type = StringVar(root)
        self.filter_type.set("bilateral")  # default value
        self.filter_type.trace("w", self.update_filter_parameters)  # Track changes

        self.mode = StringVar(root)
        self.mode.set("part")  # default to part mode

        self.control_frame = Canvas(root)
        self.control_frame.pack(side="right", fill="y")

        # Filter options
        self.filter_menu = OptionMenu(self.control_frame, self.filter_type, "bilateral", "gaussian", "median", "sobel", "lowpass", "highpass")
        self.filter_menu.pack()

        # Mode options
        Radiobutton(self.control_frame, text="Part", variable=self.mode, value="part").pack(anchor="w")
        Radiobutton(self.control_frame, text="Whole", variable=self.mode, value="whole").pack(anchor="w")

        # Filter parameters frame
        self.parameters_frame = Canvas(self.control_frame)
        self.parameters_frame.pack()

        # Apply button
        self.apply_button = Button(self.control_frame, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack()

        # Save button
        self.save_button = Button(self.control_frame, text="Save Image", command=self.save_image)
        self.save_button.pack()

        self.load_image()

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Initialize the parameters display
        self.update_filter_parameters()
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        self.image = Image.open(file_path)
        self.cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)

        # Resize image to fit window considering the control frame width
        max_width = self.root.winfo_screenwidth() - self.control_frame.winfo_reqwidth()
        max_height = self.root.winfo_screenheight()
        self.image.thumbnail((max_width, max_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.root.geometry(f"{self.image.width}x{self.image.height}")

    def on_button_press(self, event):
        if self.mode.get() == "part":
            self.points = [(event.x, event.y)]

    def on_mouse_drag(self, event):
        if self.mode.get() == "part":
            self.points.append((event.x, event.y))
            self.canvas.create_line(self.points[-2], self.points[-1], fill='red')

    def on_button_release(self, event):
        if self.mode.get() == "part":
            self.points.append((event.x, event.y))

    def apply_filter(self):
        selected_filter = self.filter_type.get()
        if self.mode.get() == "whole":
            if selected_filter == "bilateral":
                self.cv_image = bilateral(self.cv_image, d=self.bilateral_d.get(),
                                          sigmaColor=self.bilateral_sigmaColor.get(),
                                          sigmaSpace=self.bilateral_sigmaSpace.get(),
                                          count=self.bilateral_count.get())
            elif selected_filter == "gaussian":
                self.cv_image = gaussian(self.cv_image, ksize=self.gaussian_ksize.get(), sigmaX=self.gaussian_sigmaX.get())
            elif selected_filter == "median":
                self.cv_image = median(self.cv_image, ksize=self.median_ksize.get())
            elif selected_filter == "sobel":
                self.cv_image = sobel(self.cv_image, ksize=self.sobel_ksize.get())
            elif selected_filter == "lowpass":
                self.cv_image = low_pass_filter(self.cv_image, cutoff_frequency=self.lowpass_cutoff.get())
            elif selected_filter == "highpass":
                self.cv_image = high_pass_filter(self.cv_image, cutoff_frequency=self.highpass_cutoff.get())
        elif self.mode.get() == "part" and self.points:
            # Create a mask with the selected region
            mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
            scaled_points = [(int(x * self.cv_image.shape[1] / self.image.width),
                              int(y * self.cv_image.shape[0] / self.image.height)) for x, y in self.points]
            cv2.fillPoly(mask, [np.array(scaled_points)], 255)

            # Apply the selected filter only to the selected region
            filtered_image = self.cv_image.copy()
            
            if selected_filter == "bilateral":
                filtered_image = bilateral(filtered_image, d=self.bilateral_d.get(),
                                            sigmaColor=self.bilateral_sigmaColor.get(),
                                            sigmaSpace=self.bilateral_sigmaSpace.get(),
                                            count=self.bilateral_count.get())
            elif selected_filter == "gaussian":
                filtered_image = gaussian(filtered_image, ksize=self.gaussian_ksize.get(), sigmaX=self.gaussian_sigmaX.get())
            elif selected_filter == "median":
                filtered_image = median(filtered_image, ksize=self.median_ksize.get())
            elif selected_filter == "sobel":
                filtered_image = sobel(filtered_image, ksize=self.sobel_ksize.get())
            elif selected_filter == "lowpass":
                filtered_image = low_pass_filter(filtered_image, cutoff_frequency=self.lowpass_cutoff.get())
            elif selected_filter == "highpass":
                filtered_image = high_pass_filter(filtered_image, cutoff_frequency=self.highpass_cutoff.get())
            
            self.cv_image[mask == 255] = filtered_image[mask == 255]

        # Update the displayed image
        self.image = Image.fromarray(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))
        self.image.thumbnail((self.root.winfo_screenwidth() - self.control_frame.winfo_reqwidth(), self.root.winfo_screenheight()), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            save_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_path, save_image)
            print(f"Image saved as {file_path}")

    def update_filter_parameters(self, *args):
        # Clear the existing parameters
        for widget in self.parameters_frame.winfo_children():
            widget.destroy()

        selected_filter = self.filter_type.get()

        if selected_filter == "bilateral":
            self.bilateral_d = IntVar(value=9)
            self.bilateral_sigmaColor = IntVar(value=75)
            self.bilateral_sigmaSpace = IntVar(value=75)
            self.bilateral_count = IntVar(value=10)

            Label(self.parameters_frame, text="Diameter:").pack()
            Scale(self.parameters_frame, from_=1, to_=50, orient=HORIZONTAL, variable=self.bilateral_d).pack()
            Label(self.parameters_frame, text="Sigma Color:").pack()
            Scale(self.parameters_frame, from_=1, to_=150, orient=HORIZONTAL, variable=self.bilateral_sigmaColor).pack()
            Label(self.parameters_frame, text="Sigma Space:").pack()
            Scale(self.parameters_frame, from_=1, to_=150, orient=HORIZONTAL, variable=self.bilateral_sigmaSpace).pack()
            Label(self.parameters_frame, text="Iterations:").pack()
            Scale(self.parameters_frame, from_=1, to_=20, orient=HORIZONTAL, variable=self.bilateral_count).pack()
        
        elif selected_filter == "gaussian":
            self.gaussian_ksize = IntVar(value=5)
            self.gaussian_sigmaX = IntVar(value=0)

            Label(self.parameters_frame, text="Kernel Size:").pack()
            Scale(self.parameters_frame, from_=1, to_=50, orient=HORIZONTAL, variable=self.gaussian_ksize).pack()
            Label(self.parameters_frame, text="Sigma X:").pack()
            Scale(self.parameters_frame, from_=0, to_=50, orient=HORIZONTAL, variable=self.gaussian_sigmaX).pack()

        elif selected_filter == "median":
            self.median_ksize = IntVar(value=5)

            Label(self.parameters_frame, text="Kernel Size:").pack()
            Scale(self.parameters_frame, from_=1, to_=50, orient=HORIZONTAL, variable=self.median_ksize).pack()

        elif selected_filter == "sobel":
            self.sobel_ksize = IntVar(value=3)

            Label(self.parameters_frame, text="Kernel Size:").pack()
            Scale(self.parameters_frame, from_=1, to_=31, orient=HORIZONTAL, variable=self.sobel_ksize).pack()
        
        elif selected_filter == "lowpass":
            self.lowpass_cutoff = IntVar(value=100)

            Label(self.parameters_frame, text="Cutoff Frequency:").pack()
            Scale(self.parameters_frame, from_=1, to_=500, orient=HORIZONTAL, variable=self.lowpass_cutoff).pack()

        elif selected_filter == "highpass":
            self.highpass_cutoff = IntVar(value=100)

            Label(self.parameters_frame, text="Cutoff Frequency:").pack()
            Scale(self.parameters_frame, from_=1, to_=500, orient=HORIZONTAL, variable=self.highpass_cutoff).pack()

if __name__ == "__main__":
    root = Tk()
    app = ImageEditor(root)
    root.mainloop()
