import numpy as np
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from settings import *


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('DLHM GUI')
        self.geometry('1366x768')
        self.after(0, lambda:self.state('zoomed'))

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # This variable keeps track of the captured image from the camera or the reconstruction
        # in order to keep sequential images
        self.current_capture_c=0
        self.current_capture_r=0

        # Inicializar la cámara (0 es generalmente la cámara por defecto)
        self.cap = cv2.VideoCapture(0)

        # Verificar si la cámara se abrió correctamente
        if not self.cap.isOpened():
            print("No se puede abrir la cámara")
            exit()

        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(self.width)
        self.aspect_ratio = self.width/self.height

        self.scale = MAX_IMG_SCALE

        self.arr_c = np.zeros((int(self.width), int(self.height)))
        self.arr_r = np.zeros((int(self.width), int(self.height)))
        self.im_c = self.arr2im(self.arr_c)
        self.im_r = self.arr2im(self.arr_r)
        self.img_c = self.create_image(self.im_c)
        self.img_r = self.create_image(self.im_r)
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)

        # Create two frames, one for navigation
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=8)
        self.navigation_frame.grid(row=0, column=0, sticky='nsew')
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky='nsew')
        self.viewing_frame.grid_columnconfigure(0, weight=1)
        self.viewing_frame.grid_columnconfigure(1, weight=1)
        self.viewing_frame.grid_rowconfigure(2, weight=1)

        ## Elements and layout of the navigation frame

        # Main title for the navigation frame
        self.main_title_nav = ctk.CTkLabel(self.navigation_frame, text='DLHM Reconstruction', compound='left', font=ctk.CTkFont(size=15, weight='bold'))
        self.main_title_nav.grid(row=0, column=0, padx=20, pady=40)

        # Missing commands for now
        mb_config = {'corner_radius':6, 
                                'height':40,
                                'border_spacing':10, 
                                'fg_color':("gray75", "gray25"), 
                                'text_color':("gray10", "gray90"), 
                                'hover_color':("gray80", "gray20"),
                                'anchor':"c"}
        
        mb_grid_config = {'sticky':'ew', 'padx':1, 'pady':3}

        text_config = {'compound':'left', 'font':ctk.CTkFont(size=15, weight='bold')}

        self.param_button = ctk.CTkButton(self.navigation_frame, text='Parameters', **mb_config)
        self.param_button.grid(row=1, column=0, **mb_grid_config)

        self.filters_button = ctk.CTkButton(self.navigation_frame, text='Filters', **mb_config)
        self.filters_button.grid(row=2, column=0, **mb_grid_config)

        self.it_button = ctk.CTkButton(self.navigation_frame, text='Image Tools', **mb_config)
        self.it_button.grid(row=3, column=0, **mb_grid_config)

        self.so_button = ctk.CTkButton(self.navigation_frame, text='Saving Options', **mb_config)
        self.so_button.grid(row=4, column=0, **mb_grid_config)


        # Theme selection menu
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=5, column=0, padx=20, pady=20, sticky="s")


        ## Elements and layout of the viewing frame

        # Main title for the viewing frame
        self.main_title_view = ctk.CTkLabel(self.viewing_frame, text='DLHM Viewing Window', compound='left', font=ctk.CTkFont(size=15, weight='bold'), anchor=ctk.CENTER)
        self.main_title_view.grid(row=0, column=0, padx=20, pady=40, columnspan=2, sticky='nsew')

        # An image frame containing the captured image and the processed image
        self.image_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.image_frame.grid(row=1, column=0, padx=20, pady=15, sticky='ne')

        self.captured_title_label = ctk.CTkLabel(self.image_frame, text='Captured Image', **text_config)
        self.captured_title_label.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
        self.captured_label = ctk.CTkLabel(self.image_frame, image=self.img_c, text='')
        self.captured_label.grid(row=1, column=0, padx=20, pady=20, sticky='nsew')

        self.captured_title_label = ctk.CTkLabel(self.image_frame, text='Processed Image', **text_config)
        self.captured_title_label.grid(row=0, column=1, padx=20, pady=20, sticky='nsew')
        self.processed_label = ctk.CTkLabel(self.image_frame, image=self.img_r, text='')
        self.processed_label.grid(row=1, column=1, padx=20, pady=20, sticky='nsew')
        #####
        
        # Buttons for saving and changing image scale

        self.saving_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.saving_frame.grid(row=2, column=0, padx=20, pady=20, sticky='ws')

        self.size_label = ctk.CTkLabel(self.saving_frame, text='Viewing Size:')
        self.size_label.grid(row=0, column=0, padx=20)
        
        self.size_slider = ctk.CTkSlider(self.saving_frame, width=100, from_=MIN_IMG_SCALE, to=MAX_IMG_SCALE, command=self.update_im_size)
        self.size_slider.grid(row=0, column=1, padx=10, pady=20)
        self.size_slider.set(MAX_IMG_SCALE)

        self.save_captured_button = ctk.CTkButton(self.saving_frame, text='Save Capture', command=self.save_capture)
        self.save_captured_button.grid(row=0, column=2, padx=20, pady=20)

        self.save_processed_button = ctk.CTkButton(self.saving_frame, text='Save Reconstruction', command=self.save_processed)
        self.save_processed_button.grid(row=0, column=3, padx=20, pady=20)



    def update_im_size(self, size):
        self.scale = size
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.captured_label.configure(image=self.img_c)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)
        self.processed_label.configure(image=self.img_r)

    def save_capture(self, ext:str='bmp'):
        self.im_c.save(f'saves/capture/capture{self.current_capture_c}.{ext}')
        self.current_capture_c += 1

    def save_processed(self, ext:str='bmp'):
        self.im_r.save(f'saves/reconstruction/reconstruction{self.current_capture_c}.{ext}')
        self.current_capture_c += 1



    def im2arr(self, path: str):
        '''Converts file image into numpy array.'''
        return np.asarray(Image.open(path))

    def arr2im(self, array: np.ndarray):
        '''Converts numpy array into PhotoImage type'''
        return Image.fromarray(array)
    
    def create_image(self, img: Image.Image, size: list = (400, 300)):
        return ctk.CTkImage(light_image=img, dark_image=img, size=tuple(size))

    def change_appearance_mode_event(self, new_appearance_mode):
        '''Changes between light and dark mode.'''
        ctk.set_appearance_mode(new_appearance_mode)

    def streaming(self):
        self.arr_c= cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        self.arr_c = cv2.flip(self.arr_c, 1)
        self.im_c = self.arr2im(self.arr_c)
        self.img_c = self.create_image(self.im_c)
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.captured_label.img = self.img_c
        self.captured_label.configure(image=self.img_c)
        self.after(20, self.streaming)


    def run(self):
        pass

if __name__=='__main__':
    app = App()
    app.streaming()
    app.mainloop()