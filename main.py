import numpy as np
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from settings import *
from _3DHR_Utilities import *


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
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Verificar si la cámara se abrió correctamente
        if not self.cap.isOpened():
            print("No se puede abrir la cámara")
            exit()

        # Asegura que la cámara se inicie en la máxima resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_WIDTH)

        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.aspect_ratio = self.width/self.height

        self.scale = (MAX_IMG_SCALE - MIN_IMG_SCALE)/2

        print(f'Width: {self.width}')
        print(f'Height: {self.height}')

        self.L = INIT_L
        self.Z = INIT_Z
        self.r = self.L-self.Z
        self.wavelength = DEFAULT_WAVELENGTH #Microns
        self.dxy = DEFAULT_DXY #Microns
        self.scale_factor = self.L/self.Z#

        self.fix_r = ctk.BooleanVar(self, value=False)

        self.arr_c = np.zeros((int(self.width), int(self.height)))
        self.arr_r = np.zeros((int(self.width), int(self.height)))
        self.im_c = self.arr2im(self.arr_c)
        self.im_r = self.arr2im(self.arr_r)
        self.img_c = self.create_image(self.im_c)
        self.img_r = self.create_image(self.im_r)
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)

        self.init_viewing_frame()
        self.init_parameters_frame()

        
        ## Elements of the parameter menu

        # self.parameters_frame = ctk.CTkFrame(self, )


    def init_viewing_frame(self):
        # Create two frames, one for navigation
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=8)
        self.navigation_frame.grid(row=0, column=0, sticky='nsew')
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky='nsew')
        self.viewing_frame.grid_rowconfigure(1, weight=1)

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

        self.param_button = ctk.CTkButton(self.navigation_frame, text='Parameters', **mb_config, command=lambda: self.change_menu_to('parameters'))
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
        self.size_slider.set(self.scale)

        self.save_captured_button = ctk.CTkButton(self.saving_frame, text='Save Capture', command=self.save_capture)
        self.save_captured_button.grid(row=0, column=2, padx=20, pady=20)

        self.save_processed_button = ctk.CTkButton(self.saving_frame, text='Save Reconstruction', command=self.save_processed)
        self.save_processed_button.grid(row=0, column=3, padx=20, pady=20)

    def init_parameters_frame(self):
        self.parameters_frame = ctk.CTkFrame(self, corner_radius=8, width=PARAMETER_FRAME_WIDTH)
        self.parameters_frame.rowconfigure(8, weight=1)
        self.parameters_frame.grid_propagate(False)

        self.main_title_param= ctk.CTkLabel(self.parameters_frame, text='Parameters')
        self.main_title_param.grid(row=0, column=0, columnspan=3, padx=20, pady=40, sticky='nsew')

        self.L_slider_title = ctk.CTkLabel(self.parameters_frame, text=f'Distancia entre la cámara y la fuente (L): {round(self.L, 4)}')
        self.L_slider_title.grid(row=1, column=0, columnspan=3, padx=20, pady=(20, 5), sticky='nsew')

        self.L_slider = ctk.CTkSlider(self.parameters_frame, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, corner_radius=8, from_=MIN_L, to=MAX_L, command=self.update_L)
        self.L_slider.grid(row=2, column=0, padx=20, pady=25, sticky='nsew')
        self.L_slider.set(round(self.L, 4))

        self.L_slider_entry = ctk.CTkEntry(self.parameters_frame, placeholder_text=f'{round(self.L, 4)}', width=50)
        self.L_slider_entry.grid(row=2, column=1, padx=20, pady=20, sticky='nsew')

        self.L_slider_entry.setvar(value=f'{round(self.L, 4)}')
        
        self.L_slider_button = ctk.CTkButton(self.parameters_frame, text='Set', command=self.set_value_L, width=50)
        self.L_slider_button.grid(row=2, column=2, padx=5, pady=20, sticky='nsew')



        self.Z_slider_title = ctk.CTkLabel(self.parameters_frame, text=f'Distancia entre la muestra y la fuente (z): {round(self.Z, 4)}', width=50)
        self.Z_slider_title.grid(row=3, column=0, columnspan=3, padx=20, pady=(20, 5), sticky='nsew')

        self.Z_slider = ctk.CTkSlider(self.parameters_frame, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, corner_radius=8, from_=MIN_Z, to=MAX_Z, command=self.update_Z)
        self.Z_slider.grid(row=4, column=0, padx=20, pady=25, sticky='nsew')
        self.Z_slider.set(round(self.Z, 4))

        self.Z_slider_entry = ctk.CTkEntry(self.parameters_frame, placeholder_text=f'{round(self.Z, 4)}', width=50)
        self.Z_slider_entry.grid(row=4, column=1, padx=20, pady=20, sticky='nsew')
        self.Z_slider_entry.setvar(value=f'{round(self.Z, 4)}')

        self.Z_slider_button = ctk.CTkButton(self.parameters_frame, text='Set', command=self.set_value_Z, width=50)
        self.Z_slider_button.grid(row=4, column=2, padx=5, pady=20, sticky='nsew')

        self.r_slider_title = ctk.CTkLabel(self.parameters_frame, text=f'Distancia de reconstrucción (r): {round(self.r, 4)}', width=50)
        self.r_slider_title.grid(row=5, column=0, columnspan=3, padx=20, pady=(20, 5), sticky='nsew')

        self.r_slider = ctk.CTkSlider(self.parameters_frame, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, corner_radius=8, from_=MIN_L, to=MAX_L, command=self.update_r)
        self.r_slider.grid(row=6, column=0, padx=20, pady=25, sticky='nsew')
        self.r_slider.set(round(self.r, 4))

        self.r_slider_entry = ctk.CTkEntry(self.parameters_frame, placeholder_text=f'{round(self.r, 4)}', width=50)
        self.r_slider_entry.grid(row=6, column=1, padx=20, pady=20, sticky='nsew')
        self.r_slider_entry.setvar(value=f'{round(self.r, 4)}')

        self.r_slider_button = ctk.CTkButton(self.parameters_frame, text='Set', command=self.set_value_r, width=50)
        self.r_slider_button.grid(row=6, column=2, padx=5, pady=20, sticky='nsew')

        self.magnification_label = ctk.CTkLabel(self.parameters_frame, text=f'Magnificación: {round(self.scale_factor, 4)}')
        self.magnification_label.grid(row=7, column=0, padx=10, pady=(20, 5), sticky='nsew')

        self.fix_r_checkbox = ctk.CTkCheckBox(self.parameters_frame, text='Fix reconstruction distance', variable=self.fix_r, width=50)
        self.fix_r_checkbox.grid(row=8, column=0, padx=10, pady=20, sticky='nsew')

                # Theme selection menu
        
        self.home_button = ctk.CTkButton(self.parameters_frame, text='Home', width=50, command=lambda: self.change_menu_to('home'))
        self.home_button.grid(row=9, column=0, padx=20, pady=20, sticky='s')

        self.appearance_mode_menu = ctk.CTkOptionMenu(self.parameters_frame, values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=9, column=1, padx=20, pady=20, sticky="s")

    def update_parameters(self):
        self.Z_slider_title.configure(text=f'Distancia entre la muestra y la fuente (Z): {round(self.Z, 4)}')
        self.Z_slider.set(round(self.Z, 4))
        self.Z_slider_entry.configure(placeholder_text=f'{round(self.Z, 4)}')
        self.L_slider_title.configure(text=f'Distancia entre la cámara y la fuente (L): {round(self.L, 4)}')
        self.L_slider.set(round(self.L, 4))
        self.L_slider_entry.configure(placeholder_text=f'{round(self.L, 4)}')
        self.r_slider_title.configure(text=f'Distancia de reconstrucción (r): {round(self.r, 4)}')
        self.r_slider.set(round(self.r, 4))
        self.r_slider_entry.configure(placeholder_text=f'{round(self.r, 4)}')
        self.magnification_label.configure(text=f'Magnificación: {round(self.scale_factor, 4)}')

        self.scale_factor = self.L/self.Z


    def update_L(self, val):
        self.L = val

        if self.fix_r.get():
            self.Z = self.L-self.r
        else:
            if self.L<=self.Z:
                self.Z = self.L

            self.r = self.L-self.Z

        self.update_parameters()


    def update_Z(self, val):
        self.Z = val

        if self.fix_r.get():
            self.L = self.Z+self.r
        else:

            self.r = self.L-self.Z

            if self.Z >= self.L:
                self.L = self.Z
        
        self.update_parameters()


    def update_r(self, val):
        self.r = val

        self.update_parameters()

    def set_value_L(self):
        try:
            val = int(self.L_slider_entry.get())
        except:
            val = self.L

        if val<=MIN_L:
            val = MIN_L
        elif val >= MAX_L:
            val = MAX_L

        self.update_L(val)

    def set_value_Z(self):
        try:
            val = int(self.Z_slider_entry.get())
        except:
            val = self.Z

        if val<=MIN_Z:
            val = MIN_Z
        elif val >= MAX_Z:
            val = MAX_Z
            
        self.update_Z(val)

    def set_value_r(self):
        try:
            val = int(self.r_slider_entry.get())
        except:
            val = self.r

        if val<=MIN_L:
            val = MIN_L
        elif val >= MAX_L:
            val = MAX_L
        
            
        self.update_r(val)


    def change_menu_to(self, name:str):
        if name=='home':
            self.navigation_frame.grid(row=0, column=0, sticky='nsew')
            self.navigation_frame.grid_rowconfigure(5, weight=1)
        else:
            self.navigation_frame.grid_forget()
        
        if name=='parameters':
            self.parameters_frame.grid(row=0, column=0, sticky='nsew')
            self.parameters_frame.grid_rowconfigure(5, weight=1)
        else:
            self.parameters_frame.grid_forget()


    def update_im_size(self, size):
        self.scale = size

    def save_capture(self, ext:str='bmp'):
        self.im_c.save(f'saves/capture/capture{self.current_capture_c}.{ext}')
        self.current_capture_c += 1

    def save_processed(self, ext:str='bmp'):
        self.im_r.save(f'saves/reconstruction/reconstruction{self.current_capture_c}.{ext}')
        self.current_capture_c += 1



    def im2arr(self, path: str):
        '''Converts file image into numpy array.'''
        return np.asarray(Image.open(path).convert('L'))

    def arr2im(self, array: np.ndarray):
        '''Converts numpy array into PhotoImage type'''
        return Image.fromarray(array, 'L')
    
    def create_image(self, img: Image.Image, size: list = (400, 300)):
        return ctk.CTkImage(light_image=img, dark_image=img, size=tuple(size))

    def change_appearance_mode_event(self, new_appearance_mode):
        '''Changes between light and dark mode.'''
        ctk.set_appearance_mode(new_appearance_mode)

    def streaming(self):
        self.arr_c= cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
        self.arr_c = cv2.flip(self.arr_c, 1)
        self.im_c = self.arr2im(self.arr_c)
        self.img_c = self.create_image(self.im_c)
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.captured_label.img = self.img_c
        self.captured_label.configure(image=self.img_c)

        self.arr_r = self.reconstruct(self.arr_c)
        self.arr_r = np.uint8(normalize(self.arr_r, 255))
        self.im_r = self.arr2im(self.arr_r)
        self.img_r = self.create_image(self.im_r)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)
        self.processed_label.img = self.img_r
        self.processed_label.configure(image=self.img_r)


        self.after(20, self.streaming)

    def reconstruct(self, img):
        field = np.sqrt(img)
        recon = propagate(field, self.r, self.wavelength, self.dxy, self.dxy, self.scale_factor)

        return np.abs(recon)




    def run(self):
        pass

if __name__=='__main__':
    app = App()
    app.streaming()
    app.mainloop()