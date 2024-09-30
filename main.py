import numpy as np
import customtkinter as ctk
import cv2
import time
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)

        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.aspect_ratio = self.width/self.height

        self.scale = (MAX_IMG_SCALE - MIN_IMG_SCALE)/2

        print(f'Width: {self.width}')
        print(f'Height: {self.height}')

        self.MIN_L = INIT_MIN_L
        self.MAX_L = INIT_MAX_L
        self.MIN_Z = INIT_MIN_L
        self.MAX_Z = INIT_MAX_L
        self.MIN_R = INIT_MIN_L
        self.MAX_R = INIT_MAX_L

        self.L = INIT_L
        self.Z = INIT_Z
        self.r = self.L-self.Z
        self.wavelength = DEFAULT_WAVELENGTH #Microns
        self.dxy = DEFAULT_DXY #Microns
        self.scale_factor = self.L/self.Z#

        self.fix_r = ctk.BooleanVar(self, value=False)
        self.square_field = ctk.BooleanVar(self, value=False)
        self.algorithm_var = ctk.StringVar(self, value='AS')

        self.arr_c = np.zeros((int(self.width), int(self.height)))
        self.arr_r = np.zeros((int(self.width), int(self.height)))
        self.im_c = self.arr2im(self.arr_c)
        self.im_r = self.arr2im(self.arr_r)
        self.img_c = self.create_image(self.im_c)
        self.img_r = self.create_image(self.im_r)
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)

        self.fps = 0

        self.init_viewing_frame()
        self.init_parameters_frame()

        
        ## Elements of the parameter menu

        # self.parameters_frame = ctk.CTkFrame(self, )


    def init_viewing_frame(self):
        # Create two frames, one for navigation
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=8, width=MENU_FRAME_WIDTH)
        self.navigation_frame.grid(row=0, column=0, padx=5, sticky='nsew')
        self.navigation_frame.grid_rowconfigure(5, weight=1)
        self.navigation_frame.grid_propagate(False)

        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky='nsew')
        self.viewing_frame.grid_rowconfigure(1, weight=1)

        ## Elements and layout of the navigation frame

        # Main title for the navigation frame
        self.main_title_nav = ctk.CTkLabel(self.navigation_frame, text='DLHM Reconstruction', compound='left', font=ctk.CTkFont(size=15, weight='bold'))
        self.main_title_nav.grid(row=0, column=0, padx=20, pady=40)

        # Missing commands for now
        mb_config = {'corner_radius':6, 
                                'height':MENU_BUTTONS_HEIGHT,
                                'width':MENU_FRAME_WIDTH,
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

        self.fps_label = ctk.CTkLabel(self.saving_frame, text=f'FPS: {self.fps}')
        self.fps_label.grid(row=0, column=4, padx=20, pady=20)

    def init_parameters_frame(self):
        self.parameters_frame = ctk.CTkFrame(self, corner_radius=8, width=PARAMETER_FRAME_WIDTH)
        self.parameters_frame.grid_propagate(False)

        self.main_title_param= ctk.CTkLabel(self.parameters_frame, text='Parameters')
        self.main_title_param.grid(row=0, column=0, columnspan=3, padx=20, pady=40, sticky='nsew')

        self.magnification_label = ctk.CTkLabel(self.parameters_frame, text=f'Magnificación: {round(self.scale_factor, 4)}')
        self.magnification_label.grid(row=1, column=0, pady=20, sticky='ew')

        # Frame para los parámetros de L
        self.L_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.L_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.L_frame.columnconfigure(0, weight=2)
        self.L_frame.grid_propagate(False)

        self.L_slider_title = ctk.CTkLabel(self.L_frame, text=f'Distancia entre la cámara y la fuente (L): {round(self.L, 4)}')
        self.L_slider_title.grid(row=0, column=0, columnspan=3, sticky='ew', pady=5)

        self.L_slider = ctk.CTkSlider(self.L_frame, height=SLIDER_HEIGHT, corner_radius=8, from_=self.MIN_L, to=self.MAX_L, command=self.update_L)
        self.L_slider.grid(row=1, column=0, sticky='ew')
        self.L_slider.set(round(self.L, 4))

        self.L_slider_entry = ctk.CTkEntry(self.L_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.L, 4)}')
        self.L_slider_entry.grid(row=1, column=1, sticky='ew', padx=5)
        self.L_slider_entry.setvar(value=f'{round(self.L, 4)}')

        self.L_slider_button = ctk.CTkButton(self.L_frame, width=PARAMETER_BUTTON_WIDTH, text='Set', command=self.set_value_L)
        self.L_slider_button.grid(row=1, column=2, sticky='ew', padx=10)


        # Frame para los parámetros de Z
        self.Z_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.Z_frame.grid(row=3, column=0, sticky='ew', pady=2)
        self.Z_frame.columnconfigure(0, weight=2)
        self.Z_frame.grid_propagate(False)


        self.Z_slider_title = ctk.CTkLabel(self.Z_frame, text=f'Distancia entre la muestra y la fuente (z): {round(self.Z, 4)}')
        self.Z_slider_title.grid(row=0, column=0, columnspan=3, sticky='ew', pady=5)

        self.Z_slider = ctk.CTkSlider(self.Z_frame, height=SLIDER_HEIGHT, corner_radius=8, from_=self.MIN_Z, to=self.MAX_Z, command=self.update_Z)
        self.Z_slider.grid(row=1, column=0, sticky='ew')
        self.Z_slider.set(round(self.Z, 4))

        self.Z_slider_entry = ctk.CTkEntry(self.Z_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.Z, 4)}')
        self.Z_slider_entry.grid(row=1, column=1, sticky='ew', padx=5)
        self.Z_slider_entry.setvar(value=f'{round(self.Z, 4)}')

        self.Z_slider_button = ctk.CTkButton(self.Z_frame, width=PARAMETER_BUTTON_WIDTH, text='Set', command=self.set_value_Z)
        self.Z_slider_button.grid(row=1, column=2, sticky='ew', padx=10)


        # Frame para los parámetros de r
        self.r_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.r_frame.grid(row=4, column=0, sticky='ew', pady=2)
        self.r_frame.columnconfigure(0, weight=2)
        self.r_frame.grid_propagate(False)


        self.r_slider_title = ctk.CTkLabel(self.r_frame, text=f'Distancia de reconstrucción (r): {round(self.r, 4)}')
        self.r_slider_title.grid(row=0, column=0, columnspan=3, sticky='ew', pady=5)

        self.r_slider = ctk.CTkSlider(self.r_frame, height=SLIDER_HEIGHT, corner_radius=8, from_=self.MIN_R, to=self.MAX_R, command=self.update_r)
        self.r_slider.grid(row=1, column=0, sticky='ew')
        self.r_slider.set(round(self.r, 4))

        self.r_slider_entry = ctk.CTkEntry(self.r_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.r, 4)}')
        self.r_slider_entry.grid(row=1, column=1, sticky='ew', padx=5)
        self.r_slider_entry.setvar(value=f'{round(self.r, 4)}')

        self.r_slider_button = ctk.CTkButton(self.r_frame, width=PARAMETER_BUTTON_WIDTH, text='Set', command=self.set_value_r)
        self.r_slider_button.grid(row=1, column=2, sticky='ew', padx=10)

        self.adit_options_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.adit_options_frame.grid(row=5, column=0, sticky='ew', pady=2)

        self.adit_options_frame.rowconfigure(0, weight=1)
        self.adit_options_frame.rowconfigure(1, weight=0)
        self.adit_options_frame.rowconfigure(2, weight=1)

        self.adit_options_frame.columnconfigure(0, weight=1)
        self.adit_options_frame.columnconfigure(1, weight=0)
        self.adit_options_frame.columnconfigure(2, weight=0)
        self.adit_options_frame.columnconfigure(3, weight=1)

        self.adit_options_frame.grid_propagate(False)

        self.fix_r_checkbox = ctk.CTkCheckBox(self.adit_options_frame, text='Fix reconstruction distance', variable=self.fix_r)
        self.fix_r_checkbox.grid(row=1, column=1, sticky='ew', padx=10, pady=5)

        self.square_field_checkbox = ctk.CTkCheckBox(self.adit_options_frame, text='Show Intensity', variable=self.square_field)
        self.square_field_checkbox.grid(row=1, column=2, sticky='ew', padx=10, pady=5)


        self.algorithm_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.algorithm_frame.grid(row=6, column=0, sticky='ew', pady=2)

        self.algorithm_frame.columnconfigure(0, weight=1)
        self.algorithm_frame.columnconfigure(1, weight=0)
        self.algorithm_frame.columnconfigure(2, weight=0)
        self.algorithm_frame.columnconfigure(3, weight=1)

        self.algorithm_frame.grid_propagate(False)

        self.algorithm_title = ctk.CTkLabel(self.algorithm_frame, text='Algoritmo de reconstrucción:')
        self.algorithm_title.grid(row=0, column=1, columnspan=2, sticky='ew', pady=5)

        self.as_algorithm_radio = ctk.CTkRadioButton(self.algorithm_frame, text='Angular Spectrum', variable=self.algorithm_var, value='AS')
        self.as_algorithm_radio.grid(row=1, column=1, sticky='ew', padx=10, pady=5)

        self.kr_algorithm_radio = ctk.CTkRadioButton(self.algorithm_frame, text='Kreuzer Method', variable=self.algorithm_var, value='KR')
        self.kr_algorithm_radio.grid(row=1, column=2, sticky='ew', padx=10, pady=5)

        self.limits_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT+LIMITS_FRAME_EXTRA_SPACE)
        self.limits_frame.grid(row=7, column=0, sticky='ew', pady=2)

        self.limits_frame.columnconfigure(0, weight=1)
        self.limits_frame.columnconfigure(1, weight=0)
        self.limits_frame.columnconfigure(2, weight=0)
        self.limits_frame.columnconfigure(3, weight=0)
        self.limits_frame.columnconfigure(4, weight=1)
        self.limits_frame.rowconfigure(0, weight=1)
        self.limits_frame.rowconfigure(1, weight=0)
        self.limits_frame.rowconfigure(2, weight=0)
        self.limits_frame.rowconfigure(3, weight=1)

        self.limit_min_label = ctk.CTkLabel(self.limits_frame, text='Mínimo')
        self.limit_min_label.grid(row=1, column=0, sticky='ew', padx=5)

        self.limit_max_label = ctk.CTkLabel(self.limits_frame, text='Máximo')
        self.limit_max_label.grid(row=2, column=0, sticky='ew', padx=5)

        self.limit_L_label = ctk.CTkLabel(self.limits_frame, text='L')
        self.limit_L_label.grid(row=0, column=1, sticky='ew', padx=5)

        self.limit_Z_label = ctk.CTkLabel(self.limits_frame, text='Z')
        self.limit_Z_label.grid(row=0, column=2, sticky='ew', padx=5)

        self.limit_R_label = ctk.CTkLabel(self.limits_frame, text='r')
        self.limit_R_label.grid(row=0, column=3, sticky='ew', padx=5)

        self.limit_min_L_entry = ctk.CTkEntry(self.limits_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.MIN_L, 4)}')
        self.limit_min_L_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        self.limit_max_L_entry = ctk.CTkEntry(self.limits_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.MAX_L, 4)}')
        self.limit_max_L_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=2)

        self.limit_min_Z_entry = ctk.CTkEntry(self.limits_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.MIN_Z, 4)}')
        self.limit_min_Z_entry.grid(row=1, column=2, sticky='ew', padx=5, pady=2)
        self.limit_max_Z_entry = ctk.CTkEntry(self.limits_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.MAX_Z, 4)}')
        self.limit_max_Z_entry.grid(row=2, column=2, sticky='ew', padx=5, pady=2)

        self.limit_min_r_entry = ctk.CTkEntry(self.limits_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.MIN_Z, 4)}')
        self.limit_min_r_entry.grid(row=1, column=3, sticky='ew', padx=5, pady=2)
        self.limit_max_r_entry = ctk.CTkEntry(self.limits_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{round(self.MAX_Z, 4)}')
        self.limit_max_r_entry.grid(row=2, column=3, sticky='ew', padx=5, pady=2)


        self.set_limits_button = ctk.CTkButton(self.limits_frame, width=PARAMETER_BUTTON_WIDTH, text='Set all', command=self.set_limits)
        self.set_limits_button.grid(row=1, column=4, sticky='ew', padx=10)

        self.restore_limits_button = ctk.CTkButton(self.limits_frame, width=PARAMETER_BUTTON_WIDTH, text='Restore all', command=self.restore_limits)
        self.restore_limits_button.grid(row=2, column=4, sticky='ew', padx=10)

        
        self.parameters_frame.rowconfigure(8, weight=1)
        
        self.home_button = ctk.CTkButton(self.parameters_frame, text='Home', command=lambda: self.change_menu_to('home'))
        self.home_button.grid(row=8, column=0, pady=20, sticky='s')


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

        if self.fix_r.get():
            self.L = self.Z+self.r
        else:
            self.Z = self.L-self.r

        self.update_parameters()

    def set_value_L(self):
        try:
            val = float(self.L_slider_entry.get())
        except:
            val = self.L

        if val<=self.MIN_L:
            val = self.MIN_L
        elif val >= self.MAX_L:
            val = self.MAX_L

        self.update_L(val)

    def set_value_Z(self):
        try:
            val = float(self.Z_slider_entry.get())
        except:
            val = self.Z

        if val<=self.MIN_Z:
            val = self.MIN_Z
        elif val >= self.MAX_Z:
            val = self.MAX_Z
            
        self.update_Z(val)

    def set_value_r(self):
        try:
            val = float(self.r_slider_entry.get())
        except:
            val = self.r

        if val<=self.MIN_L:
            val = self.MIN_L
        elif val >= self.MAX_L:
            val = self.MAX_L
        
            
        self.update_r(val)

    def set_limits(self):
        try:
            self.MIN_L = float(self.limit_min_L_entry.get())
        except:
            print(f'self.MIN_L received invalid value.')
        try:
            self.MAX_L = float(self.limit_max_L_entry.get())
        except:
            print(f'self.MAX_L received invalid value.')
        try:
            self.MIN_Z = float(self.limit_min_Z_entry.get())
        except:
            print(f'self.MIN_Z received invalid value.')
        try:
            self.MAX_Z = float(self.limit_max_Z_entry.get())
        except:
            print(f'self.MAX_Z received invalid value.')
        try:
            self.MIN_R = float(self.limit_min_r_entry.get())
        except:
            print(f'self.MIN_R received invalid value.')
        try:
            self.MAX_R = float(self.limit_max_r_entry.get())
        except:
            print(f'self.MAX_R received invalid value.')
        
        self.L_slider.configure(from_=self.MIN_L, to=self.MAX_L)
        self.Z_slider.configure(from_=self.MIN_Z, to=self.MAX_Z)
        self.r_slider.configure(from_=self.MIN_R, to=self.MAX_R)

    def restore_limits(self):
        self.MIN_L = INIT_MIN_L
        self.MAX_L = INIT_MAX_L
        self.MIN_Z = INIT_MIN_L
        self.MAX_Z = INIT_MAX_L
        self.MIN_R = INIT_MIN_L
        self.MAX_R = INIT_MAX_L

        self.L_slider.configure(from_=self.MIN_L, to=self.MAX_L)
        self.Z_slider.configure(from_=self.MIN_Z, to=self.MAX_Z)
        self.r_slider.configure(from_=self.MIN_R, to=self.MAX_R)


    def change_menu_to(self, name:str):
        if name=='home':
            self.navigation_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.navigation_frame.grid_forget()
        
        if name=='parameters':
            self.parameters_frame.grid(row=0, column=0, sticky='nsew', padx=5)
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
        start_time = time.time()

        # This function will read the image, process it and pass the resulting arrays to the corresponding widgets
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

        end_time = time.time()

        elapsed_time = end_time-start_time

        self.fps = round(1/elapsed_time, 1)
        self.fps_label.configure(text=f'FPS: {self.fps}')


        self.after(20, self.streaming)

    def reconstruct(self, img):
        field = np.sqrt(img)

        if self.algorithm_var.get() == 'AS':
            recon = propagate(field, self.r, self.wavelength, self.dxy, self.dxy, self.scale_factor)
        elif self.algorithm_var.get() == 'KR':
            recon = field

        if self.square_field.get():
            return np.abs(recon)**2
        else:
            return np.abs(recon)




    def run(self):
        pass

if __name__=='__main__':
    app = App()
    app.streaming()
    app.mainloop()