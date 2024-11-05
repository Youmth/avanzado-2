import numpy as np
import customtkinter as ctk
import time
import os
from multiprocessing import Process, Queue
from kreuzer_functions import filtcosenoF

from settings import *
from _3DHR_Utilities import *
from parallel_rc import *
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('DLHM GUI')
        self.geometry('1366x768')
        self.after(0, lambda:self.state('zoomed'))

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.width = MAX_WIDTH
        self.height = MAX_HEIGHT

        # Scale for visualization, doesn't change the original resolution, just the size
        # on screen
        self.scale = (MAX_IMG_SCALE - MIN_IMG_SCALE)/2

        # Limits for the reconstruction parameters
        self.MIN_L = INIT_MIN_L
        self.MAX_L = INIT_MAX_L
        self.MIN_Z = INIT_MIN_L
        self.MAX_Z = INIT_MAX_L
        self.MIN_R = INIT_MIN_L
        self.MAX_R = INIT_MAX_L
        
        # Physical parameters for reconstructions
        self.L = INIT_L
        self.Z = INIT_Z
        self.r = self.L-self.Z
        self.wavelength = DEFAULT_WAVELENGTH #Microns
        self.dxy = DEFAULT_DXY #Microns
        self.scale_factor = self.L/self.Z#

        # FC parameter setting
        self.cosine_period = DEFAULT_COSINE_PERIOD

        self.FC = filtcosenoF(self.cosine_period, np.array((self.width, self.height)))

        # Vars for choosing parameters in the parameters menu
        self.fix_r = ctk.BooleanVar(self, value=False)
        self.square_field = ctk.BooleanVar(self, value=False)
        self.phase_r = ctk.BooleanVar(self, value=False)
        self.algorithm_var = ctk.StringVar(self, value='AS')
        self.filter_image_var = ctk.StringVar(self, value='CA') # CA for captured by default
        
        self.file_path = ''

        self.gamma_checkbox_var = ctk.BooleanVar(self, value=False)
        self.contrast_checkbox_var = ctk.BooleanVar(self, value=False)
        self.adaptative_eq_checkbox_var = ctk.BooleanVar(self, value=False)
        self.highpass_checkbox_var = ctk.BooleanVar(self, value=False)
        self.lowpass_checkbox_var = ctk.BooleanVar(self, value=False)

        self.manual_gamma_c_var = ctk.BooleanVar(self, value=False)
        self.manual_gamma_r_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_c_var = ctk.BooleanVar(self, value=False)
        self.manual_contrast_r_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_c_var = ctk.BooleanVar(self, value=False)
        self.manual_adaptative_eq_r_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_c_var = ctk.BooleanVar(self, value=False)
        self.manual_highpass_r_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_c_var = ctk.BooleanVar(self, value=False)
        self.manual_lowpass_r_var = ctk.BooleanVar(self, value=False)

        self.filters_c = []
        self.filters_r = []

        self.gamma_c = 0
        self.gamma_r = 0
        self.contrast_c = 0
        self.contrast_r = 0
        self.adaptative_eq_c = False
        self.adaptative_eq_r = False
        self.highpass_c = 0
        self.highpass_r = 0
        self.lowpass_c = 0
        self.lowpass_r = 0

        # Arrays and images for the captured and reconstructed matrices
        self.arr_c = np.zeros((int(self.width), int(self.height)))
        self.arr_r = np.zeros((int(self.width), int(self.height)))

        self.arr_c_f = np.zeros((int(self.width), int(self.height)))

        self.im_c = arr2im(self.arr_c)
        self.im_r = arr2im(self.arr_r)
        self.img_c = create_image(self.im_c, self.width, self.height)
        self.img_r = create_image(self.im_r, self.width, self.height)

        # This is the visualization size, not the actual size of the processed image
        self.img_c._size = (self.width*self.scale, self.height*self.scale)
        self.img_r._size = (self.width*self.scale, self.height*self.scale)

        self.w_fps = 0
        self.c_fps = 0
        self.r_fps = 0

        self.settings = False

        self.capture_parameters = [self.arr_c, 
                                   self.arr_c_f,
                                   self.filters_c,
                                    self.file_path,
                                    self.width,
                                    self.height,
                                    self.settings]

        self.recon_parameters = [self.arr_c,
                                 self.arr_r,
                                 self.filters_r,
                                 self.algorithm_var.get(),
                                 self.L,
                                 self.Z,
                                 self.r,
                                 self.wavelength,
                                 self.dxy,
                                 self.scale_factor,
                                 self.FC,
                                 self.square_field.get(),
                                 self.phase_r.get()]

        self.captured_q = Queue()
        self.filtered_q = Queue()
        self.recon_q = Queue()
        self.filters_c_q = Queue()
        self.filters_r_q = Queue()
        self.algorithm_q = Queue()
        self.L_q = Queue()
        self.Z_q = Queue()
        self.r_q = Queue()
        self.wavelength_q = Queue()
        self.dxy_q = Queue()
        self.scale_factor_q = Queue()
        self.FC_q = Queue()
        self.squared_q = Queue()
        self.phase_q = Queue()
        self.file_path_q = Queue()
        self.width_q = Queue()
        self.height_q = Queue()

        self.settings_q = Queue()

        self.capture_queues = (self.captured_q,
                               self.filtered_q,
                               self.filters_c_q,
                               self.file_path_q,
                               self.width_q,
                               self.height_q,
                               self.settings_q)

        self.recon_queues =   (self.captured_q,
                    self.recon_q,
                    self.filters_r_q,
                    self.algorithm_q,
                    self.L_q,
                    self.Z_q,
                    self.r_q,
                    self.wavelength_q,
                    self.dxy_q,
                    self.scale_factor_q,
                    self.FC_q,
                    self.squared_q,
                    self.phase_q)
        
        for queue, parameter in zip(self.capture_queues, self.capture_parameters):
            if queue==self.captured_q:
                queue.put((self.arr_c, self.c_fps))
                continue
            if queue==self.filtered_q:
                continue
            if queue==self.filters_c_q:
                continue

            queue.put(parameter)

        for queue, parameter in zip(self.recon_queues, self.recon_parameters):
            if queue==self.captured_q:
                queue.put((self.arr_c, self.r_fps))
                continue
            if queue==self.recon_q:
                queue.put((self.arr_r, self.r_fps))
                continue
            if queue==self.filters_r_q:
                continue

            queue.put(parameter)

        self.capture = Process(target=capture, args=self.capture_queues)
        self.capture.start()
        self.reconstruction = Process(target=reconstruct, args=self.recon_queues)
        self.reconstruction.start()

        # Initialize all the elements of the gui at the same time, only once
        self.init_viewing_frame()
        self.init_parameters_frame()
        self.init_filters_frame()
        self.init_saving_frame()

    def init_viewing_frame(self):
        # Frame for navigation
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=8, width=MENU_FRAME_WIDTH)
        self.navigation_frame.grid(row=0, column=0, padx=5, sticky='nsew')

        # Extra space goes to the last row of the menu, which is the home button and theme list
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        # Prevents dynamic adjusting of size that gets messed up from the change in length of numbers
        self.navigation_frame.grid_propagate(False)

        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky='nsew')

        # Extra vertical space goes to the title
        self.viewing_frame.grid_rowconfigure(1, weight=1)

        # Empty columns handle extra space so the element in column 1 is centered
        self.viewing_frame.columnconfigure(0, weight=1)
        self.viewing_frame.columnconfigure(1, weight=0)
        self.viewing_frame.columnconfigure(2, weight=1)

        ## Elements and layout of the navigation frame

        # Main title for the navigation frame
        self.main_title_nav = ctk.CTkLabel(self.navigation_frame, text='DLHM Reconstruction', compound='left', font=ctk.CTkFont(size=15, weight='bold'))
        self.main_title_nav.grid(row=0, column=0, padx=20, pady=40)

        # Common configuration for the buttons
        mb_config = {'corner_radius':6, 
                                'height':MENU_BUTTONS_HEIGHT,
                                'width':MENU_FRAME_WIDTH,
                                'border_spacing':10, 
                                'fg_color':("gray75", "gray25"), 
                                'text_color':("gray10", "gray90"), 
                                'hover_color':("gray80", "gray20"),
                                'anchor':"c"}
        
        # Same thing
        mb_grid_config = {'sticky':'ew', 'padx':1, 'pady':3}

        text_config = {'compound':'left', 'font':ctk.CTkFont(size=15, weight='bold')}

        self.param_button = ctk.CTkButton(self.navigation_frame, text='Parameters', **mb_config, command=lambda: self.change_menu_to('parameters'))
        self.param_button.grid(row=1, column=0, **mb_grid_config)

        self.filters_button = ctk.CTkButton(self.navigation_frame, text='Filters', **mb_config, command=lambda: self.change_menu_to('filters'))
        self.filters_button.grid(row=2, column=0, **mb_grid_config)

        self.it_button = ctk.CTkButton(self.navigation_frame, text='Image Tools', **mb_config)
        self.it_button.grid(row=3, column=0, **mb_grid_config)

        self.so_button = ctk.CTkButton(self.navigation_frame, text='Saving Options', **mb_config, command=lambda: self.change_menu_to('so'))
        self.so_button.grid(row=4, column=0, **mb_grid_config)


        # Theme selection menu
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=5, column=0, padx=20, pady=20, sticky="s")



        ## Elements and layout of the viewing frame

        # Main title for the viewing frame
        self.main_title_view = ctk.CTkLabel(self.viewing_frame, text='DLHM Viewing Window', compound='left', font=ctk.CTkFont(size=15, weight='bold'), anchor=ctk.CENTER)
        self.main_title_view.grid(row=0, column=1, padx=20, pady=40, sticky='nsew')

        # An image frame containing the captured image and the processed image
        self.image_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.image_frame.grid(row=1, column=1, padx=20, pady=15, sticky='ne')

        self.captured_title_label = ctk.CTkLabel(self.image_frame, text='Captured Image', **text_config)
        self.captured_title_label.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
        self.captured_label = ctk.CTkLabel(self.image_frame, image=self.img_c, text='')
        self.captured_label.grid(row=1, column=0, padx=20, pady=20, sticky='nsew')

        # For displaying frames per second (actual real life time, not tick time)
        self.c_fps_label = ctk.CTkLabel(self.image_frame, text=f'FPS: {self.c_fps}')
        self.c_fps_label.grid(row=2, column=0, padx=20, pady=20)

        self.processed_title_label = ctk.CTkLabel(self.image_frame, text='Processed Image', **text_config)
        self.processed_title_label.grid(row=0, column=1, padx=20, pady=20, sticky='nsew')
        self.processed_label = ctk.CTkLabel(self.image_frame, image=self.img_r, text='')
        self.processed_label.grid(row=1, column=1, padx=20, pady=20, sticky='nsew')
        # For displaying frames per second (actual real life time, not tick time)
        self.r_fps_label = ctk.CTkLabel(self.image_frame, text=f'FPS: {self.r_fps}')
        self.r_fps_label.grid(row=2, column=1, padx=20, pady=20)
        
        # Buttons for saving and changing image scale

        self.saving_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.saving_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=20, sticky='ws')

        self.size_label = ctk.CTkLabel(self.saving_frame, text='Viewing Size:')
        self.size_label.grid(row=0, column=0, padx=20)
        
        self.size_slider = ctk.CTkSlider(self.saving_frame, width=100, from_=MIN_IMG_SCALE, to=MAX_IMG_SCALE, command=self.update_im_size)
        self.size_slider.grid(row=0, column=1, padx=10, pady=20)
        self.size_slider.set(self.scale)

        self.save_captured_button = ctk.CTkButton(self.saving_frame, text='Save Capture', command=self.save_capture)
        self.save_captured_button.grid(row=0, column=2, padx=20, pady=20)

        self.save_processed_button = ctk.CTkButton(self.saving_frame, text='Save Reconstruction', command=self.save_processed)
        self.save_processed_button.grid(row=0, column=3, padx=20, pady=20)
        
        self.camera_settings_button = ctk.CTkButton(self.saving_frame, text='Open Camera Settings', command=lambda:self.settings_q.put(True))
        self.camera_settings_button.grid(row=0, column=4, padx=20, pady=20)

        # For displaying frames per second (actual real life time, not tick time)
        self.w_fps_label = ctk.CTkLabel(self.saving_frame, text=f'FPS: {self.w_fps}')
        self.w_fps_label.grid(row=0, column=5, padx=20, pady=20)

    def init_parameters_frame(self):
        # Menu with the parameter options 

        self.parameters_frame = ctk.CTkFrame(self, corner_radius=8, width=PARAMETER_FRAME_WIDTH)
        
        self.parameters_frame.grid_propagate(False)

        self.main_title_param= ctk.CTkLabel(self.parameters_frame, text='Parameters')
        self.main_title_param.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')

        self.magnification_label = ctk.CTkLabel(self.parameters_frame, text=f'Magnificación: {round(self.scale_factor, 4)}')
        self.magnification_label.grid(row=1, column=0, pady=20, sticky='ew')

        self.variables_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.variables_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.variables_frame.grid_propagate(False)

        self.variables_frame.columnconfigure(0, weight=1)
        self.variables_frame.columnconfigure(1, weight=0)
        self.variables_frame.columnconfigure(2, weight=0)
        self.variables_frame.columnconfigure(3, weight=1)

        self.lambda_label = ctk.CTkLabel(self.variables_frame, text='Wavelength')
        self.lambda_label.grid(row=0, column=1, sticky='ew', padx=5)

        self.dxy_label = ctk.CTkLabel(self.variables_frame, text='Pixel pitch')
        self.dxy_label.grid(row=0, column=2, sticky='ew', padx=5)

        self.lambda_entry = ctk.CTkEntry(self.variables_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{DEFAULT_WAVELENGTH}')
        self.lambda_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

        self.dxy_entry = ctk.CTkEntry(self.variables_frame, width=PARAMETER_ENTRY_WIDTH, placeholder_text=f'{DEFAULT_DXY}')
        self.dxy_entry.grid(row=1, column=2, sticky='ew', padx=5, pady=2)

        self.set_variables = ctk.CTkButton(self.variables_frame, width=PARAMETER_BUTTON_WIDTH, text='Set', command=self.set_variables)
        self.set_variables.grid(row=1, column=4, sticky='ew', padx=10)

        # Frame for L parameters
        self.L_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.L_frame.grid(row=3, column=0, sticky='ew', pady=2)
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


        # Frame for z parameters
        self.Z_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.Z_frame.grid(row=4, column=0, sticky='ew', pady=2)
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


        # Frame for r parameters
        self.r_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.r_frame.grid(row=5, column=0, sticky='ew', pady=2)
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


        # Options for fixing are and displaying intensity instead of amplitude
        self.adit_options_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.adit_options_frame.grid(row=6, column=0, sticky='ew', pady=2)

        self.adit_options_frame.rowconfigure(0, weight=1)
        self.adit_options_frame.rowconfigure(1, weight=0)
        self.adit_options_frame.rowconfigure(2, weight=1)

        self.adit_options_frame.columnconfigure(0, weight=1)
        self.adit_options_frame.columnconfigure(1, weight=0)
        self.adit_options_frame.columnconfigure(2, weight=0)
        self.adit_options_frame.columnconfigure(3, weight=0)
        self.adit_options_frame.columnconfigure(4, weight=1)

        self.adit_options_frame.grid_propagate(False)

        self.fix_r_checkbox = ctk.CTkCheckBox(self.adit_options_frame, text='Fix r', variable=self.fix_r)
        self.fix_r_checkbox.grid(row=1, column=1, sticky='ew', padx=10, pady=5)

        self.square_field_checkbox = ctk.CTkCheckBox(self.adit_options_frame, text='Show Intensity', variable=self.square_field)
        self.square_field_checkbox.grid(row=1, column=2, sticky='ew', padx=10, pady=5)

        self.phase_r_checkbox = ctk.CTkCheckBox(self.adit_options_frame, text='Show Phase', variable=self.phase_r)
        self.phase_r_checkbox.grid(row=1, column=3, sticky='nsew', padx=10, pady=5)

        # Frame for selecting the reconstruction method with radio buttons
        self.algorithm_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT)
        self.algorithm_frame.grid(row=7, column=0, sticky='ew', pady=2)

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

        # Frame to redefine the limits of the sliders for L, Z and r
        self.limits_frame = ctk.CTkFrame(self.parameters_frame, width=PARAMETER_FRAME_WIDTH, height=PARAMETER_FRAME_HEIGHT+LIMITS_FRAME_EXTRA_SPACE)
        self.limits_frame.grid(row=8, column=0, sticky='ew', pady=2)

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

        # Sets all the limits provided in the entries, all at once
        self.set_limits_button = ctk.CTkButton(self.limits_frame, width=PARAMETER_BUTTON_WIDTH, text='Set all', command=self.set_limits)
        self.set_limits_button.grid(row=1, column=4, sticky='ew', padx=10)

        # Resets them to their initial values
        self.restore_limits_button = ctk.CTkButton(self.limits_frame, width=PARAMETER_BUTTON_WIDTH, text='Restore all', command=self.restore_limits)
        self.restore_limits_button.grid(row=2, column=4, sticky='ew', padx=10)

        
        self.parameters_frame.rowconfigure(9, weight=1)
        
        self.home_button = ctk.CTkButton(self.parameters_frame, text='Home', command=lambda: self.change_menu_to('home'))
        self.home_button.grid(row=9, column=0, pady=20, sticky='s')

    def init_filters_frame(self):
        # Frame to activate and configure image enhancement filters
        self.filters_frame = ctk.CTkFrame(self, corner_radius=8, width=FILTER_FRAME_WIDTH)
        self.filters_frame.grid_propagate(False)

        self.main_title_filters = ctk.CTkLabel(self.filters_frame, text='Filters')
        self.main_title_filters.grid(row=0, column=0, padx=20, pady=40, sticky='nsew')

        # Frame for selecting the image with radio buttons for the filter to be applied
        self.choose_image_frame = ctk.CTkFrame(self.filters_frame, width=FILTER_FRAME_WIDTH, height=FILTER_FRAME_HEIGHT)
        self.choose_image_frame.grid(row=1, column=0, sticky='ew', pady=2)
        self.choose_image_frame.grid_propagate(False)

        self.choose_image_frame.columnconfigure(0, weight=1)
        self.choose_image_frame.columnconfigure(1, weight=0)
        self.choose_image_frame.columnconfigure(2, weight=0)
        self.choose_image_frame.columnconfigure(3, weight=1)

        self.choose_image_title = ctk.CTkLabel(self.choose_image_frame, text=f'Selector de imagen:')
        self.choose_image_title.grid(row=0, column=1, columnspan=2, sticky='ew', pady=5)

        self.ca_image_radio = ctk.CTkRadioButton(self.choose_image_frame, text='Captured image', variable=self.filter_image_var, value='CA', command=self.update_image_filters)
        self.ca_image_radio.grid(row=1, column=1, sticky='ew', padx=10, pady=5)

        self.pr_image_radio = ctk.CTkRadioButton(self.choose_image_frame, text='Processed image', variable=self.filter_image_var, value='PR', command=self.update_image_filters)
        self.pr_image_radio.grid(row=1, column=2, sticky='ew', padx=10, pady=5)

        # Gamma filter section
        self.gamma_frame = ctk.CTkFrame(self.filters_frame, width=FILTER_FRAME_WIDTH, height=FILTER_FRAME_HEIGHT)
        self.gamma_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.gamma_frame.grid_propagate(False)

        self.gamma_checkbox = ctk.CTkCheckBox(self.gamma_frame, text='Gamma filter', variable=self.gamma_checkbox_var, command=self.update_manual_filter)
        self.gamma_checkbox.grid(row=0, column=0, sticky='ew', pady=10, padx=10)

        self.gamma_slider = ctk.CTkSlider(self.gamma_frame, height=SLIDER_HEIGHT, from_=MIN_GAMMA, to=MAX_GAMMA, command=self.adjust_gamma)
        self.gamma_slider.grid(row=1, column=0, sticky='ew', pady=10, padx=10)
        self.gamma_slider.setvar(value=0)

        # Contrast filter section
        self.contrast_frame = ctk.CTkFrame(self.filters_frame, width=FILTER_FRAME_WIDTH, height=FILTER_FRAME_HEIGHT)
        self.contrast_frame.grid(row=3, column=0, sticky='ew', pady=2)
        self.contrast_frame.grid_propagate(False)

        self.contrast_checkbox = ctk.CTkCheckBox(self.contrast_frame, text='Contrast filter', variable=self.contrast_checkbox_var, command=self.update_manual_filter)
        self.contrast_checkbox.grid(row=0, column=0, sticky='ew', pady=10, padx=10)

        self.contrast_slider = ctk.CTkSlider(self.contrast_frame, height=SLIDER_HEIGHT, from_=MIN_CONTRAST, to=MAX_CONTRAST, command=self.adjust_contrast)
        self.contrast_slider.grid(row=1, column=0, sticky='ew', pady=10, padx=10)
        self.contrast_slider.setvar(value=1)

        # Adaptive Equalization filter section (Ecualización adaptativa)
        self.adaptative_eq_frame = ctk.CTkFrame(self.filters_frame, width=FILTER_FRAME_WIDTH, height=FILTER_FRAME_HEIGHT)
        self.adaptative_eq_frame.grid(row=4, column=0, sticky='ew', pady=2)
        self.adaptative_eq_frame.grid_propagate(False)

        self.adaptative_eq_checkbox = ctk.CTkCheckBox(self.adaptative_eq_frame, text='Adaptive Equalization', variable=self.adaptative_eq_checkbox_var, command=self.update_manual_filter)
        self.adaptative_eq_checkbox.grid(row=0, column=0, sticky='ew', pady=10, padx=10)

        # High-Pass Butterworth filter section
        self.highpass_frame = ctk.CTkFrame(self.filters_frame, width=FILTER_FRAME_WIDTH, height=FILTER_FRAME_HEIGHT)
        self.highpass_frame.grid(row=5, column=0, sticky='ew', pady=2)
        self.highpass_frame.grid_propagate(False)

        self.highpass_checkbox = ctk.CTkCheckBox(self.highpass_frame, text='High-Pass Butterworth filter', variable=self.highpass_checkbox_var, command=self.update_manual_filter)
        self.highpass_checkbox.grid(row=0, column=0, sticky='ew', pady=10, padx=10)

        self.highpass_slider = ctk.CTkSlider(self.highpass_frame, height=SLIDER_HEIGHT, from_=MIN_CUTOFF, to=MAX_CUTOFF, command=self.adjust_highpass)
        self.highpass_slider.grid(row=1, column=0, sticky='ew', pady=10, padx=10)
        self.highpass_slider.setvar(value=DEFAULT_CUTOFF)

        # Low-Pass Butterworth filter section (Nuevo filtro Low-Pass)
        self.lowpass_frame = ctk.CTkFrame(self.filters_frame, width=FILTER_FRAME_WIDTH, height=FILTER_FRAME_HEIGHT)
        self.lowpass_frame.grid(row=6, column=0, sticky='ew', pady=2)
        self.lowpass_frame.grid_propagate(False)

        self.lowpass_checkbox = ctk.CTkCheckBox(self.lowpass_frame, text='Low-Pass Butterworth filter', variable=self.lowpass_checkbox_var, command=self.update_manual_filter)
        self.lowpass_checkbox.grid(row=0, column=0, sticky='ew', pady=10, padx=10)

        self.lowpass_slider = ctk.CTkSlider(self.lowpass_frame, height=SLIDER_HEIGHT, from_=MIN_CUTOFF, to=MAX_CUTOFF, command=self.adjust_lowpass)
        self.lowpass_slider.grid(row=1, column=0, sticky='ew', pady=10, padx=10)
        self.lowpass_slider.setvar(value=DEFAULT_CUTOFF)

        # Row configuration
        self.filters_frame.rowconfigure(8, weight=1)

        # Home button
        self.home_button = ctk.CTkButton(self.filters_frame, text='Home', command=lambda: self.change_menu_to('home'))
        self.home_button.grid(row=8, column=0, pady=20, sticky='s')

    def init_saving_frame(self):
        # Frame to activate and configure image enhancement filters
        self.so_frame = ctk.CTkFrame(self, corner_radius=8, width=SAVING_FRAME_WIDTH)
        self.so_frame.grid_propagate(False)

        self.main_title_so= ctk.CTkLabel(self.so_frame, text='Saving Options')
        self.main_title_so.grid(row=0, column=0, padx=20, pady=40, sticky='nsew')

        self.static_frame = ctk.CTkFrame(self.so_frame, width=SAVING_FRAME_WIDTH, height=SAVING_FRAME_HEIGHT)
        self.static_frame.grid(row=1, column=0, sticky='ew', pady=2)
        self.static_frame.grid_propagate(False)

        self.static_frame.columnconfigure(0, weight=1)
        self.static_frame.columnconfigure(1, weight=0)
        self.static_frame.columnconfigure(2, weight=0)
        self.static_frame.columnconfigure(3, weight=1)

        self.static_button = ctk.CTkButton(self.static_frame, text='Use static image', command=self.selectfile)
        self.static_button.grid(row=0, column=1, padx=20, pady=20)

        self.real_button = ctk.CTkButton(self.static_frame, text='Real time view', command=self.return_to_stream)
        self.real_button.grid(row=0, column=2, padx=20, pady=20)

        self.nofilter_frame = ctk.CTkFrame(self.so_frame, width=SAVING_FRAME_WIDTH, height=SAVING_FRAME_HEIGHT)
        self.nofilter_frame.grid(row=2, column=0, sticky='ew', pady=2)
        self.nofilter_frame.grid_propagate(False)

        self.nofilter_frame.columnconfigure(0, weight=1)
        self.nofilter_frame.columnconfigure(1, weight=0)
        self.nofilter_frame.columnconfigure(2, weight=0)
        self.nofilter_frame.columnconfigure(3, weight=1)

        self.nf_title_label = ctk.CTkLabel(self.nofilter_frame, text='Guardado sin filtros')
        self.nf_title_label.grid(row=0, column=1, columnspan=2, padx=20, pady=5, sticky='nsew')

        self.nf_c_button = ctk.CTkButton(self.nofilter_frame, text='Guardar captura', command=self.no_filter_save_c)
        self.nf_c_button.grid(row=1, column=1, padx=20, pady=20)
        self.nf_r_button = ctk.CTkButton(self.nofilter_frame, text='Guardar reconstrucción', command=self.no_filter_save_r)
        self.nf_r_button.grid(row=1, column=2, padx=20, pady=20)


        self.so_frame.rowconfigure(8, weight=1)
        
        self.home_button = ctk.CTkButton(self.so_frame, text='Home', command=lambda: self.change_menu_to('home'))
        self.home_button.grid(row=8, column=0, pady=20, sticky='s')

    def no_filter_save_c(self):
        '''Saves a capture with an increasing number'''
        i = 0
        while os.path.exists("saves/capture/capture%s.bmp" % i):
            i += 1
        im_c = arr2im(self.arr_c)
        im_c.save('saves/capture/capture%s.bmp' % i)

    def no_filter_save_r(self):
        '''Saves a reconstruction with an increasing number'''
        i = 0
        while os.path.exists("saves/reconstruction/reconstruction%s.bmp" % i):
            i += 1

        im_r = arr2im(self.arr_r)
        im_r.save('saves/reconstruction/reconstruction%s.bmp' % i)

    def set_variables(self):
        try:
            self.wavelength = float(self.lambda_entry.get())
        except:
            self.wavelength = DEFAULT_WAVELENGTH
            print('Invalid number entered as wavelength')
        
        try:
            self.dxy = float(self.dxy_entry.get())
        except:
            self.dxy = DEFAULT_DXY
            print('Invalid number entered as pixel width')

        print(f'Wavelength: {self.wavelength}, DXY: {self.dxy}')

    def update_image_filters(self):
        if self.filter_image_var.get()=='CA':
            self.gamma_checkbox_var.set(value=self.manual_gamma_c_var.get())
            self.gamma_slider.set(self.gamma_c)
            self.contrast_checkbox_var.set(value=self.manual_contrast_c_var.get())
            self.contrast_slider.set(self.contrast_c)
            self.adaptative_eq_checkbox_var.set(value=self.manual_adaptative_eq_c_var.get())
            self.highpass_checkbox_var.set(value=self.manual_highpass_c_var.get())
            self.highpass_slider.set(self.highpass_c)
            self.lowpass_checkbox_var.set(value=self.manual_lowpass_c_var.get())
            self.lowpass_slider.set(self.lowpass_c)
            
        elif self.filter_image_var.get()=='PR':
            self.gamma_checkbox_var.set(value=self.manual_gamma_r_var.get())
            self.gamma_slider.set(self.gamma_r)
            self.contrast_checkbox_var.set(value=self.manual_contrast_r_var.get())
            self.contrast_slider.set(self.contrast_r)
            self.adaptative_eq_checkbox_var.set(value=self.manual_adaptative_eq_r_var.get())
            self.highpass_checkbox_var.set(value=self.manual_highpass_r_var.get())
            self.highpass_slider.set(self.highpass_r)
            self.lowpass_checkbox_var.set(value=self.manual_lowpass_r_var.get())
            self.lowpass_slider.set(self.lowpass_r)

    def update_manual_filter(self):
        if self.filter_image_var.get()=='CA':
            self.manual_gamma_c_var.set(value=self.gamma_checkbox_var.get())
            self.manual_contrast_c_var.set(value=self.contrast_checkbox_var.get())
            self.manual_adaptative_eq_c_var.set(value=self.adaptative_eq_checkbox_var.get())
            self.manual_highpass_c_var.set(value=self.highpass_checkbox_var.get())
            self.manual_lowpass_c_var.set(value=self.lowpass_checkbox_var.get())
        elif self.filter_image_var.get()=='PR':
            self.manual_gamma_r_var.set(value=self.gamma_checkbox_var.get())
            self.manual_contrast_r_var.set(value=self.contrast_checkbox_var.get())
            self.manual_adaptative_eq_r_var.set(value=self.adaptative_eq_checkbox_var.get())
            self.manual_highpass_r_var.set(value=self.highpass_checkbox_var.get())
            self.manual_lowpass_r_var.set(value=self.lowpass_checkbox_var.get())


        if self.manual_gamma_c_var.get() or self.manual_gamma_r_var.get():
            self.adjust_gamma(self.gamma_slider.get())

        if self.manual_contrast_c_var.get() or self.manual_contrast_r_var.get():
            self.adjust_contrast(self.contrast_slider.get())

        if self.manual_adaptative_eq_c_var.get() or self.manual_adaptative_eq_r_var.get():
            self.adjust_adaptative_eq()

        if self.manual_highpass_c_var.get() or self.manual_highpass_r_var.get():
            self.adjust_highpass(self.highpass_slider.get())
            print(self.highpass_slider.get())

        if self.manual_lowpass_c_var.get() or self.manual_lowpass_r_var.get():
            self.adjust_highpass(self.lowpass_slider.get())
            print(self.lowpass_slider.get())

    def adjust_gamma(self, val):
        if (self.filter_image_var.get()=='CA'):
            if self.manual_gamma_c_var.get():
                self.gamma_c = val

        if (self.filter_image_var.get()=='PR'):
            if self.manual_gamma_r_var.get():
                self.gamma_r = val

    def adjust_contrast(self, val):
        if (self.filter_image_var.get()=='CA'):
            if self.manual_contrast_c_var.get():
                self.contrast_c = val

        if (self.filter_image_var.get()=='PR'):
            if self.manual_contrast_r_var.get():
                self.contrast_r = val

    def adjust_adaptative_eq(self):
        if (self.filter_image_var.get()=='CA'):
            if self.manual_adaptative_eq_c_var.get():
                self.adaptative_eq_c = True
            
        if (self.filter_image_var.get()=='PR'):
            if self.manual_adaptative_eq_r_var.get():
                self.adaptative_eq_r = True

    def adjust_highpass(self, val):
        if (self.filter_image_var.get()=='CA'):
            if self.manual_highpass_c_var.get():
                self.highpass_c = val
            
        if self.filter_image_var.get()=='PR':
            if self.manual_highpass_r_var.get():
                self.highpass_r = val

    def adjust_lowpass(self, val):
        if (self.filter_image_var.get()=='CA'):
            if self.manual_lowpass_c_var.get():
                self.lowpass_c = val
            
        if self.filter_image_var.get()=='PR':
            if self.manual_lowpass_r_var.get():
                self.lowpass_r = val

    def update_parameters(self):
        '''Updates all slider values, magnification and scale factor'''
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
        '''Updates the value of L based on the slider'''
        self.L = val

        # Z depends on r and L, if r is fixed, Z and L move together
        if self.fix_r.get():
            self.Z = self.L-self.r
        else:
            # neither Z nor r can be larger than L
            if self.L<=self.Z:
                self.Z = self.L

            self.r = self.L-self.Z

        self.update_parameters()

    def update_Z(self, val):
        '''Updates the value of Z based on the slider'''

        self.Z = val

        # L depends on Z and r, if r is fixed L and Z move together
        # if not, r is just the difference between L and Z
        if self.fix_r.get():
            self.L = self.Z+self.r
        else:

            # L cannot be lower than Z
            if self.Z >= self.L:
                self.L = self.Z
        
            self.r = self.L-self.Z


        self.update_parameters()

    def update_r(self, val):
        '''Updates the value of r based on the slider'''

        self.r = val

        # If r is fixed, Z will be fixed since it's more probable to be correct
        if self.fix_r.get():
            self.L = self.Z+self.r
        else:
            self.Z = self.L-self.r

        self.update_parameters()

    def set_value_L(self):
        '''Allows to enter a specific value from entry, handles empty or mistaken entries'''
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
        '''Allows to enter a specific value from entry, handles empty or mistaken entries'''

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
        '''Allows to enter a specific value from entry, handles empty or mistaken entries'''

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
        '''Handles the limits and the entry of none or mistaken values'''
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
        '''Sets the parameters to their initial values'''
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
        '''Allows to change the menu frame that's being shown'''
        if name=='home':
            self.navigation_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.navigation_frame.grid_forget()
        
        if name=='parameters':
            self.parameters_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.parameters_frame.grid_forget()

        if name=='filters':
            self.filters_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.filters_frame.grid_forget()

        if name=='so':
            self.so_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        else:
            self.so_frame.grid_forget()

    def update_im_size(self, size):
        '''Updates scale from slider'''
        self.scale = size

    def save_capture(self, ext:str='bmp'):
        '''Saves a capture with an increasing number'''
        i = 0
        while os.path.exists("saves/capture/capture%s.bmp" % i):
            i += 1

        self.im_c.save('saves/capture/capture%s.bmp' % i)

    def save_processed(self, ext:str='bmp'):
        '''Saves a capture of reconstruction with an increasing number'''
        i = 0
        while os.path.exists("saves/reconstruction/reconstruction%s.bmp" % i):
            i += 1

        self.im_r.save('saves/reconstruction/reconstruction%s.bmp' % i)

    def change_appearance_mode_event(self, new_appearance_mode):
        '''Changes between light and dark mode.'''
        ctk.set_appearance_mode(new_appearance_mode)

    def selectfile(self):
        self.file_path = ctk.filedialog.askopenfilename(title="Selecciona un archivo de imagen")
        if self.file_path:
            if self.file_path_q.empty():
                self.file_path_q.put(self.file_path)

    def return_to_stream(self):
        self.file_path = ''
        if self.file_path_q.empty():
            self.file_path_q.put(self.file_path)

    def draw(self):
        '''Handles capture and processing of the images from the camera'''

        w_start_time = time.time()  # Para medir el tiempo de fps


        self.recon_parameters = [self.arr_c,
                        self.arr_r,
                        self.filters_r,
                        self.algorithm_var.get(),
                        self.L,
                        self.Z,
                        self.r,
                        self.wavelength,
                        self.dxy,
                        self.scale_factor,
                        self.FC,
                        self.square_field.get(),
                        self.phase_r.get()]

        for queue, parameter in zip(self.recon_queues, self.recon_parameters):
            if queue==self.captured_q:
                continue
            if queue==self.recon_q:
                continue
            if queue==self.filters_r_q:
                continue
            if queue.empty():
                queue.put(parameter)

        self.filters_c = []
        filter_params_c = []

        if self.manual_contrast_c_var.get():
            self.filters_c.append('contrast')
            filter_params_c.append(self.contrast_c)

        if self.manual_gamma_c_var.get():
            self.filters_c.append('gamma')
            filter_params_c.append(self.gamma_c)

        if self.manual_adaptative_eq_c_var.get():
            self.filters_c.append('adaptative_eq')
            filter_params_c.append([])

        if self.manual_highpass_c_var.get():
            self.filters_c.append('highpass')
            filter_params_c.append(self.highpass_c)

        if self.manual_lowpass_c_var.get():
            self.filters_c.append('lowpass  ')
            filter_params_c.append(self.lowpass_c)
        
        if self.file_path_q.empty():
            self.file_path_q.put(self.file_path)

        if self.filters_c_q.empty():
            self.filters_c_q.put((self.filters_c, filter_params_c))

        if not self.captured_q.empty():
            capture = self.captured_q.get()
            self.arr_c = capture[0]
            self.c_fps = capture[1]
        
        if not self.filtered_q.empty():
            self.arr_c_f = self.filtered_q.get()
        
        if not (self.width_q.empty() or self.height_q.empty()):
            self.width, self.height = self.width_q.get(), self.height_q.get()

        self.im_c = arr2im(self.arr_c_f)  # Convierte el array a imagen
        self.img_c = create_image(self.im_c, self.width, self.height)
        self.img_c._size = (self.width * self.scale, self.height * self.scale)
        self.captured_label.img = self.img_c
        self.captured_label.configure(image=self.img_c)

        self.filters_r = []
        filter_params_r = []

        if self.manual_contrast_r_var.get():
            self.filters_r.append(contrast_filter)
            filter_params_r.append(self.contrast_r)

        if self.manual_gamma_r_var.get():
            self.filters_r.append(gamma_filter)
            filter_params_r.append(self.gamma_r)

        if self.manual_adaptative_eq_r_var.get():
            self.filters_r.append(adaptative_eq_filter)
            filter_params_r.append([])

        if self.manual_highpass_r_var.get():
            self.filters_r.append(highpass_filter)
            filter_params_r.append(self.highpass_r)

        if self.manual_lowpass_r_var.get():
            self.filters_r.append(lowpass_filter)
            filter_params_r.append(self.lowpass_r)
        
        if self.file_path_q.empty():
            self.file_path_q.put(self.file_path)

        if self.filters_r_q.empty():
            self.filters_r_q.put((self.filters_r, filter_params_r))

        if not self.recon_q.empty():
            # Procesar y normalizar antes de mostrar
            reconstruction = self.recon_q.get()
            self.arr_r = reconstruction[0]
            self.r_fps = reconstruction[1]
        
            self.arr_r = np.uint8(normalize(self.arr_r, 255))  # Normaliza la imagen

        self.im_r = arr2im(self.arr_r)  # Convierte el array a imagen
        self.img_r = create_image(self.im_r, self.width, self.height)
        self.img_r._size = (self.width * self.scale, self.height * self.scale)
        self.processed_label.img = self.img_r
        self.processed_label.configure(image=self.img_r)

        w_end_time = time.time()  # Solo estas líneas cuentan para el fps

        w_elapsed_time = w_end_time - w_start_time
        self.w_fps = round(1 / w_elapsed_time, 1)
        self.w_fps_label.configure(text=f'FPS: {self.w_fps}')
        self.c_fps_label.configure(text=f'FPS: {self.c_fps}')
        self.r_fps_label.configure(text=f'FPS: {self.r_fps}')

        self.after(15, self.draw)

    def check_current_FC(self):
        self.FC = filtcosenoF(self.cosine_period, np.array((self.width, self.height)))
        plt.imshow(self.FC, cmap='gray')
        plt.show()

    def set_FC_param(self, cosine_period):
        self.cosine_period = cosine_period

    def reset_FC_param(self):
        self.cosine_period = DEFAULT_COSINE_PERIOD

    def release(self):
        # Safer
        os.system("taskkill /f /im python.exe")

if __name__=='__main__':
    app = App()
    #app.check_current_FC()
    app.draw()
    app.mainloop()
    app.release()