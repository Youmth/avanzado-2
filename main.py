import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('DLHM GUI')
        self.geometry('1366x768')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


        # Create two frames, one for navigation
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=8)
        self.navigation_frame.grid(row=0, column=0, sticky='nsew')
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.viewing_frame = ctk.CTkFrame(self, corner_radius=8)
        self.viewing_frame.grid(row=0, column=1, sticky='nsew')
        self.viewing_frame.grid_columnconfigure(0, weight=1)
        self.viewing_frame.grid_columnconfigure(1, weight=1)

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
        self.main_title_view = ctk.CTkLabel(self.viewing_frame, text='DLHM Reconstruction', compound='left', font=ctk.CTkFont(size=15, weight='bold'), anchor=ctk.CENTER)
        self.main_title_view.grid(row=0, column=0, padx=20, pady=40, columnspan=2, sticky='nsew')

        # An image frame containing the captured image and the processed image
        self.image_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.image_frame.grid(row=1, column=0, padx=20, pady=20, sticky='e')

        ##### Test
        arr = self.im2arr("sample_images/test_image.jpeg")
        im = self.arr2im(arr)
        
        img = self.create_image(im)

        self.captured_label = ctk.CTkLabel(self.image_frame, image=img, text='')
        self.captured_label.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
        
        self.processed_label = ctk.CTkLabel(self.image_frame, image=img, text='')
        self.processed_label.grid(row=0, column=1, padx=20, pady=20, sticky='nsew')
        #####

        self.saving_frame = ctk.CTkFrame(self.viewing_frame, corner_radius=8)
        self.saving_frame.grid(row=2, column=0, padx=20, pady=20, sticky='e')

        self.save_captured_button = ctk.CTkButton(self.saving_frame, text='Save Capture')
        self.save_captured_button.grid(row=0, column=0, padx=20, pady=20)

        self.save_processed_button = ctk.CTkButton(self.saving_frame, text='Save Reconstruction')
        self.save_processed_button.grid(row=0, column=1, padx=20, pady=20)



    def im2arr(self, path: str):
        '''Converts file image into numpy array.'''
        return np.asarray(Image.open(path))

    def arr2im(self, array: np.ndarray):
        '''Converts numpy array into PhotoImage type'''
        return Image.fromarray(array)
    
    def create_image(self, img: Image.Image, size= tuple):
        return ctk.CTkImage(light_image=img, dark_image=img, size=(400, 400))

    def change_appearance_mode_event(self, new_appearance_mode):
        '''Changes between light and dark mode.'''
        ctk.set_appearance_mode(new_appearance_mode)





    #     self.home_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
    #                                                fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
    #                                                  anchor="w", command=self.home_button_event)
    #     self.home_button.grid(row=1, column=0, sticky="ew")

    #     self.frame_2_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 2",
    #                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
    #                                                 anchor="w", command=self.frame_2_button_event)
    #     self.frame_2_button.grid(row=2, column=0, sticky="ew")

    #     self.frame_3_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 3",
    #                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
    #                                                   anchor="w", command=self.frame_3_button_event)
    #     self.frame_3_button.grid(row=3, column=0, sticky="ew")

    #     self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
    #                                                             command=self.change_appearance_mode_event)
    #     self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

    #             # create home frame
    #     self.home_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
    #     self.home_frame.grid_columnconfigure(0, weight=1)

    #     # self.home_frame_large_image_label = ctk.CTkLabel(self.home_frame, text="", image=self.large_test_image)
    #     # self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

    #     # self.home_frame_button_1 = ctk.CTkButton(self.home_frame, text="", image=self.image_icon_image)
    #     # self.home_frame_button_1.grid(row=1, column=0, padx=20, pady=10)
    #     # self.home_frame_button_2 = ctk.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="right")
    #     # self.home_frame_button_2.grid(row=2, column=0, padx=20, pady=10)
    #     # self.home_frame_button_3 = ctk.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="top")
    #     # self.home_frame_button_3.grid(row=3, column=0, padx=20, pady=10)
    #     # self.home_frame_button_4 = ctk.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="bottom", anchor="w")
    #     # self.home_frame_button_4.grid(row=4, column=0, padx=20, pady=10)

    #     # create second frame
    #     self.second_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

    #     # create third frame
    #     self.third_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

    #     # select default frame
    #     self.select_frame_by_name("home")

    # def select_frame_by_name(self, name):
    #     # set button color for selected button
    #     self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
    #     self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
    #     self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

    #     # show selected frame
    #     if name == "home":
    #         self.home_frame.grid(row=0, column=1, sticky="nsew")
    #     else:
    #         self.home_frame.grid_forget()
    #     if name == "frame_2":
    #         self.second_frame.grid(row=0, column=1, sticky="nsew")
    #     else:
    #         self.second_frame.grid_forget()
    #     if name == "frame_3":
    #         self.third_frame.grid(row=0, column=1, sticky="nsew")
    #     else:
    #         self.third_frame.grid_forget()

    # def home_button_event(self):
    #     self.select_frame_by_name("home")

    # def frame_2_button_event(self):
    #     self.select_frame_by_name("frame_2")

    # def frame_3_button_event(self):
    #     self.select_frame_by_name("frame_3")

    # def change_appearance_mode_event(self, new_appearance_mode):
    #     ctk.set_appearance_mode(new_appearance_mode)



    def run(self):
        pass

if __name__=='__main__':
    app = App()
    app.mainloop()