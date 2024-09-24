        self.L_frame = ctk.CTkFrame(self.parameters_frame)
        self.L_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky='nsew')

        self.L_slider_title = ctk.CTkLabel(self.L_frame, text=f'Distancia entre la cámara y la fuente (L): {round(self.L, 4)}')
        self.L_slider_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 5), sticky='nsew')

        self.L_slider = ctk.CTkSlider(self.L_frame, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, corner_radius=8, from_=MIN_L, to=MAX_L, command=self.update_L)
        self.L_slider.grid(row=1, column=0, padx=20, pady=25, sticky='nsew')
        self.L_slider.set(round(self.L, 4))

        self.L_slider_entry = ctk.CTkEntry(self.L_frame, placeholder_text=f'{round(self.L, 4)}', width=50)
        self.L_slider_entry.grid(row=1, column=1, padx=20, pady=20, sticky='nsew')
        self.L_slider_entry.setvar(value=f'{round(self.L, 4)}')

        self.L_slider_button = ctk.CTkButton(self.L_frame, text='Set', command=self.set_value_L, width=50)
        self.L_slider_button.grid(row=1, column=2, padx=5, pady=20, sticky='nsew')


        # Frame para los parámetros de Z
        self.Z_frame = ctk.CTkFrame(self.parameters_frame)
        self.Z_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky='nsew')

        self.Z_slider_title = ctk.CTkLabel(self.Z_frame, text=f'Distancia entre la muestra y la fuente (z): {round(self.Z, 4)}', width=50)
        self.Z_slider_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 5), sticky='nsew')

        self.Z_slider = ctk.CTkSlider(self.Z_frame, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, corner_radius=8, from_=MIN_Z, to=MAX_Z, command=self.update_Z)
        self.Z_slider.grid(row=1, column=0, padx=20, pady=25, sticky='nsew')
        self.Z_slider.set(round(self.Z, 4))

        self.Z_slider_entry = ctk.CTkEntry(self.Z_frame, placeholder_text=f'{round(self.Z, 4)}', width=50)
        self.Z_slider_entry.grid(row=1, column=1, padx=20, pady=20, sticky='nsew')
        self.Z_slider_entry.setvar(value=f'{round(self.Z, 4)}')

        self.Z_slider_button = ctk.CTkButton(self.Z_frame, text='Set', command=self.set_value_Z, width=50)
        self.Z_slider_button.grid(row=1, column=2, padx=5, pady=20, sticky='nsew')


        # Frame para los parámetros de r
        self.r_frame = ctk.CTkFrame(self.parameters_frame)
        self.r_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky='nsew')

        self.r_slider_title = ctk.CTkLabel(self.r_frame, text=f'Distancia de reconstrucción (r): {round(self.r, 4)}', width=50)
        self.r_slider_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 5), sticky='nsew')

        self.r_slider = ctk.CTkSlider(self.r_frame, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, corner_radius=8, from_=MIN_L, to=MAX_L, command=self.update_r)
        self.r_slider.grid(row=1, column=0, padx=20, pady=25, sticky='nsew')
        self.r_slider.set(round(self.r, 4))

        self.r_slider_entry = ctk.CTkEntry(self.r_frame, placeholder_text=f'{round(self.r, 4)}', width=50)
        self.r_slider_entry.grid(row=1, column=1, padx=20, pady=20, sticky='nsew')
        self.r_slider_entry.setvar(value=f'{round(self.r, 4)}')

        self.r_slider_button = ctk.CTkButton(self.r_frame, text='Set', command=self.set_value_r, width=50)
        self.r_slider_button.grid(row=1, column=2, padx=5, pady=20, sticky='nsew')