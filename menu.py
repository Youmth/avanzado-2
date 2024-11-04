import numpy as np
import customtkinter as ctk
import time
import os
from multiprocessing import Process, Queue
from kreuzer_functions import filtcosenoF
from skimage import exposure, filters

from settings import *
from _3DHR_Utilities import *
from parallel_rc import *

class Menu:
    '''Class for a default menu frame for DLHM GUI'''
    def __init__(self, 
                 root:ctk.CTkFrame, 
                 row:int=0, 
                 col:int=0, 
                 width:float=DEFAULT_MENU_WIDTH, 
                 name:str='menu', 
                 title:str='Menu',
                 corner:float=8,
                 padx:tuple|float=DEFAULT_MENU_PADX,
                 pady:tuple|float=DEFAULT_MENU_PADY,
                 title_padx:tuple|float=DEFAULT_TITLE_PADX,
                 title_pady:tuple|float=DEFAULT_TITLE_PADY,
                 font:ctk.CTkFont=None,
                 scrollable:bool=True) -> None:
        
        if scrollable:
            self.frame = ctk.CTkScrollableFrame(root, 
                                                width, 
                                                corner_radius=corner, 
                                                label_text=title, 
                                                label_font=font,
                                                )
        else:
            self.frame = ctk.CTkFrame(root, width, corner_radius=corner)
            self.title_lab = ctk.CTkLabel(self.frame, width, font=font, text=title)
            self.title_lab.grid(row=0, column=0, padx=title_padx, pady=title_pady)


        self.row = row
        self.col = col
        self.width = width
        self.name = name
        self.title = title
        self.corner = corner
        self.padx = padx
        self.pady = pady
        self.font = font
        self.t_padx = title_padx
        self.t_pady = title_pady

        self.lowlim = DEFAULT_MINIMUM
        self.highlim = DEFAULT_MAXIMUM

        self.modules = {'routing':[],
                        'tuning':[],
                        'checklist':[]}

    def add_checklist(self,
                         title_:str,
                         row:int,
                         col:int,
                         labels:tuple,
                         variables:tuple,
                         width:float=DEFAULT_CHECKLIST_WIDTH,
                         height:float=DEFAULT_CHECKLIST_HEIGHT,
                         padx:tuple|float=DEFAULT_CHECKLIST_PADX,
                         pady:tuple|float=DEFAULT_CHECKLIST_PADY,
                         corner:float=8,
                         spadx:tuple|float=DEFAULT_CHECKLIST_SPADX,
                         spady:tuple|float=DEFAULT_CHECKLIST_SPADY):
        
        frame = ctk.CTkFrame(self.frame, width, height, corner)
        frame.grid(row=row, column=col, padx=padx, pady=pady, sticky='ew')
        frame.columnconfigure(0, weight=1)
        for i in range(1, len(variables)):
            frame.columnconfigure(i, weight=0)
        frame.columnconfigure(len(variables), weight=1)

        frame.grid_propagate(False)

        title = ctk.CTkLabel(frame, width, corner_radius=corner, text=title_)
        title.grid(row=0, column=0, columnspan=len(variables)+1, sticky='ew', padx=spadx, pady=spady)

        rcol = 1
        checkboxes = [] 

        for label, var in zip(labels, variables):
            print(rcol)
            cb = ctk.CTkCheckBox(frame, corner_radius=corner, text=label, variable=var)
            cb.grid(row=1, column=rcol, padx=spadx, pady=spady, sticky='ew')
            checkboxes.append(cb)
            rcol+=1

        widgets = {'frame':frame, 
                   'title':title, 
                   'checkbox':checkboxes
                }

        return widgets



    def add_tuning(self, 
                   label:str,
                   row:int,
                   col:int,
                   update,
                   set_value,
                   init_val:int=0,
                   limits:tuple=(0, 1),
                   width:float=DEFAULT_TUNING_WIDTH,
                   height:float=DEFAULT_TUNING_HEIGHT,
                   padx:tuple|float=DEFAULT_TUNING_PADX,
                   pady:tuple|float=DEFAULT_TUNING_PADY,
                   corner:float=8,
                   spadx:tuple|float=DEFAULT_TUNING_SPADX,
                   spady:tuple|float=DEFAULT_TUNING_SPADY,
                   sheight:float=SLIDER_HEIGHT,
                   ewidth:float=PARAMETER_ENTRY_WIDTH,
                   bwidth:float=PARAMETER_BUTTON_WIDTH,
                   digits:int=4):
        # Frame for L parameters
        frame = ctk.CTkFrame(self.frame, width, height, corner)
        frame.grid(row=row, column=col, padx=padx, pady=pady, sticky='ew')
        frame.columnconfigure(0, weight=2)
        frame.grid_propagate(False)

        title = ctk.CTkLabel(frame, text=f'{label}   {round(init_val, digits)}')
        title.grid(row=0, column=0, columnspan=3, sticky='ew', padx=spadx, pady=spady)

        slider = ctk.CTkSlider(frame, height=sheight, corner_radius=corner, from_=limits[0], to=limits[1], command=update)
        slider.grid(row=1, column=0, sticky='ew')
        slider.set(round(init_val, digits))

        entry = ctk.CTkEntry(frame, width=ewidth, placeholder_text=f'{round(init_val, digits)}')
        entry.grid(row=1, column=1, sticky='ew', padx=spadx, pady=spady)
        entry.setvar(value=f'{round(init_val, digits)}')

        button = ctk.CTkButton(frame, width=bwidth, text='Set', command=set_value)
        button.grid(row=1, column=2, sticky='ew', padx=spadx, pady=spady)

        lowlim = ctk.CTkEntry(frame, width=ewidth, placeholder_text=f'{round(limits[0], digits)}')
        lowlim.grid(row=2, column=0, padx=spadx, pady=spady, sticky='w')
        highlim = ctk.CTkEntry(frame, width=ewidth, placeholder_text=f'{round(limits[1], digits)}')
        highlim.grid(row=2, column=0, padx=spadx, pady=spady, sticky='e')

        setlim = ctk.CTkButton(frame, width=bwidth, text='Set limits', command=lambda:self.set_tuning_limits(lowlim.get(),
                                                                                                             highlim.get(),
                                                                                                             slider))
        setlim.grid(row=2, column=1, columnspan=2, padx=spadx, pady=spady, sticky='ew')

        widgets = {'frame':frame, 
                   'title':title, 
                   'slider':slider, 
                   'entry':(entry, lowlim, highlim), 
                   'button':(button, setlim)
                }

        self.modules['tuning'].append(widgets)
        
        return widgets
    
    def update_tuning_parameters(self, 
                                 widgets:dict, 
                                 label:str, 
                                 value:float,
                                 decimals:int=4):
        '''Updates all slider values, magnification and scale factor'''
        widgets['title'].configure(text=f'{label}   {round(value, decimals)}')
        widgets['slider'].set(round(value, decimals))
        widgets['entry'][0].configure(placeholder_text=f'{round(value, decimals)}')

    def set_tuning_limits(self,
                          min_val:str,
                          max_val:str,
                          slider:ctk.CTkSlider) -> None:
        try:
            min_ = float(min_val)
        except:
            min_ = 0
        try:
            max_ = float(max_val)
        except:
            max_ = 0
            
        slider.configure(from_=min_, to=max_)   

    def add_routing(self,
                    menu:object,
                    row:int=0,
                    col:int=0,
                    text:str='Go to Page',
                    width:float=DEFAULT_ROUTING_WIDTH,
                    height:float=DEFAULT_ROUTING_HEIGHT,
                    padx:tuple|float=DEFAULT_ROUTING_PADX,
                    pady:tuple|float=DEFAULT_ROUTING_PADY,
                    corner:float=8
                    ) -> ctk.CTkButton:
        
        button = ctk.CTkButton(self.frame, width, height, corner, text=text, command=lambda:self.routing(menu))
        button.grid(row=row, column=col, padx=padx, pady=pady, sticky='ew')

        self.modules['routing'].append(button)
        return button
    
    def add_routing_arr(self, 
                        menus:tuple,
                        texts:tuple,
                        init_row:int=0,
                        col:int=0,
                        width:float=DEFAULT_ROUTING_WIDTH,
                        height:float=DEFAULT_ROUTING_HEIGHT,
                        padx:tuple|float=DEFAULT_ROUTING_PADX,
                        pady:tuple|float=DEFAULT_ROUTING_PADY,
                        corner:float=8,
                        ) -> list:
    
        row = init_row
        buttons = []

        for menu, text in zip(menus, texts):
            buttons.append(self.add_routing(menu, row, col, text, width, height, padx, pady, corner))
            row+=1
        
        return buttons

    def place(self):
        self.frame.grid(row=self.row, 
                        column=self.col, 
                        padx=self.padx,
                        pady=self.pady,
                        sticky='nsew')
        self.frame.grid_propagate()
        
    def remove(self):
        self.frame.grid_forget()

    def routing(self, menu:object) -> None:
        self.remove()
        menu.place()