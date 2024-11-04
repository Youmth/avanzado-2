from turtle import update
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
from menu import Menu

window = ctk.CTk()
window.title('DLHM GUI')
titulo_fuente = ctk.CTkFont(family="Arial", size=24, weight="bold")

window.geometry('1366x768')
window.after(0, lambda:window.state('zoomed'))
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)




main = Menu(window, title=f'Main Menu', font=titulo_fuente, scrollable=False)
main.place()

param = Menu(window, title=f'Parameters Menu', font=titulo_fuente, scrollable=False)

def update_L(val):
    '''Updates the value of L based on the slider'''

    main.update_tuning_parameters(tuning_w, 'Distancia L: ', val)

def set_value_L():
    '''Allows to enter a specific value from entry, handles empty or mistaken entries'''
    try:
        val = float(tuning_w['entry'].get())
    except:
        val = 0

    update_L(val)

fix_r = ctk.BooleanVar(value=False)
intensity = ctk.BooleanVar(value=False)
phase = ctk.BooleanVar(value=False)

checks = {'Fix r':fix_r,
          'Show intensity':intensity,
          'Show phase':phase}

goParam = main.add_routing(param, 1, 0, 'Go to Parameters Menu')
goHome = param.add_routing(main, 2, 0, 'Go to Main Menu')

tuning_w = param.add_tuning('Distancia L: ', 1, 0, update=update_L, set_value=set_value_L)
checklist_w = param.add_checklist('Opciones Adicionales', 2, 0, list(checks.keys()), list(checks.values()))

window.mainloop()
