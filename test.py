import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import kreuzer_functions as kf

# Configuración de la ventana de CustomTkinter
window = ctk.CTk()
window.title("Captura de cámara con procesamiento de hologramas")

# Variable para almacenar el stream de la cámara
cap = cv2.VideoCapture(0)

# Función que se ejecuta cada vez que se actualiza el frame de la cámara
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convertir el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Generar un holograma ficticio (en este caso aplicando un filtro básico)
        FC = kf.filtcosenoF(5, gray.shape[0])
        hologram_reconstructed = kf.kreuzer3F(gray, z=1000, L=0.5, wavelength=650e-9, dx=0.001, deltaX=0.001, FC=FC)
        
        # Convertir el holograma reconstruido en una imagen que pueda mostrar en tkinter
        hologram_img = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(hologram_reconstructed * 255)))

        # Mostrar la imagen en la etiqueta
        label.imgtk = hologram_img
        label.configure(image=hologram_img)
    
    # Llamar a esta función repetidamente
    window.after(10, update_frame)

# Etiqueta para mostrar el video
label = ctk.CTkLabel(window)
label.pack()

# Iniciar la actualización del video
update_frame()

# Iniciar la aplicación
window.mainloop()

# Liberar la cámara al cerrar la ventana
cap.release()