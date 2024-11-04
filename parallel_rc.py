import numpy as np
import time
from customtkinter import CTkImage
from multiprocessing import Queue
from kreuzer_functions import kreuzer3F

from settings import *
from _3DHR_Utilities import *



def im2arr(path: str):
    '''Converts file image into numpy array.'''
    return np.asarray(Image.open(path).convert('L'))

def arr2im(array: np.ndarray):
    '''Converts numpy array into PhotoImage type'''
    return Image.fromarray(array.astype(np.uint8), 'L')

def create_image(img: Image.Image, width, height):
    '''Converts image into type usable by customtkinter'''
    return CTkImage(light_image=img, dark_image=img, size=(width, height))

def capture(image:Queue,
            path:Queue,
            width:Queue,
            height:Queue,
            settings:Queue,
            ):
    
    # Initialize camera (0 by default most of the time means the integrated camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Verify that the camera opened correctly
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        exit()

    # Sets the camera resolution to the closest chose in settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)

    # Gets the actual resolution of the image
    width_  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_ = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f'Width: {width_}')
    print(f'Height: {height_}')
        
    while True:
        init_time = time.time()
        # Captura la imagen de la cámara
        img= cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        img = cv2.flip(img, 1)  # Voltea horizontalmente

        if not path.empty():
            path_ = path.get()

            if path_:
                img = im2arr(path_)

                # Gets the actual resolution of the image
                height_, width_ = img.shape

        if not settings.empty():
            if settings.get():
                open_camera_settings(cap)
        
        end_time = time.time()

        elapsed_time = end_time-init_time
        fps = round(1 / elapsed_time, 1)

        if image.empty():
            image.put((img, fps))
            width.put(width_)
            height.put(height_)
        
def open_camera_settings(cap):
    try:
        cap.set(cv2.CAP_PROP_SETTINGS, 0)
    except:
        print('Cannot access camera settings.')

def reconstruct(image:Queue,
                output:Queue,
                algorithm:Queue,
                L:Queue,
                Z:Queue,
                r:Queue,
                wavelength:Queue,
                dxy:Queue, 
                scale_factor:Queue,
                FC:Queue,
                squared:Queue,
                phase:Queue
                ):
    while True:
        if not (image.empty() or
                algorithm.empty() or
                L.empty() or
                Z.empty() or
                r.empty() or
                wavelength.empty() or
                dxy.empty() or
                scale_factor.empty() or
                FC.empty() or
                squared.empty() or 
                phase.empty()):
            
            init_time = time.time()

            field = np.sqrt(normalize(image.get()[0], 1))

            alg = algorithm.get()
            dxy_ = dxy.get()
            L_ = L.get()
            Z_ = Z.get()
            r_ = r.get()
            wavelength_ = wavelength.get()
            scale_factor_ = scale_factor.get()
            FC_ = FC.get()


            if alg == 'AS':
                recon = propagate(field, r_, wavelength_, dxy_, dxy_, scale_factor_)
            elif alg == 'KR':
                deltaX = Z_*dxy_/L_
                recon = kreuzer3F(field, Z_, L_, wavelength_, dxy_, deltaX, FC_)

            sq = squared.get()
            ph = phase.get()

            if sq:
                    arr = normalize(np.abs(recon)**2,1)
            elif not sq and ph:
                arr = normalize(np.angle(recon),1)
            else:
                arr = normalize(np.abs(recon),1)

            end_time = time.time()
            elapsed_time = end_time-init_time
            fps = round(1 / elapsed_time, 1)

            if output.empty():
                output.put((arr, fps))  