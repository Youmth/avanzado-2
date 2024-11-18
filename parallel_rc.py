import numpy as np
import time
from customtkinter import CTkImage
from multiprocessing import Queue
from kreuzer_functions import kreuzer3F
from skimage import exposure, filters

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

def gamma_filter(arr, gamma):
    return np.uint8(np.clip(arr + gamma * 255, 0, 255))

def contrast_filter(arr, contrast):
    return np.uint8(np.clip(arr * contrast, 0, 255))

def adaptative_eq_filter(arr, _):
    arr = exposure.equalize_adapthist(normalize(arr, 1), clip_limit=DEFAULT_CLIP_LIMIT)
    return np.uint8(arr * 255)  # Convertir de 0-1 a 0-255

def highpass_filter(arr, freq):
    arr = filters.butterworth(normalize(arr, 1), freq, high_pass=True)
    return np.uint8(arr*255)

def lowpass_filter(arr, freq):
    arr = filters.butterworth(normalize(arr, 1), freq, high_pass=False)
    return np.uint8(arr*255)

def capture(queue_manager:dict[dict[Queue, Queue], dict[Queue, Queue], dict[Queue, Queue]]):
    
    filter_dict =  {'gamma':gamma_filter,
                    'contrast':contrast_filter,
                    'adaptative_eq':adaptative_eq_filter,
                    'highpass':highpass_filter,
                    'lowpass':lowpass_filter}
    
    input_dict = {'path':None,
                  'reference path':None,
                  'settings':None,
                  'filters':None,
                  'filter':None}
    
    output_dict = {'image':None,
                   'filtered':None,
                   'fps':None,
                   'size':None}

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
        img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        img = cv2.flip(img, 1)  # Voltea horizontalmente

        filt_img = img

        height_, width_ = img.shape

        if not queue_manager['capture']['input'].empty():
            input = queue_manager['capture']['input'].get()

            for key in input_dict.keys():
                input_dict[key] = input[key]

        if input_dict['path']:
            img = im2arr(input_dict['path'])
            filt_img = img

            # Gets the actual resolution of the image
            height_, width_ = img.shape

        if input_dict['reference path']:
            ref = im2arr(input_dict['reference path'])
            if img.shape == ref.shape:
                img = img-ref
            else:
                print('Image sizes do not match')
            
            filt_img = img

        if input_dict['settings']:
            open_camera_settings(cap)
        
        if input_dict['filters']:
            filter_functions = input_dict['filters'][0]
            filter_params = input_dict['filters'][1]

            if input_dict['filter']:
                for filter, param, in zip(filter_functions, filter_params):
                    filt_img = filter_dict[filter](filt_img, param)
        
        end_time = time.time()

        elapsed_time = end_time-init_time
        fps = round(1 / elapsed_time, 1)

        if not queue_manager['capture']['output'].full():
            
            output_dict['image']= img
            output_dict['filtered'] = filt_img
            output_dict['fps'] = fps
            output_dict['size'] = (width_, height_)

            queue_manager['capture']['output'].put(output_dict)

        
def open_camera_settings(cap):
    try:
        cap.set(cv2.CAP_PROP_SETTINGS, 0)
        print('Settings opened')
    except:
        print('Cannot access camera settings.')

def reconstruct(queue_manager:dict[dict[Queue, Queue], dict[Queue, Queue], dict[Queue, Queue]]):
    filter_dict =  {'gamma':gamma_filter,
                    'contrast':contrast_filter,
                    'adaptative_eq':adaptative_eq_filter,
                    'highpass':highpass_filter,
                    'lowpass':lowpass_filter}

    input_dict = {'image':None,
                  'filters':None,
                  'filter':None,
                  'algorithm':None,
                  'L':None,
                  'Z':None,
                  'r':None,
                  'wavelength':None,
                  'dxy':None,
                  'scale_factor':None,
                  'FC':None,
                  'squared':None,
                  'phase':None
                  }
    
    output_dict = {'image':None,
                   'filtered':None,
                   'fps':None
                   }
    while True:
        if not queue_manager['reconstruction']['input'].empty():
            # We want this processing to ocurr only if there is an image to process
            
            init_time = time.time()

            input = queue_manager['reconstruction']['input'].get()

            for key in input_dict.keys():
                input_dict[key] = input[key]

            field = np.sqrt(normalize(input_dict['image'], 1))

            if input_dict['algorithm'] == 'AS':
                recon = propagate(field, 
                                  input_dict['r'], 
                                  input_dict['wavelength'], 
                                  input_dict['dxy'], 
                                  input_dict['dxy'], 
                                  input_dict['scale_factor'])
            elif input_dict['algorithm'] == 'KR':

                Z = input_dict['Z']
                L = input_dict['L']
                dxy = input_dict['dxy']

                deltaX = Z*dxy/L
                recon = kreuzer3F(field, Z, L, input_dict['wavelength'], dxy, deltaX, input_dict['FC'])

            sq = input_dict['squared']
            ph = input_dict['phase']

            if sq:
                arr = normalize(np.abs(recon)**2,255)
            elif not sq and ph:
                arr = normalize(np.angle(recon),255)
            else:
                arr = normalize(np.abs(recon),255)

            filt_img = arr

            if input_dict['filters']:
                filter_functions = input_dict['filters'][0]
                filter_params = input_dict['filters'][1]

                if input_dict['filter']:
                    for filter, param, in zip(filter_functions, filter_params):
                        filt_img = filter_dict[filter](filt_img, param)

            end_time = time.time()
            elapsed_time = end_time-init_time
            fps = round(1 / elapsed_time, 1)

            if not queue_manager['reconstruction']['output'].full():
                
                output_dict['image']= arr.astype(np.uint8)
                output_dict['filtered']=filt_img.astype(np.uint8)
                output_dict['fps'] = fps

                queue_manager['reconstruction']['output'].put(output_dict)