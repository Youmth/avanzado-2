import numpy as np
from numpy.fft import fftshift, fft2, ifftshift, ifftn
from _3DHR_Utilities import *
import matplotlib.pyplot as plt


def ang_spectrum(field, z, wavelength, dx, dy):
    """
    Función para difractar un campo complejo usando el método del espectro angular.

    Parámetros:
    field -- Campo complejo (array 2D)
    z -- Distancia de propagación
    wavelength -- Longitud de onda (lambda)
    dx, dy -- Tamaños de paso de muestreo en los ejes x e y

    Retorna:
    out -- Campo después de la propagación

    NOTA: Por consistencia no se usa esta función para kreuzer ya que tengo otra que tiene una diferencia mínima debido a los índices y que 
    además propaga en la otra dirección
    """
    N, M = field.shape

    # Crear malla de coordenadas
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))

    # Frecuencias espaciales
    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    # Transformada de Fourier del campo de entrada
    field_spec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))

    # Cálculo de la fase de propagación
    phase = np.exp(1j * z * 2 * np.pi * np.sqrt((1 / wavelength)**2 - (m * dfx)**2 - (n * dfy)**2))

    # Campo de salida después de la propagación
    out = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_spec * phase)))

    return out

def filtcosenoF(par, fi):
    # Coordenadas
    Xfc, Yfc = np.meshgrid(np.linspace(-fi/2, fi/2, fi), np.linspace(fi/2, -fi/2, fi))

    # Normalizar coordenadas en intervalo (-pi, pi) y crear filtros en direcciones horizontal y vertical
    FC1 = np.cos(Xfc * (np.pi / par) * (1 / np.max(np.abs(Xfc))))**2
    FC2 = np.cos(Yfc * (np.pi / par) * (1 / np.max(np.abs(Yfc))))**2

    # Intersecar ambas direcciones
    FC = (FC1 > 0) * FC1 * (FC2 > 0) * FC2

    # Re-escalar de 0 a 1
    FC = FC / np.max(FC)

    # plt.imshow(FC, cmap='gray')
    # plt.show()

    return FC

def normalize(image):
    # Normalización de la imagen al rango [0, 1]
    imageNormalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return imageNormalized

def point_src(M, z, x0, y0, wavelength, dx):
    """
    Función para crear una fuente puntual de iluminación centrada en (x0, y0) 
    y observada en un plano a una distancia z.
    
    Parámetros:
    M -- Tamaño de la matriz
    z -- Distancia de la pantalla
    x0, y0 -- Coordenadas del centro de la fuente
    wavelength -- Longitud de onda (lambda)
    dx -- Paso de muestreo (dx y dy son iguales)

    Retorna:
    P -- Matriz con la representación de la fuente puntual
    """
    N = M
    dy = dx

    # Crear malla de coordenadas
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))

    # Constante k
    k = 2 * np.pi / wavelength

    # Distancia r desde cada punto de la malla hasta el centro (x0, y0)
    r = np.sqrt(z**2 + (m * dx - x0)**2 + (n * dy - y0)**2)

    # Campo de la fuente puntual
    P = np.exp(1j * k * r) / r

    return P

def prepairholoF(CH_m, xop, yop, Xp, Yp):
    """
    Función para preparar el holograma usando la interpolación del vecino más cercano.

    Parámetros:
    CH_m -- Matriz del holograma original
    xop, yop -- Coordenadas del centro del holograma
    Xp, Yp -- Nuevas coordenadas

    Retorna:
    CHp_m -- Holograma preparado
    """
    row, _ = CH_m.shape

    # Nuevas coordenadas medidas en unidades de -2*xop/(row) (tamaño de píxel)
    Xcoord = (Xp - xop) / (-2 * xop / row)
    Ycoord = (Yp - yop) / (-2 * xop / row)

    # Encontrar el entero más bajo (aproximación hacia abajo)
    iXcoord = np.floor(Xcoord).astype(int)
    iYcoord = np.floor(Ycoord).astype(int)

    # Asegurarse de que no haya posiciones de píxeles nulas
    iXcoord[iXcoord == 0] = 1
    iYcoord[iYcoord == 0] = 1

    # Calcular la fracción para la interpolación
    x1frac = (iXcoord + 1.0) - Xcoord  # Valor superior al entero
    x2frac = 1.0 - x1frac              # Valor inferior al entero
    y1frac = (iYcoord + 1.0) - Ycoord
    y2frac = 1.0 - y1frac

    # Correspondencia de áreas de píxeles en cada dirección
    x1y1 = x1frac * y1frac
    x1y2 = x1frac * y2frac
    x2y1 = x2frac * y1frac
    x2y2 = x2frac * y2frac

    # Pre-asignar la matriz del holograma preparado
    CHp_m = np.zeros((row, row), dtype=complex)

    # Preparar el holograma (cada píxel remapeado)
    for it in range(row - 2):
        for jt in range(row - 2):
            CHp_m[iYcoord[it, jt], iXcoord[it, jt]] += (x1y1[it, jt]) * CH_m[it, jt]
            CHp_m[iYcoord[it, jt], iXcoord[it, jt] + 1] += (x2y1[it, jt]) * CH_m[it, jt]
            CHp_m[iYcoord[it, jt] + 1, iXcoord[it, jt]] += (x1y2[it, jt]) * CH_m[it, jt]
            CHp_m[iYcoord[it, jt] + 1, iXcoord[it, jt] + 1] += (x2y2[it, jt]) * CH_m[it, jt]

    return CHp_m


def kreuzer3F(CH_m, z, L, lambda_, deltax, deltaX, FC):
    """
    Función para reconstruir un holograma "in-line" usando la metodología de
    la patente de Jürgen Kreuzer.

    Parámetros:
    CH_m -- Matriz del holograma
    z -- Distancia de propagación
    L -- Longitud
    lambda_ -- Longitud de onda
    deltax, deltaX -- Tamaños de los píxeles en las respectivas etapas
    FC -- Filtro de coseno
    filename -- Nombre del archivo para guardar o cargar el holograma preparado

    Retorna:
    K -- Holograma reconstruido
    """
    
    # Tamaño de la matriz
    row = CH_m.shape[0]

    # Parámetros
    k = 2 * np.pi / lambda_
    W = deltax * row

    deltaY = deltaX

    # Coordenadas de la matriz
    X, Y = np.meshgrid(np.arange(1, row+1), np.arange(1, row+1))

    # Coordenadas de origen del holograma
    xo = -W / 2
    yo = -W / 2

    # Coordenadas de origen del holograma preparado
    xop = xo * L / np.sqrt(L**2 + xo**2)
    yop = yo * L / np.sqrt(L**2 + yo**2)

    # Tamaño del píxel para el holograma preparado
    deltaxp = xop / (-row / 2)
    deltayp = deltaxp

    # Coordenadas de origen para el plano de reconstrucción
    Xo = -deltaX * (row) / 2
    Yo = -deltaX * (row) / 2

    # Coordenadas Xp y Yp
    Xp = (deltax * (X - row/2) * L) / np.sqrt(L**2 + (deltax**2) * (X - row/2)**2 + (deltax**2) * (Y - row/2)**2)
    Yp = (deltax * (Y - row/2) * L) / np.sqrt(L**2 + (deltax**2) * (X - row/2)**2 + (deltax**2) * (Y - row/2)**2)

    CHp_m = prepairholoF(CH_m, xop, yop, Xp, Yp)

    # Fase de propagación
    Rp = np.sqrt(L**2 - (deltaxp*X + xop)**2 - (deltayp*Y + yop)**2)
    r = np.sqrt((deltaX**2) * ((X - row/2)**2 + (Y - row/2)**2) + z**2)
    
    CHp_m *= ((L / Rp)**4) * np.exp(-0.5j * k * (r**2 - 2*z*L) * Rp / (L**2))

    # Padding constante
    pad = row // 2

    # Padding en el filtro de coseno
    FC_padded = np.pad(FC, ((pad, pad), (pad, pad)), mode='constant')

    # Primera transformación
    T1 = CHp_m * np.exp((1j * k / (2 * L)) * (2*Xo*X*deltaxp + 2*Yo*Y*deltayp + (X**2)*deltaxp*deltaX + (Y**2)*deltayp*deltaY))
    T1 = np.pad(T1, ((pad, pad), (pad, pad)), mode='constant')
    T1 = fftshift(fft2(fftshift(T1 * FC_padded)))

    # Segunda transformación
    T2 = np.exp(-1j * (k / (2 * L)) * ((X - row/2)**2 * deltaxp * deltaX + (Y - row/2)**2 * deltayp * deltaY))
    T2 = np.pad(T2, ((pad, pad), (pad, pad)), mode='constant')
    T2 = fftshift(fft2(fftshift(T2 * FC_padded)))

    # Tercera transformación
    K = ifftshift(ifftn(ifftshift(T2 * T1)))
    K = K[pad:pad+row, pad:pad+row]

    return K


holo_contrast = plt.imread('saves/capture/capture0.bmp')

# Función de reconstrucción DLHM (Método de Kreuzer)
def reconstruct(hologram, z, L, lambda_, dx):
    fi, co = hologram.shape
    dX = (z) * dx / L  # Tamaño de pixel en el plano de reconstrucción

    FC = filtcosenoF(100, fi)

    K = kreuzer3F(hologram, z, L, lambda_, dx, dX, FC)

    amplitude = np.abs(K)
    phase = np.angle(K)
    
    return K, amplitude, phase, dX


# Visualización de imágenes
def show_image(image, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('equal')
    plt.show()


# Parámetros geométricos
z_rec = 2e-3
L = 15e-3
dx = 3.45e-6
lambda_blue = 457e-9

# Bucle para encontrar la mejor reconstrucción compensada
for z_rec in np.arange(1e-3, 3e-3, 0.1e-3):
    K, amplitude, phase, dX = reconstruct(holo_contrast, z_rec, L, lambda_blue, dx)
    show_image(amplitude**2, f'Amplitude at z_rec = {z_rec*1e3} mm')

# Compensación de la fase esférica
z_ini = 30e-3
z_fin = 50e-3
z_step = 1e-3
z_out = compensate_phase(K, holo_contrast.shape[0], lambda_blue, dX, z_ini, z_fin, z_step)

# Reconstrucción final compensada
Ph_comp_final, comp_field_final = recons_compens(z_out, K, holo_contrast.shape[0], lambda_blue, dX, 0)

show_image(Ph_comp_final, 'Compensated Phase')
show_image(np.abs(comp_field_final)**2, 'Amplitude Reconstructed')

# Espectro angular para reconstrucción con bajo NA
amplitude_stack = []
for z_rec in np.arange(1e-3, 5e-3, 0.1e-3):
    K = ang_spectrum(holo_contrast, z_rec, lambda_blue, dx, dx)
    amplitude = np.abs(K)**2
    image_normalized = normalize(amplitude)
    amplitude_stack.append(image_normalized)

# Reproducción de las imágenes apiladas
# Se puede usar cv2.imshow para mostrar las imágenes en una secuencia como un video
# O utilizar alguna otra librería que permita mostrar la secuencia

# Guardar la intensidad resultante como archivo
z_rec = 2e-3
K = ang_spectrum(holo_contrast, z_rec, lambda_blue, dx, dx)
G = np.abs(K)**2
MAXPhase = np.max(G)
MINPhase = np.min(G)
FilePhase = (G - MINPhase) / (MAXPhase - MINPhase) * 255
FilePhase = FilePhase.astype(np.uint8)

save_image(FilePhase, f'Resulting_intensity_{int(z_rec*1e3)}.jpg')




