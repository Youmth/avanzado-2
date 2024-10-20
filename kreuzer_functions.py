import numpy as np
from numpy.fft import fftshift, fft2, ifftshift, ifft2

def ang_spectrum(field, z, wavelength, dx, dy):
    """
    Propagates a complex field using the angular spectrum method.

    Parameters:
    field (2D array): Complex field
    z (float): Propagation distance
    wavelength (float): Wavelength
    dx, dy (float): Sampling step sizes along x and y axes

    Returns:
    out (2D array): Field after propagation
    """
    N, M = field.shape
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))
    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    field_spec = fftshift(fft2(fftshift(field)))
    phase = np.exp(1j * z * 2 * np.pi * np.sqrt((1 / wavelength) ** 2 - (m * dfx) ** 2 - (n * dfy) ** 2))
    out = ifftshift(ifft2(ifftshift(field_spec * phase)))

    return out


def filtcosenoF(par, fi):
    """
    Creates a cosine filter.

    Parameters:
    par (int): Cosine period
    fi (int): Size of the filter

    Returns:
    FC (2D array): Cosine filter
    """
    Xfc, Yfc = np.meshgrid(np.linspace(-fi / 2, fi / 2, fi), np.linspace(fi / 2, -fi / 2, fi))
    FC1 = np.cos(Xfc * (np.pi / par) * (1 / np.max(np.abs(Xfc)))) ** 2
    FC2 = np.cos(Yfc * (np.pi / par) * (1 / np.max(np.abs(Yfc)))) ** 2
    FC = (FC1 > 0) * FC1 * (FC2 > 0) * FC2
    FC = FC / np.max(FC)

    return FC


def normalize(image):
    """
    Normalizes an image to the [0, 1] range.

    Parameters:
    image (2D array): Input image

    Returns:
    imageNormalized (2D array): Normalized image
    """
    imageNormalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    return imageNormalized


def point_src(M, z, x0, y0, wavelength, dx):
    """
    Generates a point source at the coordinates (x0, y0) observed at a distance z.

    Parameters:
    M (int): Matrix size
    z (float): Propagation distance
    x0, y0 (float): Coordinates of the source center
    wavelength (float): Wavelength
    dx (float): Sampling step

    Returns:
    P (2D array): Field of the point source
    """
    N = M
    dy = dx
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))
    k = 2 * np.pi / wavelength
    r = np.sqrt(z ** 2 + (m * dx - x0) ** 2 + (n * dy - y0) ** 2)
    P = np.exp(1j * k * r) / r

    return P


def prepairholoF(hologram, xop, yop, Xp, Yp):
    """
    Prepares a hologram using nearest-neighbor interpolation.

    Parameters:
    hologram (2D array): Original hologram matrix
    xop, yop (float): Center coordinates of the hologram
    Xp, Yp (2D arrays): New coordinates

    Returns:
    CHp_m (2D array): Prepared hologram
    """
    row, _ = hologram.shape
    Xcoord = (Xp - xop) / (-2 * xop / row)
    Ycoord = (Yp - yop) / (-2 * xop / row)
    iXcoord = np.floor(Xcoord).astype(int)
    iYcoord = np.floor(Ycoord).astype(int)

    iXcoord[iXcoord == 0] = 1
    iYcoord[iYcoord == 0] = 1

    x1frac = (iXcoord + 1.0) - Xcoord
    x2frac = 1.0 - x1frac
    y1frac = (iYcoord + 1.0) - Ycoord
    y2frac = 1.0 - y1frac

    x1y1 = x1frac * y1frac
    x1y2 = x1frac * y2frac
    x2y1 = x2frac * y1frac
    x2y2 = x2frac * y2frac

    CHp_m = np.zeros((row, row), dtype=complex)

    for it in range(row - 2):
        for jt in range(row - 2):
            CHp_m[iYcoord[it, jt], iXcoord[it, jt]] += x1y1[it, jt] * hologram[it, jt]
            CHp_m[iYcoord[it, jt], iXcoord[it, jt] + 1] += x2y1[it, jt] * hologram[it, jt]
            CHp_m[iYcoord[it, jt] + 1, iXcoord[it, jt]] += x1y2[it, jt] * hologram[it, jt]
            CHp_m[iYcoord[it, jt] + 1, iXcoord[it, jt] + 1] += x2y2[it, jt] * hologram[it, jt]

    return CHp_m


def kreuzer3F(hologram, z, L, wavelength, dx, deltaX, FC):
    """
    Reconstructs an in-line hologram using Kreuzer's method.

    Parameters:
    hologram (2D array): Hologram matrix
    z (float): Propagation distance
    L (float): Length parameter
    wavelength (float): Wavelength
    dx, deltaX (float): Pixel sizes at different stages
    FC (2D array): Cosine filter

    Returns:
    K (2D array): Reconstructed hologram
    """
    row = hologram.shape[0]
    k = 2 * np.pi / wavelength
    W = dx * row

    deltaY = deltaX
    X, Y = np.meshgrid(np.arange(1, row + 1), np.arange(1, row + 1))

    xo = -W / 2
    yo = -W / 2
    xop = xo * L / np.sqrt(L ** 2 + xo ** 2)
    yop = yo * L / np.sqrt(L ** 2 + yo ** 2)

    deltaxp = xop / (-row / 2)
    deltayp = deltaxp
    Xo = -deltaX * row / 2
    Yo = -deltaX * row / 2

    Xp = (dx * (X - row / 2) * L) / np.sqrt(L ** 2 + (dx ** 2) * (X - row / 2) ** 2 + (dx ** 2) * (Y - row / 2) ** 2)
    Yp = (dx * (Y - row / 2) * L) / np.sqrt(L ** 2 + (dx ** 2) * (X - row / 2) ** 2 + (dx ** 2) * (Y - row / 2) ** 2)

    CHp_m = prepairholoF(hologram, xop, yop, Xp, Yp)

    Rp = np.sqrt(L ** 2 - (deltaxp * X + xop) ** 2 - (deltayp * Y + yop) ** 2)
    r = np.sqrt((deltaX ** 2) * ((X - row / 2) ** 2 + (Y - row / 2) ** 2) + z ** 2)
    CHp_m *= ((L / Rp) ** 4) * np.exp(-0.5j * k * (r ** 2 - 2 * z * L) * Rp / (L ** 2))

    pad = row // 2

    # Redimensionar FC si no coincide
    if FC.shape != CHp_m.shape:
        FC = np.pad(FC, ((pad, pad), (pad, pad)), mode='constant')

    T1 = CHp_m * np.exp((1j * k / (2 * L)) * (2 * Xo * X * deltaxp + 2 * Yo * Y * deltayp + (X ** 2) * deltaxp * deltaX + (Y ** 2) * deltayp * deltaY))

    # Asegurarse de que la matriz T1 también se redimensione correctamente
    if T1.shape != FC.shape:
        T1 = np.pad(T1, ((pad, pad), (pad, pad)), mode='constant')

    K = ang_spectrum(T1, L, wavelength, deltaX, deltaY)
    K = K[pad:pad + row, pad:pad + row]
    K = np.abs(K) ** 2
    K = normalize(K)
    K = K * FC[:K.shape[0], :K.shape[1]]  # Asegurar que las dimensiones coincidan

    return K