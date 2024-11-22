import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

def nm2ev(energy):
    return 1239.81 / energy

def ev2nm(wavelength):
    return 1239.81 / wavelength


def gaussian_broadening(x, y, sigma):
    """ 
    Add gaussian broadening for a spectrum defined by
    x: energies or wavelengths
    y: intensites
    with vectorization 
    """
    x = np.array(x).reshape((len(x), 1))
    xi = np.repeat(x, repeats = x.shape[0], axis = 1)
    xj = np.repeat(np.transpose(x), repeats = x.shape[0], axis = 0)
    gaussian_matrix = np.exp(-(xi - xj)**2 / sigma **2)
    spec = np.sum(gaussian_matrix * np.array(y), axis = 1)
    return spec / np.max(spec)
    
    
class Spectrum:

    """A class to represent VIS absorption spectrum and convert it to xyz and rgb"""
    

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    cmf = np.array([[1.4000e-03, 0.0000e+00, 6.5000e-03], [2.2000e-03, 1.0000e-04, 1.0500e-02],
                    [4.2000e-03, 1.0000e-04, 2.0100e-02], [7.6000e-03, 2.0000e-04, 3.6200e-02],
                    [1.4300e-02, 4.0000e-04, 6.7900e-02], [2.3200e-02, 6.0000e-04, 1.1020e-01],
                    [4.3500e-02, 1.2000e-03, 2.0740e-01], [7.7600e-02, 2.2000e-03, 3.7130e-01],
                    [1.3440e-01, 4.0000e-03, 6.4560e-01], [2.1480e-01, 7.3000e-03, 1.0391e+00],
                    [2.8390e-01, 1.1600e-02, 1.3856e+00], [3.2850e-01, 1.6800e-02, 1.6230e+00],
                    [3.4830e-01, 2.3000e-02, 1.7471e+00], [3.4810e-01, 2.9800e-02, 1.7826e+00],
                    [3.3620e-01, 3.8000e-02, 1.7721e+00], [3.1870e-01, 4.8000e-02, 1.7441e+00],
                    [2.9080e-01, 6.0000e-02, 1.6692e+00], [2.5110e-01, 7.3900e-02, 1.5281e+00],
                    [1.9540e-01, 9.1000e-02, 1.2876e+00], [1.4210e-01, 1.1260e-01, 1.0419e+00],
                    [9.5600e-02, 1.3900e-01, 8.1300e-01], [5.8000e-02, 1.6930e-01, 6.1620e-01],
                    [3.2000e-02, 2.0800e-01, 4.6520e-01], [1.4700e-02, 2.5860e-01, 3.5330e-01],
                    [4.9000e-03, 3.2300e-01, 2.7200e-01], [2.4000e-03, 4.0730e-01, 2.1230e-01],
                    [9.3000e-03, 5.0300e-01, 1.5820e-01], [2.9100e-02, 6.0820e-01, 1.1170e-01],
                    [6.3300e-02, 7.1000e-01, 7.8200e-02], [1.0960e-01, 7.9320e-01, 5.7300e-02],
                    [1.6550e-01, 8.6200e-01, 4.2200e-02], [2.2570e-01, 9.1490e-01, 2.9800e-02],
                    [2.9040e-01, 9.5400e-01, 2.0300e-02], [3.5970e-01, 9.8030e-01, 1.3400e-02],
                    [4.3340e-01, 9.9500e-01, 8.7000e-03], [5.1210e-01, 1.0000e+00, 5.7000e-03],
                    [5.9450e-01, 9.9500e-01, 3.9000e-03], [6.7840e-01, 9.7860e-01, 2.7000e-03],
                    [7.6210e-01, 9.5200e-01, 2.1000e-03], [8.4250e-01, 9.1540e-01, 1.8000e-03],
                    [9.1630e-01, 8.7000e-01, 1.7000e-03], [9.7860e-01, 8.1630e-01, 1.4000e-03],
                    [1.0263e+00, 7.5700e-01, 1.1000e-03], [1.0567e+00, 6.9490e-01, 1.0000e-03],
                    [1.0622e+00, 6.3100e-01, 8.0000e-04], [1.0456e+00, 5.6680e-01, 6.0000e-04],
                    [1.0026e+00, 5.0300e-01, 3.0000e-04], [9.3840e-01, 4.4120e-01, 2.0000e-04],
                    [8.5440e-01, 3.8100e-01, 2.0000e-04], [7.5140e-01, 3.2100e-01, 1.0000e-04],
                    [6.4240e-01, 2.6500e-01, 0.0000e+00], [5.4190e-01, 2.1700e-01, 0.0000e+00],
                    [4.4790e-01, 1.7500e-01, 0.0000e+00], [3.6080e-01, 1.3820e-01, 0.0000e+00],
                    [2.8350e-01, 1.0700e-01, 0.0000e+00], [2.1870e-01, 8.1600e-02, 0.0000e+00],
                    [1.6490e-01, 6.1000e-02, 0.0000e+00], [1.2120e-01, 4.4600e-02, 0.0000e+00],
                    [8.7400e-02, 3.2000e-02, 0.0000e+00], [6.3600e-02, 2.3200e-02, 0.0000e+00],
                    [4.6800e-02, 1.7000e-02, 0.0000e+00], [3.2900e-02, 1.1900e-02, 0.0000e+00],
                    [2.2700e-02, 8.2000e-03, 0.0000e+00], [1.5800e-02, 5.7000e-03, 0.0000e+00],
                    [1.1400e-02, 4.1000e-03, 0.0000e+00], [8.1000e-03, 2.9000e-03, 0.0000e+00],
                    [5.8000e-03, 2.1000e-03, 0.0000e+00], [4.1000e-03, 1.5000e-03, 0.0000e+00],
                    [2.9000e-03, 1.0000e-03, 0.0000e+00], [2.0000e-03, 7.0000e-04, 0.0000e+00],
                    [1.4000e-03, 5.0000e-04, 0.0000e+00], [1.0000e-03, 4.0000e-04, 0.0000e+00],
                    [7.0000e-04, 2.0000e-04, 0.0000e+00], [5.0000e-04, 2.0000e-04, 0.0000e+00],
                    [3.0000e-04, 1.0000e-04, 0.0000e+00], [2.0000e-04, 1.0000e-04, 0.0000e+00],
                    [2.0000e-04, 1.0000e-04, 0.0000e+00], [1.0000e-04, 0.0000e+00, 0.0000e+00],
                    [1.0000e-04, 0.0000e+00, 0.0000e+00], [1.0000e-04, 0.0000e+00, 0.0000e+00],
                    [0.0000e+00, 0.0000e+00, 0.0000e+00]])
    # The CIE D65 light source intensity for 380 - 780 nm in 5 nm intervals
    d65 = np.array([ 49.9755,  52.3118,  54.6482,  68.7015,  82.7549,  87.1204,
                     91.486 ,  92.4589,  93.4318,  90.057 ,  86.6823,  95.7736,
                    104.865 , 110.936 , 117.008 , 117.41  , 117.812 , 116.336 ,
                    114.861 , 115.392 , 115.923 , 112.367 , 108.811 , 109.082 ,
                    109.354 , 108.578 , 107.802 , 106.296 , 104.79  , 106.239 ,
                    107.689 , 106.047 , 104.405 , 104.225 , 104.046 , 102.023 ,
                    100.    ,  98.1671,  96.3342,  96.0611,  95.788 ,  92.2368,
                     88.6856,  89.3459,  90.0062,  89.8026,  89.5991,  88.6489,
                     87.6987,  85.4936,  83.2886,  83.4939,  83.6992,  81.863 ,
                     80.0268,  80.1207,  80.2146,  81.2462,  82.2778,  80.281 ,
                     78.2842,  74.0027,  69.7213,  70.6652,  71.6091,  72.979 ,
                     74.349 ,  67.9765,  61.604 ,  65.7448,  69.8856,  72.4863,
                     75.087 ,  69.3398,  63.5927,  55.0054,  46.4182,  56.6118,
                     66.8054,  65.0941,  63.3828])
    
    # Adobe RGB (1998): D65
    xyz2rgb = np.array([[ 2.0413690, -0.5649464, -0.3446944],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0134474, -0.1183897,  1.0154096]])
                        
    rgb2xyz = np.array([[0.5767309,  0.1855540,  0.1881852],
                        [0.2973769,  0.6273491,  0.0752741],
                        [0.0270343,  0.0706872,  0.9911085]])
                        
                        
    def __init__(self, spectrum_range, spectrum_intensities, absorbance = True):
        
        self.spectrum_intensities = spectrum_intensities
        self.spectrum_range = spectrum_range
       
        #make sure spectra are sorted before the interpolation
        self.spectrum_range, self.spectrum_intensities = zip(*sorted(zip(self.spectrum_range, self.spectrum_intensities)))
        
        try:
           self.spectrum_intensities = np.array(self.spectrum_intensities).astype(float)
           self.spectrum_range = np.array(self.spectrum_range).astype(float)
        except:
           raise TypeError("Spectrum intensities or spectrum range cannot be converted into numpy arrays")
        
        # change potential negative values to 0
        self.spectrum_intensities[self.spectrum_intensities < 0] = 0.

        if absorbance:
            self.spectrum_intensities = 1 - 10**(-self.spectrum_intensities)

        # convert spectrum to range 380-780 nm with 5 nm step
        self.interp_spectrum = np.interp(np.linspace(380, 780, 81),
                                         self.spectrum_range,
                                         self.spectrum_intensities)
                                         
        #scale spectrum between 0 and 1 currently but do not do not check whether we recorded transmission or absorbance
        self.spectrum_intensities = self.spectrum_intensities / np.max(self.spectrum_intensities)
        

    def generate_XYZ(self):
        reflectance = (1 - self.interp_spectrum) * self.d65 # only reflected/transmitted light is perceived
        XYZ = np.sum(reflectance.reshape(81, 1) *           # reshape spectrum to make three channels
                     np.ones((81, 3)) *                     # shape is predefined 380 -> 780 with 5 nm step makes hardcodes number of spectrum values
                     Spectrum.cmf,                          # colour matching function for red, green, and blue 
                     axis = 0)
        
        if sum(XYZ) == 0.:
           return XYZ
           
        return XYZ / sum(XYZ)
        
        
    def generate_RGB(self):
        XYZ = self.generate_XYZ()
        rgb = self.xyz2rgb @ XYZ[:, np.newaxis]
        
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)
            
        return rgb.reshape(3)
        
    def rgb_to_hex(self):
        """Convert from fractional rgb values to HTML-style hex string."""
        
        hex_rgb = (255 * self.generate_RGB()).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)
    




