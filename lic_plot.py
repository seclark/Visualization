import colorsys
import numpy as np
import matplotlib as mpl
import seaborn.apionly as sns

def lic_plot(lic_data, background_data = None, F_m = 0.0, F_M = 0.0, cmap = "YlOrRd"):

    """
    Code to visualize an LIC plot. By Susan Clark.
    
    lic_data        :: output of LIC code
    background_data :: background color map, e.g. density or vector magnitude
    F_m             :: contrast enhancement parameter - see below
    F_M             :: contrast enhancement parameter - see below
    cmap            :: matplotlib recognized colormap
    
    Contrast Enhancement from http://www.paraview.org/Wiki/ParaView/Line_Integral_Convolution#Image_LIC_CE_stages
    L_ij = (L_ij - m) / (M - m)
    L = HSL lightness. m = lightness to map to 0. M = lightness to map to 1.
    m = min(L) + F_m * (max(L) - min(L))
    M = max(L) - F_M * (max(L) - min(L))
    F_m and F_M take values between 0 and 1. Increase F_m -> darker colors. Increase F_M -> brighter colors.
    
    """

    # 1. Compute nhi values
    # 2. Interpolate these values onto cmap to find corresponding RGBA value
    # 3. Convert RGB value to HSV, use these Hues + Saturations
    # 4. Assign Lightness as lic amplitude
    # 5. Display HLS map.
    
    # Normalize background data
    if background_data == None:
        background_data = np.ones(lic_data.shape)
    hues = background_data / np.nanmax(background_data)
    
    sats = np.ones(lic_data.shape)
    licmax = np.nanmax(lic_data)
    licmin = np.nanmin(lic_data)
    
    # Contrast enhancement
    m = licmin + F_m * (licmax - licmin)
    M = licmax - F_M * (licmax - licmin)
    vals = (lic_data - m) / (M - m)
    
    y, x = hues.shape
    
    # Map background data onto RGB colormap 
    cmap = mpl.colors.ListedColormap(sns.color_palette(cmap, 256))
    background_data_rgb = cmap(background_data)
    
    # Only need RGB, not RGBA
    background_data_rgb = background_data_rgb[:, :, 0:3]
    
    # Map to Hue - Saturation - Value
    hsv = mpl.colors.rgb_to_hsv(background_data_rgb)
    
    # to work in hls instead of hsv
    hs = hsv[:, :, 0].flatten()
    ls = vals.flatten()
    ss = hsv[:, :, 1].flatten()
    r = np.zeros(len(hues.flatten()))
    g = np.zeros(len(hues.flatten()))
    b = np.zeros(len(hues.flatten()))
    
    maxls = np.nanmax(ls)
    minls = np.nanmin(ls)
    
    # Translate HLS to RGB
    for i in xrange(len(hues.flatten())):
        r[i], g[i], b[i] = colorsys.hls_to_rgb(hs[i], ls[i], ss[i])
        
    r = r.reshape(lic_data.shape)
    g = g.reshape(lic_data.shape)
    b = b.reshape(lic_data.shape)
    
    rgb = np.zeros((y, x, 3), np.float_)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    
    return rgb