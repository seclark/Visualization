from __future__ import division
import numpy as np
import cPickle
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pylab
import time
import copy
import math
import os.path
import cPickle as pickle
from matplotlib import rc
from astropy.io import fits
import colorsys
import matplotlib as mpl
import matplotlib.colors as colors

def maketestdata(local=True):
    if local is True:
        path = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/QUmaps/"
        nhidata_fn = "/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits"
    else:
        path = "/disks/jansky/a/users/goldston/susan/Wide_maps/QUmaps/"
        nhidata_fn = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits"
    
    nhidata = fits.getdata(nhidata_fn)

    vstep = 0.736122839600 # CDELT3 of original Wide cube
    vstart = -35.4 # km/s vel of channel 0974
    vels = np.zeros(21, np.float_)

    all_ggs = np.zeros((nhidata.shape[0], nhidata.shape[1], 21), np.float_)
    mom1cube = np.zeros((nhidata.shape[0], nhidata.shape[1], 21), np.float_)
    #all_ggs = np.zeros((2432, 2432, 21), np.float_)
    #mom1cube = np.zeros((2432, 2432, 21), np.float_)
    all_ggs_big = copy.copy(all_ggs)
    mom1cube_big = copy.copy(mom1cube)
    
    for i in xrange(21):
        vels[i] = vstart + vstep*5*i

        numstart = 974 + i*5
        numend = numstart + 4
        if numstart < 1000:
            numstartstr = "0"
        else:
            numstartstr = ""
        if numend < 1000:
            numendstr = "0"
        else:
            numendstr = ""
    
        gg = fits.getdata(path + "intrht_GALFA_HI_allsky_chS"+numstartstr+str(numstart)+"_"+numendstr+str(numend)+"_w75_s15_t70.fits")
    
        all_ggs[:, :, i] = gg#[:, 0:2432]
        mom1cube[:, :, i] = all_ggs[:, :, i]*vels[i]
        
        #all_ggs_big_slice = copy.copy(all_ggs[:, :, i])
        #all_ggs_big_slice[np.where(all_ggs_big[:, :, i] < 1)] = None
        #all_ggs_big[:, :, i] = all_ggs_big_slice
        #mom1cube_big[:, :, i] = all_ggs_big[:, :, i]*vels[i]
    
        # testing.
        #nhidata = nhidata[:, 0:2432]
        nhidata = nhidata/np.nanmax(nhidata)
    
    print(vels)    
    return all_ggs, mom1cube, nhidata#, all_ggs_big, mom1cube_big

def renormalize(data, minval, maxval):

    newdata = (maxval - minval) * ((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))) + minval
    
    return newdata   
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
def test_chans(data):
    (nx, ny, nz) = data.shape
    
    cmap = 'spectral'
    datargba = np.zeros((data.shape[0], data.shape[1], 4), np.float_)
    
    zstep = 1.0/nz
    for _z in xrange(3):
        small_cmap = truncate_colormap(cmap, zstep*_z, zstep*(_z+1))

        smallcmap = small_cmap(data[:, :, _z])
        for _slice in xrange(4):
            small_slice = copy.copy(smallcmap[:, :, _slice])
            small_slice[np.where(data[:, :, _z] <= np.nanmedian(data[:, :, _z]))] = 0.0
            smallcmap[:, :, _slice] = small_slice
        
        print(np.nanmin(smallcmap[:, :, 0:3]), np.nanmax(smallcmap[:, :, 0:3]), np.nanmedian(smallcmap[:, :, 0:3]))
        datargba += smallcmap

    return datargba
    
def testblend(data):
    
    cmap = 'spectral'
    (nx, ny, nz) = data.shape
    zstep = 1.0/nz
    _z = 10
    small_cmap1 = truncate_colormap(cmap, zstep*_z, zstep*(_z+1))
    small_slice1 = small_cmap1(data[:, :, _z])
    small_slice1[:, :, 3] = renormalize(data[:, :, _z], 0.0, 1.0)
    
    _z = 18
    small_cmap2 = truncate_colormap(cmap, zstep*_z, zstep*(_z+1))
    small_slice2 = small_cmap2(data[:, :, _z])
    small_slice2[:, :, 3] = renormalize(data[:, :, _z], 0.0, 1.0)
    
    return small_slice2, small_slice1

def blend(slice1, slice2):
    outA = slice1[:, :, 3] + slice2[:, :, 3]*(1 - slice1[:, :, 3])
    outRGBA = np.zeros((slice1.shape[0], slice1.shape[1], 4), np.float_)
    
    for _rgb in xrange(3):
        outR = (slice1[:, :, _rgb]*slice1[:, :, 3] + slice2[:, :, _rgb]*slice2[:, :, 3]*(1 - slice1[:, :, 3])) / outA
        outR[np.where(outA == 0)] = 0
        outRGBA[:, :, _rgb] = outR
    
    outRGBA[:, :, 3] = outA
    
    return outA, outRGBA
    
def blendall(data):
    (nx, ny, nz) = data.shape
    
    cmap = 'spectral'
    datargba = np.zeros((data.shape[0], data.shape[1], 4), np.float_)
    
    blendme = np.zeros(datargba.shape)
    
    zstep = 1.0/nz
    for _z in xrange(nz):
        small_cmap = truncate_colormap(cmap, zstep*_z, zstep*(_z+1))
        small_slice = small_cmap(data[:, :, _z])
        small_slice[:, :, 3] = renormalize(data[:, :, _z], 0.0, 1.0)

        outA, blendme = blend(small_slice, blendme)
        
    return blendme

def rgb_to_hsv(rgbdata):
    # Map to Hue - Saturation - Value
    hsvdata = mpl.colors.rgb_to_hsv(rgbdata)
    
    return hsvdata
    
def hls_to_rgb(hlsdata):
    hs = hlsdata[:, :, 0].flatten()
    ss = hlsdata[:, :, 1].flatten()
    ls = hlsdata[:, :, 2].flatten()
    r = np.zeros(len(hs))
    g = np.zeros(len(hs))
    b = np.zeros(len(hs))

    # Translate HLS to RGB
    for i in xrange(len(hs)):
        r[i], g[i], b[i] = colorsys.hls_to_rgb(hs[i], ls[i], ss[i])
    
    rgbdata = np.zeros(hlsdata.shape, np.float_)
    rgbdata[:, :, 0] = r.reshape(hlsdata[:, :, 0].shape)
    rgbdata[:, :, 1] = g.reshape(hlsdata[:, :, 0].shape)
    rgbdata[:, :, 2] = b.reshape(hlsdata[:, :, 0].shape)
    
    return rgbdata

if False == True:    
    mom1 = np.sum(mom1cube, axis=2)#mom1 = gg1*984 + gg2*1054
    intrht = np.sum(all_ggs, axis=2)#gg1 + gg2

    hsv = np.zeros((mom1.shape[0], mom1.shape[1], 3), np.float_)
    hsv[:, :, 0] = mom1/np.nanmax(mom1)
    hsv[:, :, 1] = np.log10(intrht)/np.nanmax(np.log10(intrht))
    hsv[:, :, 2] = np.log10(intrht)/np.nanmax(np.log10(intrht))

    hs = hsv[:, :, 0].flatten()
    ss = hsv[:, :, 1].flatten()
    ls = hsv[:, :, 2].flatten()
    r = np.zeros(len(hs))
    g = np.zeros(len(hs))
    b = np.zeros(len(hs))

    maxls = np.nanmax(ls)
    minls = np.nanmin(ls)

    # Translate HLS to RGB
    for i in xrange(len(hs)):
        r[i], g[i], b[i] = colorsys.hls_to_rgb(hs[i], ls[i], ss[i])
    
    rgb = np.zeros(hsv.shape, np.float_)
    rgb[:, :, 0] = r.reshape(mom1.shape)
    rgb[:, :, 1] = g.reshape(mom1.shape)
    rgb[:, :, 2] = b.reshape(mom1.shape)


# try making a movie thing
def makemovie():
    bkground = fits.getdata('/Volumes/DataDavy/GALFA/DR2/FullSkyWide/GALFA_HI_W_S1024_V0000.4kms.fits')
    bkground = bkground[:, 0:2432]
    bkground = bkground/np.nanmax(bkground)
    #bkground = np.log10(bkground)/np.nanmax(np.log10(bkground))


    for t in xrange(5):#(len(vels)-2):
        slicergb = np.zeros((2432, 2432, 3), np.float_)
    
        gaussweightedR = np.zeros(slicergb[:, :, 0].shape, np.float_)
        for (weight, _ti) in zip([t - 1, t, t + 1], [0.1, 1.0, 0.1]):
            try:
                gaussweightedR += all_ggs[:, :, _ti]*weight
            except:
                gaussweightedR += 0.0
            
        gaussweightedG = np.zeros(slicergb[:, :, 0].shape, np.float_)
        for (weight, _ti) in zip([t, t + 1, t + 2], [0.1, 1.0, 0.1]):
            try:
                gaussweightedG += all_ggs[:, :, _ti]*weight
            except:
                gaussweightedG += 0.0
        
        gaussweightedB = np.zeros(slicergb[:, :, 0].shape, np.float_)
        for (weight, _ti) in zip([t + 1, t + 2, t + 3], [0.1, 1.0, 0.1]):
            try:
                gaussweightedB += all_ggs[:, :, _ti]*weight
            except:
                gaussweightedB += 0.0      
            
        slicergb[:, :, 0] = gaussweightedR/np.nanmax(gaussweightedR)
        slicergb[:, :, 1] = gaussweightedG/np.nanmax(gaussweightedG)
        slicergb[:, :, 2] = gaussweightedB/np.nanmax(gaussweightedB)
    
    
        slicergb[:, :, 0] = all_ggs[:, :, t]/np.nanmax(all_ggs[:, :, t]) + bkground
        slicergb[:, :, 1] = all_ggs[:, :, t+1]/np.nanmax(all_ggs[:, :, t+1]) + bkground
        slicergb[:, :, 2] = all_ggs[:, :, t+2]/np.nanmax(all_ggs[:, :, t+2]) + bkground
    
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(slicergb)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("slicergb_test_bkgrnd_{}.png".format(t), dpi=100)
    

if __name__ == "__main__":
    all_ggs, mom1cube, nhidata = maketestdata(local = False)
    blended_data = blendall(all_ggs)
    
    np.save("allsky_rgba_blended_test1.npy", blended_data)

