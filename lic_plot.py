def lic_plot(lic_data, background_data = None, F_m = 0.4, F_M = 0.37):

    """
    lic_data        :: output of LIC code
    background_data :: background color map, e.g. density or vector magnitude
    F_m             :: contrast enhancement parameter - see below
    F_M             :: contrast enhancement parameter - see below
    
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
    
    sats = np.ones(lic_rht.shape)
    licrhtmax = np.nanmax(lic_rht)
    licrhtmin = np.nanmin(lic_rht)
    #vals = lic_rht
    #vals = (lic_rht - 0.35*licrhtmax)/(licrhtmax - 0.35*licrhtmax)
    
    #sats = lic_rht/np.nanmax(lic_rht)
    
    # Contrast Enhancement from http://www.paraview.org/Wiki/ParaView/Line_Integral_Convolution#Image_LIC_CE_stages
    # L_ij = (L_ij - m) / (M - m)
    # L = HSL lightness. m = lightness to map to 0. M = lightness to map to 1.
    # m = min(L) + F_m * (max(L) - min(L))
    # M = max(L) - F_M * (max(L) - min(L))
    # F_m and F_M take values between 0 and 1. Increase F_m -> darker colors. Increase F_M -> brighter colors.
    
    # F_m = 0.4 and F_M = 0.37 are optimal for LIC kernel = 101
    F_m = 0.4#0.35 # Darkness
    F_M = 0.37#0.35 # Lightness
    #F_m = 0.3
    #F_M = 0.27
    
    m = licrhtmin + F_m * (licrhtmax - licrhtmin)
    M = licrhtmax - F_M * (licrhtmax - licrhtmin)
    vals = (lic_rht - m) / (M - m)
    
    huesp = tau353 / np.nanmax(tau353)
    #huesp = np.log10(tau353)/np.nanmax(np.log10(tau353))
    satsp = np.ones(lic_planck.shape)
    licplanckmax = np.nanmax(lic_planck)
    
    valsp = (lic_planck - m) / (M - m)
    
    y, x = hues.shape
    
    nhispan = np.nanmax(nhi) - np.nanmin(nhi)
    tauspan = np.nanmax(tau353) - np.nanmin(tau353)
    
    nhi_scaled = (nhi - np.nanmin(nhi))/np.float(nhispan)
    #nhi_scaled = -1 + (nhi_scaled * 2)
    nhi_scaled = -1 + (nhi_scaled * 2)
    
    tau_scaled = (tau353 - np.nanmin(tau353))/np.float(tauspan)
    tau_scaled = -1 + (tau_scaled * 2)
    
    cmap1 = mpl.colors.ListedColormap(sns.color_palette("YlOrRd", 256))
    cmap2 = mpl.colors.ListedColormap(sns.color_palette("YlGnBu", 256))
    
    tau353rbg = cmap1(tau_scaled)
    nhirbg = cmap2(nhi_scaled)
        
    # Map onto existing RGBA colormap
    #nhirbg = cm.RdYlBu_r(nhi_scaled)
    #tau353rgb = cm.RdYlBu_r(tau_scaled)
    
    # Only need RGB, not RGBA
    nhirbg = nhirbg[:, :, 0:3]
    tau353rbg = tau353rbg[:, :, 0:3]
    
    hsv = matplotlib.colors.rgb_to_hsv(nhirbg)
    hsvplanck = matplotlib.colors.rgb_to_hsv(tau353rbg)
    
    # to work in hls instead of hsv
    hs = hsv[:, :, 0].flatten()
    ls = vals.flatten()
    ss = hsv[:, :, 1].flatten()
    r = np.zeros(len(hues.flatten()))
    g = np.zeros(len(hues.flatten()))
    b = np.zeros(len(hues.flatten()))
    
    hsp = hsvplanck[:, :, 0].flatten()
    lsp = valsp.flatten()
    ssp = hsvplanck[:, :, 1].flatten()
    rp = np.zeros(len(huesp.flatten()))
    gp = np.zeros(len(huesp.flatten()))
    bp = np.zeros(len(huesp.flatten()))
    
    maxls = np.nanmax(ls)
    minls = np.nanmin(ls)
    maxlsp = np.nanmax(lsp)
    minlsp = np.nanmin(lsp)
    
    lsint = copy.copy(ls)
    lspint = copy.copy(lsp)
    
    for i in xrange(len(hues.flatten())):
        r[i],g[i],b[i] = colorsys.hls_to_rgb(hs[i], ls[i], ss[i])
        rp[i],gp[i],bp[i] = colorsys.hls_to_rgb(hsp[i], lsp[i], ssp[i])
        
        # Glitch avoidance
        if (r[i] > 1.0) or (g[i] > 1.0) or (b[i] > 1.0) or (r[i] < 0.0) or (g[i] < 0.0) or (b[i] < 0.0):
            #r[i],g[i],b[i] = colorsys.hls_to_rgb(hs[i], 0.5, ss[i])
            lsint[i] = None

        if (rp[i] > 1.0) or (gp[i] > 1.0) or (bp[i] > 1.0) or (rp[i] < 0.0) or (gp[i] < 0.0) or (bp[i] < 0.0):
            #rp[i],gp[i],bp[i] = colorsys.hls_to_rgb(hsp[i], 0.5, ssp[i])
            lspint[i] = None
        
    nans, xnan = nan_helper(lsint)
    lsint[nans]= np.interp(xnan(nans), xnan(~nans), lsint[~nans])
    nans, xnan = nan_helper(lspint)
    lspint[nans]= np.interp(xnan(nans), xnan(~nans), lspint[~nans])
        
    for i in xrange(len(hues.flatten())):
        r[i],g[i],b[i] = colorsys.hls_to_rgb(hs[i], lsint[i], ss[i])
        rp[i],gp[i],bp[i] = colorsys.hls_to_rgb(hsp[i], lspint[i], ssp[i])
        
    r = r.reshape(nhi.shape)
    g = g.reshape(nhi.shape)
    b = b.reshape(nhi.shape)
    
    rp = rp.reshape(nhi.shape)
    gp = gp.reshape(nhi.shape)
    bp = bp.reshape(nhi.shape)
    
    print y, x
    rgb = np.zeros((y, x, 3), np.float_)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    
    rgbp = np.zeros((y, x, 3), np.float_)
    rgbp[:, :, 0] = rp
    rgbp[:, :, 1] = gp
    rgbp[:, :, 2] = bp
    
    wlen = 35
    smr = 15
    rgb[:, 0:(wlen+smr)] = 0
    rgbp[:, 0:(wlen+smr)] = 0

    rgb[:, -(wlen+smr):] = 0
    rgbp[:, -(wlen+smr):] = 0

    rgb[0:wlen+smr, :] = 0
    rgbp[0:wlen+smr, :] = 0

    rgb[1150-(wlen+smr):1150, :] = 0
    rgbp[1150-(wlen+smr):1150, :] = 0
    
    # Mask Planck point source map
    if pointsource == True:
        pointsources = fits.getdata("/Volumes/DataDavy/Planck/Planck_HFI_Point_Source_Mask.fits")
        rgbp[pointsources.T <= 0] = 0
        rgb[pointsources.T <= 0] = 0
    
    # Plotting    
    if highlatonly == True:
        fig = plt.figure(figsize = (15, 10), facecolor = "white")
    else:    
        fig = plt.figure(figsize = (15, 9), facecolor = "white")
        
    ax1 = fig.add_subplot(212)
    ax2 = fig.add_subplot(211)
    
    # Plot colors to get colorbar correct
    im1plot = ax1.imshow(tau353, cmap = cmap1)
    im2plot = ax2.imshow(nhi, cmap = cmap2)
    
    # Plot LIC
    im1 = ax1.imshow(rgbp)
    im2 = ax2.imshow(rgb)
    axlist = [ax1, ax2]
    
    if plotstars == True:
        starnums = np.load("SC_241_star_index.npy")

        for i, s in enumerate(starnums):
            star = Star(s, region = "SC_241")
            if (star.sig != 0) & (star.pol != 0) & (star.dist != 0):

                p1, p2 = lineSeg(star.x, star.y, star.imang, linelen = 15)
                for ax in axlist:
                    ax.plot((p1[0], p2[0]), (p1[1], p2[1]), color = "white", lw=2)

    ax1.set_ylim(0, y)
    ax2.set_ylim(0, y)
    ax1.set_xlim(0, x)
    ax2.set_xlim(0, x)
    
    divider = make_axes_locatable(ax1)
    if highlatonly == True:
        ax1.text(0.0, 1.17, r"$\theta_{353}:$ $\mathrm{Magnetic}$ $\mathrm{field}$ $\mathrm{orientation}$ $\mathrm{from}$ $Planck$", ha = "left", va = "center", size=20, transform=ax1.transAxes)
    else:
        ax1.text(0.0, 1.2, r"$\theta_{353}:$ $\mathrm{Magnetic}$ $\mathrm{field}$ $\mathrm{orientation}$ $\mathrm{from}$ $Planck$", ha = "left", va = "center", size=20, transform=ax1.transAxes)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im1plot, cax = cax, ticks=[-8, -7, -6, -5, -4, -3])
    cbar.ax.set_yticklabels([r"$10^{-8}$", r"$10^{-7}$", r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"], size = 15)
    cbar.ax.set_ylabel(r"$\tau_{353}$", size = 20, rotation = 0)
    cbar.ax.get_yaxis().labelpad = 25
    cbar.solids.set_edgecolor("face")
    
    ytickps = []
    ytickls = []
    xtickps = []
    xtickls = []
    
    if highlatonly == True:
        decs = [22, 28, 34]
        ramin, ramax, decmin, decmax = getcorners(region = "SC_241")
    
        for i in xrange(len(decs)):
            x, y = radec_to_xy_GALFA(ramin, decs[i], region = "SC_241")
            ytickps = np.append(ytickps, y)
    
        ytickls = [r'$22^o$', r'$28^o$', r'$34^o$']
    
    else:
        decs = [20, 26, 32, 38]
        ramin, ramax, decmin, decmax = getcorners(region = "SC_241")
    
        for i in xrange(len(decs)):
            x, y = radec_to_xy_GALFA(ramin, decs[i], region = "SC_241")
            ytickps = np.append(ytickps, y)
    
        ytickls = [r'$20^o$', r'$26^o$', r'$32^o$', r'$38^o$']
        
    ax1.set_yticks(ytickps)
    ax1.set_yticklabels(ytickls, size=18)
    ax1.set_ylabel(r"$\mathrm{DEC}$", size = 18, labelpad = 15)
    
    ax1.tick_params(axis='x', which='both', bottom='off', top = "off")
    ax1.tick_params(axis='y', which='both', left='off', right = "off")
    
    ax2.set_yticks(ytickps)
    ax2.set_yticklabels(ytickls, size=18)
    ax2.set_ylabel(r"$\mathrm{DEC}$", size = 18, labelpad = 15)
    
    # RA from 194.99169665 to 288.32521665
    ras = [200, 220, 240, 260, 280]
    for i in xrange(len(ras)):
        x, y = radec_to_xy_GALFA(ras[i], decmin, region = "SC_241")
        xtickps = np.append(xtickps, x)
    
    xtickls = [r'$200^o$', r'$220^o$', r'$240^o$', r'$260^o$', r'$280^o$']
    ax2.set_xticks(xtickps)
    ax2.set_xticklabels(xtickls, size=18)
    ax2.set_xlabel(r"$\mathrm{RA}$", size = 18, labelpad = 15)#, horizontalalignment = "left", transform = ax2.transAxes)
    
    plt.setp(ax1.get_xticklabels(), visible=False)

    divider = make_axes_locatable(ax2)
    if highlatonly == True:
        ax2.text(0.0, 1.17, r"$\theta_{RHT}:$ $\mathrm{Magnetic}$ $\mathrm{field}$ $\mathrm{orientation}$ $\mathrm{from}$ $\mathrm{GALFA-H{\sc I}}$", ha = "left", va = "center", size=20, transform=ax2.transAxes)
    else:
        ax2.text(0.0, 1.2, r"$\theta_{RHT}:$ $\mathrm{Magnetic}$ $\mathrm{field}$ $\mathrm{orientation}$ $\mathrm{from}$ $\mathrm{GALFA-H{\sc I}}$", ha = "left", va = "center", size=20, transform=ax2.transAxes)
    
    ax2.tick_params(axis='x', which='both', bottom='off', top = "off")
    ax2.tick_params(axis='y', which='both', left='off', right = "off")
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im2plot, cax = cax, ticks=[18, 19, 20, 21])
    cbar.ax.set_yticklabels([r"$10^{18}$", r"$10^{19}$", r"$10^{20}$", r"$10^{21}$"], size = 15)
    cbar.ax.set_ylabel(r"$N_{HI}$", size = 20, rotation = 0)
    cbar.ax.get_yaxis().labelpad = 25
    cbar.solids.set_edgecolor("face")
    
    # Only show a small section!
    if highlatonly == True:
        ax1.set_xlim(1500, 5600-(wlen+smr))
        ax2.set_xlim(1500, 5600-(wlen+smr))
        ax1.set_ylim(wlen+smr, 1150-(wlen+smr))
        ax2.set_ylim(wlen+smr, 1150-(wlen+smr))
    
    
    # Add l, b lines
    lbcolor = "0.8"
    
    y = 1150
    xs, ys = add_b_lines(10, region = "SC_241")
    topy1 = ys[xs == np.min(np.abs(xs))]
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
    
    xs, ys = add_b_lines(30, region = "SC_241")
    jj = np.abs(ys - y)
    topx2 = xs[jj == np.min(jj)]
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
    
    xs, ys = add_b_lines(50, region = "SC_241")
    jj = np.abs(ys - y)
    topx3 = xs[jj == np.min(jj)]
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
    
    xs, ys = add_b_lines(70, region = "SC_241")
    jj = np.abs(ys - y)
    topx4 = xs[jj == np.min(jj)]
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
        
    xs, ys = add_l_lines(80, region = "SC_241")
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
        
    xs, ys = add_l_lines(50, region = "SC_241")
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
        
    xs, ys = add_l_lines(20, region = "SC_241")
    for ax in axlist:
        ax.plot(xs, ys, color = lbcolor)
        
    #ax1.text(5307, 947, r"$l = 80^o$", size=13, rotation=-38, color = "0.9", ha="center", va="center")
    
    #ax1.text(5097, 703, r"$l = 50^o$", size=13, rotation=-10, color = "0.9", ha="center", va="center")
         
    #ax1.text(5106, 410, r"$l = 20^o$", size=13, rotation=17, color = "0.9", ha="center", va="center")
    
    y = 1150
    #ax1.text(-145, topy1, r"$10^o$", size = 18, ha = "center", va = "center")
    for ax in axlist:
        if highlatonly == False:
            ax.text(topx2, y + 45, r"$30^o$", size = 18, ha = "center", va = "bottom")
            ax.text(topx3, y + 45, r"$50^o$", size = 18, ha = "center", va = "bottom")
            ax.text(topx4, y + 45, r"$70^o$", size = 18, ha = "center", va = "bottom")
            ax.text(1.0, 1.20, r"$\mathrm{Galactic}$ $\mathrm{latitude}$", ha = "right", va = "top", size=20, transform=ax.transAxes)
    
        else:
            ax.text(topx3, y - 20, r"$50^o$", size = 18, ha = "center", va = "bottom")
            ax.text(topx4, y - 20, r"$70^o$", size = 18, ha = "center", va = "bottom")
            ax.text(1.0, 1.12, r"$\mathrm{Galactic}$ $\mathrm{latitude}$", ha = "right", va = "top", size=20, transform=ax.transAxes)
            # 1.0, 1.12    
    
    return rgb, rgbp