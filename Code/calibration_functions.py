import sys
import os
import matplotlib
import fnmatch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.ndimage.filters import maximum_filter
from skimage.feature import peak_local_max


def locate_intercept(x,y,x_range):
    print 'locating offset'
    ix = np.arange(len(x))
    p = np.where(x[ix]>2)[0]
    ix = ix[p]
    p = np.where(np.logical_and(x[ix] < np.percentile(x[ix],x_range[1]),x[ix] > np.percentile(x[ix],x_range[0])))[0]
    ix = ix[p]
    for n in range(10):
        a = np.nanmedian(y[ix]-x[ix])
        print a, x.shape
        p = np.where(np.abs(y[ix]-a-x[ix])<2.5*np.nanstd(y[ix]-a-x[ix]))[0]
        ix = ix[p]
    return a, ix


def calibrate(dir,plotfile='calibration.png',magnitude_range_fraction=(0.1,8),sky_flux_cutoff_percent=0.1):

    magfile = os.path.join(dir,'ref.mags')
    fluxfile = os.path.join(dir,'ref.flux')

    mag = np.loadtxt(magfile)
    flux = np.loadtxt(fluxfile)

    p = np.where((mag[:,3] > 0) & (flux[:,0] > 0))[0]

    sky_max_flux = np.percentile(flux[p,0],sky_flux_cutoff_percent)
    q = np.where(flux[p,0] < sky_max_flux)[0]
    sky_flux = 0.9*np.mean(flux[p[q],0])
    flux[:,0] -= sky_flux

    x = np.linspace(np.min(mag[p,3]),np.max(mag[p,3]),3)
    offset, stars = locate_intercept(mag[p,3],25-2.5*np.log10(flux[p,0]),magnitude_range_fraction)

    # Axes definitions
    nullfmt = plt.NullFormatter()
    rect_scatter = [0.1, 0.1, 0.7, 0.7]
    rect_histx = [0.1, 0.8, 0.7, 0.1]
    rect_histy = [0.8, 0.1, 0.1, 0.7]

    binsize = 0.5
    bandwidth = 0.25

    fig = plt.figure()
    ax = fig.add_subplot(223, position=rect_scatter)
    ax.scatter(mag[p,3],25-2.5*np.log10(flux[p,0]),s=5,color='k')
    ax.scatter(mag[p[stars],3],25-2.5*np.log10(flux[p[stars],0]),s=5,color='c')
    ax.plot(x,x,'r--')
    ax.plot(x,x+offset,'r',label=r'$\Delta mag = %4.2f$'%offset)
    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xlabel('DAOPhot mag',fontsize='14')
    ax.set_ylabel('25-2.5*log(pyDIA flux)',fontsize='14')
    xmin, xmax  = plt.xlim()
    ymin, ymax  = plt.ylim()
    xx = np.linspace(xmin,xmax,1000)
    
    ax = fig.add_subplot(221, position=rect_histx)
    hval, bins, _ = ax.hist(mag[p,3],range=(xmin,xmax),bins=int((xmax-xmin)/binsize+1),
                            normed=True,alpha=0.3)
    kde_skl = KernelDensity(kernel='epanechnikov',bandwidth=bandwidth)
    sample = mag[p,3]
    kde_skl.fit(sample[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xx[:, np.newaxis])
    ax.plot(xx,np.exp(log_pdf),'r')
    ax.xaxis.set_major_formatter(nullfmt)
    ax.yaxis.set_major_formatter(nullfmt)
    ax.set_title(dir)

    ax = fig.add_subplot(221, position=rect_histy)
    ax.hist(25-2.5*np.log10(flux[p,0]),range=(ymin,ymax),bins=int((ymax-ymin)/binsize+1),
            orientation='horizontal',normed=True,alpha=0.3)
    ax.xaxis.set_major_formatter(nullfmt)
    ax.yaxis.set_major_formatter(nullfmt)

    mag[:,3] += offset
    if mag.shape[1] == 4:
        np.savetxt(os.path.join(dir,'ref.mags.calibrated'),mag,fmt='%5d  %8.3f  %8.3f  %7.4f')
    else:
        np.savetxt(os.path.join(dir,'ref.mags.calibrated'),mag,fmt='%5d  %8.3f  %8.3f  %7.4f  %7.4f  %7.3f  %7.3f')

    cflux = 10.0**(0.4*(25-mag[:,3]))
    if mag.shape[1] == 4:
        cfluxerr = 0.0*cflux
    else:
        cfluxerr = cflux - 10.0**(0.4*(25-mag[:,3]-mag[:,4]))
    np.savetxt(os.path.join(dir,'ref.flux.calibrated'),np.vstack((cflux,cfluxerr)).T,fmt='%12.4f  %12.4f')


    plt.savefig(os.path.join(dir,plotfile))



def makeCMD(dirI,dirV,bandwidth = 0.25):
    ifile = os.path.join(dirI,'ref.mags.calibrated')
    vfile = os.path.join(dirV,'ref.mags.calibrated')
    im = np.loadtxt(ifile)
    vm = np.loadtxt(vfile)
    p = np.where((im[:,3] > 0) & (vm[:,3] > 0))[0]
    
    plt.figure()
    plt.scatter(vm[p,3]-im[p,3],im[p,3],s=5)
    plt.grid()
    plt.gca().invert_yaxis()
    plt.xlabel(r'$(v-i)_{p}$',fontsize='14')
    plt.ylabel(r'$i_p$',fontsize='14')
    plt.title(dirI+' '+dirV)
    xmin, xmax  = plt.xlim()
    ymax, ymin  = plt.ylim()
    plt.savefig(dirI+'-'+dirV+'-CMD.png')
    plt.close()

    print xmin, xmax, ymin, ymax
    
    np.savetxt(dirI+'-'+dirV+'-CMDdata',np.vstack((vm[p,3],im[p,3],vm[p,4],im[p,4])).T,
               fmt='%7.4f   %7.4f   %7.4f   %7.4f')
    
    plt.figure()
    samples = np.vstack([vm[p,3]-im[p,3],im[p,3]]).T
    kde_skl = KernelDensity(kernel='gaussian',bandwidth=bandwidth)
    kde_skl.fit(samples)
    # score_samples() returns the log-likelihood of the samples
    xvi = np.linspace(xmin,xmax,40*(xmax-xmin)+1)
    xi = np.linspace(ymin,ymax,40*(ymax-ymin)+1)
    Y, X = np.meshgrid(xvi, xi[::-1])
    xy =  np.vstack([Y.ravel(), X.ravel()]).T
    Z = np.exp(kde_skl.score_samples(xy))
    Z = Z.reshape(X.shape)
    levels = np.linspace(0, Z.max(), 25)
    plt.contourf(Y, X, Z, levels=levels, cmap=plt.cm.Reds)
    Zmax = np.max(np.max(Z))
    mx = maximum_filter(Z,size=20)
    lm = (Z == mx) * (Z > 0.1*Zmax) * (Z < 0.99*Zmax)
    local_maxima = np.nonzero(lm)
    print Z[local_maxima]/Zmax
    print 'Red clump detected at',Y[local_maxima],X[local_maxima]
    plt.scatter(Y[local_maxima],X[local_maxima],color='c',marker='+',s=40,label='Red Clump estimated at (%6.3f,%6.3f)'%(Y[local_maxima],X[local_maxima]))

    plt.grid()
    plt.gca().invert_yaxis()
    plt.legend(loc='upper left')
    plt.xlabel(r'$(v-i)_{p}$',fontsize='14')
    plt.ylabel(r'$i_p$',fontsize='14')
    plt.title(dirI+' '+dirV)
    plt.xlim((xmin,xmax))
    plt.ylim((ymax,ymin))
    plt.savefig(dirI+'-'+dirV+'-CMD-density.png')
    plt.close()

