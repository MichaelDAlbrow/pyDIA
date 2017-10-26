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
from scipy.odr import ODR, Model, RealData


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

    #sky_max_flux = np.percentile(flux[p,0],sky_flux_cutoff_percent)
    #q = np.where(flux[p,0] < sky_max_flux)[0]
    #sky_flux = 0.9*np.mean(flux[p[q],0])
    
    #q = np.where(mag[p,3] > np.percentile(mag[p,3],99.9))[0]
    #sky_flux = 0.95*np.median(flux[p[q],0])
    #flux[:,0] -= sky_flux

    x = np.linspace(np.min(mag[p,3]),np.max(mag[p,3]),3)
    offset, stars = locate_intercept(mag[p,3],25-2.5*np.log10(flux[p,0]),magnitude_range_fraction)

    # Axes definitions
    nullfmt = plt.NullFormatter()
    rect_scatter = [0.15, 0.15, 0.7, 0.7]
    rect_histx = [0.15, 0.8, 0.7, 0.1]
    rect_histy = [0.8, 0.15, 0.1, 0.7]

    binsize = 0.5
    bandwidth = 0.25

    fig = plt.figure()
    ax1 = fig.add_subplot(223, position=rect_scatter)
    ax1.scatter(mag[p,3],25-2.5*np.log10(flux[p,0])-mag[p,3],s=5,color='k')
    ax1.scatter(mag[p[stars],3],25-2.5*np.log10(flux[p[stars],0])-mag[p[stars],3],s=5,color='c')
    #ax.plot(x,x,'r--')
    ax1.plot(x,x*0.0+offset,'r',label=r'$\Delta mag = %4.2f$'%offset)
    ax1.grid()
    ax1.legend(loc='upper left')
    ax1.set_xlabel('DAOPhot mag',fontsize='14')
    ax1.set_ylabel('25-2.5*log(pyDIA flux)-(DAOPhot mag)',fontsize='14')
    xmin, xmax  = plt.xlim()
    ymin, ymax  = plt.ylim()
    xx = np.linspace(xmin,xmax,1000)
    
    ax2 = fig.add_subplot(221, position=rect_histx)
    hval, bins, _ = ax2.hist(mag[p,3],range=(xmin,xmax),bins=int((xmax-xmin)/binsize+1),
                            normed=True,alpha=0.3)
    kde_skl = KernelDensity(kernel='epanechnikov',bandwidth=bandwidth)
    sample = mag[p,3]
    kde_skl.fit(sample[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xx[:, np.newaxis])
    ax2.plot(xx,np.exp(log_pdf),'r')
    ax2.xaxis.set_major_formatter(nullfmt)
    ax2.yaxis.set_major_formatter(nullfmt)
    ax2.set_title(dir)

    ax3 = fig.add_subplot(221, position=rect_histy)
    ax3.hist(25-2.5*np.log10(flux[p,0])-mag[p,3],range=(ymin,ymax),bins=int((ymax-ymin)/binsize+1),
            orientation='horizontal',normed=True,alpha=0.3)
    ax3.xaxis.set_major_formatter(nullfmt)
    ax3.yaxis.set_major_formatter(nullfmt)

    ax1.set_ylim((offset-1,offset+1))
    ax3.set_ylim((offset-1,offset+1))
    plt.savefig(os.path.join(dir,plotfile))

    ax1.set_ylim((offset-0.1,offset+0.1))
    ax3.set_ylim((offset-0.1,offset+0.1))
    plt.savefig(os.path.join(dir,'zoom-'+plotfile))


    mag[:,3] += offset
    if mag.shape[1] == 4:
        np.savetxt(os.path.join(dir,'ref.mags.calibrated'),mag,fmt='%5d  %8.3f  %8.3f  %7.4f')
    else:
        np.savetxt(os.path.join(dir,'ref.mags.calibrated'),mag,fmt='%5d  %8.3f  %8.3f  %7.4f  %7.4f  %7.3f  %7.3f %7.3f')

    cflux = 10.0**(0.4*(25-mag[:,3]))
    if mag.shape[1] == 4:
        cfluxerr = 0.0*cflux
    else:
        cfluxerr = cflux - 10.0**(0.4*(25-mag[:,3]-mag[:,4]))
    np.savetxt(os.path.join(dir,'ref.flux.calibrated'),np.vstack((cflux,cfluxerr)).T,fmt='%12.4f  %12.4f')



def makeCMD(dirI,dirV,bandwidth = 0.25,ifile=None,vfile=None,plot_density=True,IV=None,RC=None,source_colour=None):

    if ifile is None:
        ifile = os.path.join(dirI,'ref.mags.calibrated')

    if vfile is None:
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

    if RC is not None:
        plt.scatter(RC[0],RC[1],color='r',marker='+',s=40,label='Red Clump (%6.3f,%6.3f)'%RC)

    if IV is not None:
        plt.scatter(IV[1]-IV[0],IV[0],color='b',marker='.',s=40,label='Source (%6.3f,%6.3f)'%(IV[1]-IV[0],IV[0]))

    if source_colour is not None:
        plt.scatter(np.nan,np.nan,color='w',marker='.',s=40,label='Deblended source (V-I) = %6.3f'%source_colour)

    plt.legend(loc='upper left')

    xmin, xmax  = plt.xlim()
    ymax, ymin  = plt.ylim()
    plt.savefig(dirI+'-'+dirV+'-CMD.png')
    plt.close()

    print xmin, xmax, ymin, ymax
    
    np.savetxt(dirI+'-'+dirV+'-CMDdata',
                np.vstack((im[p,0],im[p,1],im[p,2],vm[p,3],vm[p,4],im[p,3],im[p,4])).T,
                fmt='%6d %9.3f %9.3f %7.4f   %7.4f   %7.4f   %7.4f',
                header='ID  xpos  ypos  V  V_err  I  I_err')
    
    red_clump = None

    if plot_density:

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
        lm = (Z == mx) * (Z > 0.01*Zmax) * (Z < 0.99*Zmax)
        if np.sum(lm) > 0:
            local_maxima = np.nonzero(lm)[0]
            red_clump = (float(Y[local_maxima][0,0]),float(X[local_maxima][0,0]))
            print Z[local_maxima]/Zmax
            print 'Red clump detected at',red_clump
            plt.scatter(red_clump[0],red_clump[1],color='c',marker='+',s=40,label='Red Clump estimated at (%6.3f,%6.3f)'%red_clump)
        else:
            print 'Error detecting red clump'
            red_clump = None

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

    return red_clump


def source_colour(ifile,vfile,interval=0.05,plotfile='source_colour.png'):

    Idata = np.loadtxt(ifile)
    Vdata = np.loadtxt(vfile)
    qI = np.where(Idata[:,5] < 1.0)
    qV = np.where(Vdata[:,5] < 1.0)

    Idata = Idata[qI]
    Vdata = Vdata[qV]

    interval = 0.05
    start = np.floor(np.min(Idata[:,0]))
    end = np.ceil(np.max(Idata[:,0]))

    time = np.arange(start,end,interval)

    flux1 = np.zeros_like(time) + np.nan
    flux2 = np.zeros_like(time) + np.nan
    flux1_err = np.zeros_like(time) + np.nan
    flux2_err = np.zeros_like(time) + np.nan

    for i in range(len(time)):
        q = np.where(np.abs(Idata[:,0] - time[i]) < interval/2.0)[0]
        if q.any():
            flux1[i] =  np.sum(Idata[q,1]/Idata[q,2]**2) / np.sum(1.0/Idata[q,2]**2)
            flux1_err[i] =  np.sqrt(1.0 / np.sum(1.0/Idata[q,2]**2))

            p = np.where(np.abs(Vdata[:,0] - time[i]) < interval/2.0)[0]
            if p.any():
                flux2[i] =  np.sum(Vdata[p,1]/Vdata[p,2]**2) / np.sum(1.0/Vdata[p,2]**2)
                flux2_err[i] =  np.sqrt(1.0 / np.sum(1.0/Vdata[p,2]**2))

    plt.figure()
    plt.errorbar(flux1,flux2,xerr=flux1_err,yerr=flux2_err,fmt='.')
    plt.xlabel(r'$\delta F_I$')
    plt.ylabel(r'$\delta F_V$')
    plt.grid()

    # Define a function (quadratic in our case) to fit the data with.
    def linear_func1(p, x):
        m, c = p
        return m*x + c

    # Create a model for fitting.
    linear_model = Model(linear_func1)

    good_data = np.where(np.logical_and(np.isfinite(flux1),np.isfinite(flux2)))[0]
    offset =  np.mean(flux2[good_data]-flux1[good_data])

    # Create a RealData object using our initiated data from above.
    data = RealData(flux1[good_data], flux2[good_data], sx=flux1_err[good_data], sy=flux2_err[good_data])

    # Set up ODR with the model and data.
    odr = ODR(data, linear_model, beta0=[1.0, offset])

    # Run the regression.
    out = odr.run()

    # Use the in-built pprint method to give us results.
    out.pprint()

    x1, x2 = plt.gca().get_xlim()
    x_fit = np.linspace(x1,x2, 1000)
    y_fit = linear_func1(out.beta, x_fit)

    plt.plot(x_fit,y_fit,'r-',label=r"$\delta F_V = %5.3f \delta F_I + %5.3f$"%(out.beta[0],out.beta[1]))
    colour = -2.5*np.log10(out.beta[0])

    delta_colour = -2.5*np.log10(out.beta[0]-out.sd_beta[0]) - colour

    plt.title(r'$(V-I)_S = %8.3f \pm %8.3f$'%(colour,delta_colour))
    plt.legend()

    plt.savefig(plotfile)

    return colour, delta_colour

def plot_lightcurve(file, columns=(0,3,4),plotfile='lightcurve.png'):

    data = np.loadtxt(file)
 
    plt.figure(figsize=(8,5))
    plt.errorbar(data[:,columns[0]],data[:,columns[1]],data[:,columns[2]],fmt='.')
    plt.xlabel(r'$HJD - 2450000$')
    plt.ylabel(r'$Magnitude$')
    plt.savefig(plotfile)

