import os
import fnmatch
import cPickle as Pickle
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import spectral
from scipy import sparse

from data_structures import Observation
from io_functions import *
from photometry_functions import *
from astropy.stats import mad_std

from pyraf import iraf



def find_star_at(xpos,ypos,folder):
    ref = np.loadtxt(os.path.join(folder,'ref.mags'))
    dist2 = (xpos-ref[:,1])**2 + (ypos-ref[:,2])**2
    return np.where(dist2 == np.min(dist2))[0][0]


def readflux(folder,images_file='images',seeing_file='seeing',ZP=25.0):
    fnames = []
    datelist = []
    fwhmlist = []
    bgndlist = []
    siglist = []
    try:
        with open(os.path.join(folder,'lightcurve_data.pkl'),'rb') as pkl_file:
            dates, flux, eflux, mag, emag, fwhm, bgnd, signal = Pickle.load(pkl_file)
    except:
        with open(os.path.join(folder,images_file),'r') as fid:
            file = fid.read()
            for line in file.split('\n'):
                if len(line) > 0:
                    f, d = line.split()
                    fnames.append(f)
                    datelist.append(d)
        with open(os.path.join(folder,seeing_file),'r') as fid:
            file = fid.read()
            for i, line in enumerate(file.split('\n')):
                if len(line) > 0 and i < len(fnames):
                    n, f, b, s  = line.split()
                    if n == fnames[i]:
                        fwhmlist.append(f)
                        bgndlist.append(b)
                        siglist.append(s)
                    else:
                        print 'Mismatch between images and seeing files.'
                        sys.exit(0)
        dates = np.asarray(datelist,dtype=np.float64)
        fwhm = np.asarray(fwhmlist,dtype=np.float64)
        bgnd = np.asarray(bgndlist,dtype=np.float64)
        signal = np.asarray(siglist,dtype=np.float64)
        rf = np.loadtxt(os.path.join(folder,'ref.flux.calibrated'))
        nstars = rf.shape[0]
        nfiles = len(fnames)
        flux = np.zeros([nfiles,nstars])
        eflux = np.zeros([nfiles,nstars])
        mag = np.zeros([nfiles,nstars])
        emag = np.zeros([nfiles,nstars])
        for i, f in enumerate(fnames):
            fl = np.loadtxt(os.path.join(folder,f+'.flux'))
            flux[i,:] = fl[:,0]
            eflux[i,:] = fl[:,1]
            mag[i,:] = ZP - 2.5*np.log10(rf[:,0]+fl[:,0])
            emag[i,:] = mag[i,:] - ZP + 2.5*np.log10(rf[:,0]+fl[:,0]+fl[:,1])
        with open(os.path.join(folder,'lightcurve_data.pkl'),'wb') as pkl_file:
            Pickle.dump((dates, flux, eflux, mag, emag, fwhm, bgnd, signal),pkl_file)
    return dates, flux, eflux, mag, emag, fwhm, bgnd, signal


def index_good_data(mag,threshold=2.5,fraction=0.3):
    """Returns indices of good data epochs."""
    nepoch, nstars = mag.shape
    nmag = np.zeros_like(mag)
    mag = np.nan_to_num(mag)
    for i in range(nstars):
        nmag[:,i] = np.abs((mag[:,i]-np.median(mag[:,i])))/mad_std(mag[:,i])
    nmag[nmag <= threshold] = 0.0
    nmag[nmag > 0.001] = 1.0  
    ind = np.ones(nepoch,dtype=bool)
    nmag = np.nan_to_num(nmag)
    ftest = fraction*nstars
    for i in range(nepoch):
        if np.sum(nmag[i,:]) > ftest:
            ind[i] = False
    print np.sum(ind),'epochs of data out of',nepoch,'flagged as good'
    return ind
       
        
def plot_lightcurve_at(x,y,folder=None,params=None,error_limit=0.1,tmin=None,tmax=None):
    if not(folder):
        folder = params.loc_data
    star = find_star_at(x,y,folder)
    dates, flux, eflux, mag, emag, fwhm, bgnd, signal = readflux(folder)
    q = index_good_data(mag)
    q[np.isnan(emag[:,star])] = False
    q[emag[:,star]>error_limit] = False
    q[emag[:,star]>np.median(emag[q,star])+3*np.std(emag[q,star])] = False
    plt.figure()
    plt.errorbar(dates[q],mag[q,star],emag[q,star],fmt='.',ls='None',ms=2,c='k')
    if tmin and tmax:
        plt.xlim((tmin,tmax))
    plt.gca().invert_yaxis()
    plt.xlabel('HJD - 2450000',fontsize='14')
    plt.ylabel('Mag',fontsize='14')
    plt.title(folder+'-S%05d'%(star+1),fontsize='14')
    figname = folder+'-S%05d.png'%(star+1)
    plt.savefig(folder+'-S%05d.png'%(star+1))

    np.savetxt(folder+'-S%05d.mags'%(star+1),np.vstack((dates[q],mag[q,star],emag[q,star],flux[q,star],eflux[q,star],fwhm[q],bgnd[q],signal[q])).T,fmt='%10.5f  %7.4f  %7.4f  %12.3f  %12.3f %6.2f %9.2f %9.2f')
    np.savetxt(folder+'-S%05d.all'%(star+1),np.vstack((dates,mag[:,star],emag[:,star],flux[:,star],eflux[:,star],fwhm,bgnd,signal)).T,fmt='%10.5f  %7.4f  %7.4f  %12.3f  %12.3f %6.2f %9.2f %9.2f')

    return figname



def plotlc(files,ref,n):
    d = []
    flux = []
    dflux = []
    mag = []
    dmag = []
    for f in files:
        if isinstance(f.flux,np.ndarray):
            d.append(f.date)
            flux.append(f.flux[n-1])
            dflux.append(f.dflux[n-1])
            mag.append(30-2.5*np.log10(f.flux[n-1]+ref.flux[n-1]))
            dmag.append(f.dflux[n-1]*1.08574/(f.flux[n-1]+ref.flux[n-1]))
    plt.errorbar(d,mag,yerr=dmag,fmt='ro')
    ax=plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])




def trend_filter(y,n,iterations=1,star_number=None,mode='mean',period=None,dates=None,plotfile=None):

    def run_filter(s,templates,n,g,g1,g_inv_full,y,yt,iterations,debug=False,phase=None,n_phase_bins=25):
        if s in templates:
            p = np.where(templates == s)
            m = p[0][0]
            g1[:m,:m] = g[:m,:m]
            g1[m:,:m] = g[m+1:,:m]
            g1[:m,m:] = g[:m,m+1:]
            g1[m:,m:] = g[m+1:,m+1:]
            g_inv = np.linalg.inv(g1)
            temp = np.hstack((np.arange(m),np.arange(m+1,n)))
            ntemp = n-1
        else:
            g_inv = g_inv_full
            temp = np.arange(n)
            ntemp = n
        yy = y[s,:].copy()
        ya = yy*0.0
        for i in range(iterations):
            if isinstance(phase,np.ndarray):
                for p in range(n_phase_bins):
                    q = np.where((phase > p/(1.0*n_phase_bins)) & (phase <= (p+1)/(1.0*n_phase_bins)))
                    ya[q] = np.mean(yy[q])
            h = np.zeros(ntemp)
            yy = y[s,:].copy()
            for j in range(ntemp):
                h[j] = np.sum((yy-ya)*yt[temp[j],:])
            c = np.dot(g_inv,h)
            for j in range(ntemp):
                yy -= c[j]*yt[temp[j],:]
            if debug:
                print 'templates\\n',templates
                print 's\\n', s
                print 'c\\n', c
        return yy
    
    nstars = y.shape[0]
    templates = np.random.choice(nstars/5,n,replace=False)

    g = np.zeros([n,n])
    g1 = np.zeros([n-1,n-1])

    yc = y.copy()
    ymean = np.mean(yc,1).reshape(nstars,1)
    if star_number:
        if mode == 'minimum':
            ymean[star_number] = np.min(yc,1).reshape(nstars,1)[star_number]
        elif mode == 'maximum':
            ymean[star_number] = np.max(yc,1).reshape(nstars,1)[star_number]
        elif mode == 'median':
            ymean[star_number] = np.median(yc,1).reshape(nstars,1)[star_number]
    yc -= ymean

    yt = yc[templates,:].copy()

    for j in range(n):
        for k in range(j+1):
            g[j,k] = np.sum(yt[j,:]*yt[k,:])
            g[k,j] = g[j,k]
    g_inv_full = np.linalg.inv(g)

    if star_number:
        phase = None
        if period:
            cycle = dates/period
            phase = cycle - np.floor(cycle)
        yf = run_filter(star_number,templates,n,g,g1,g_inv_full,yc,yt,iterations,phase=phase,debug=False) + ymean[star_number]
        if plotfile:
            plt.figure()
            plt.plot(phase,yf,'b.',phase+1,yf,'b.')
            ax=plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.title(str(star_number)+'     P = '+"%6.3f"%period+' d')
            plt.xlabel('PHASE')
            plt.ylabel('Magnitude')
            plt.savefig(plotfile)
            plt.close()
    else:
        yf = y*0.0;
        for s in range(nstars):
            yf[s,:] = run_filter(s,templates,n,g,g1,g_inv_full,yc,yt,iterations)
        yf += ymean
    return yf


def detrend_v2(data,err_data,v1,v2):
    nstars, ndates = data.shape
    if len(v1) != ndates:
        print 'Error: length of vector 1 is not equal to the number of data rows'
        return data
    if len(v2) != ndates:
        print 'Error: length of vector 2 is not equal to the number of data rows'
        return data
    for star in range(nstars):
        bf = np.zeros([3,ndates])
        bf[0,:] = np.ones(ndates)
        bf[1,:] = v1
        bf[2,:] = v2
        alpha = np.zeros([3,3])
        beta = np.zeros(3)
        for k in range(3):
            for j in range(3):
                alpha[k,j] = np.sum(bf[j,:]*bf[k,:]/err_data[star,:]**2)
            beta[k] = np.sum(data[star,:]*bf[k,:]/err_data[star,:]**2)
        c = np.linalg.inv(alpha)
        a = np.dot(c,beta)
        data[star,:] -= a[0] + a[1]*v1 + a[2]*v2
    return data

    
def detrend_pca(data,err,npc=10):
    from sklearn.decomposition import PCA
    from scipy import linalg
    (nlc,npts) = data.shape
    p = np.where(err<1.e-4)
    if p[0].any():
        err[p] = 1.e-4
    mdata = data - np.mean(data, axis=1)[:,None]
    ddata = data
    print 'Computing principal components'
    eigen = PCA(n_components = npc).fit(mdata.T).transform(mdata.T)
    print 'Finished computing principal components'    
    for k in np.arange(nlc):
        A = eigen/err[k,:].reshape(npts,1)
        Cd = mdata[k,:]/err[k,:]
        try:
            M = linalg.inv(np.dot(A.T,A))
            w = np.dot(M,np.dot(A.T,Cd))
            ddata[k,:] -= np.dot(eigen,w.T).T
        except ValueError:
            pass
    return ddata



def lomb_search(dates,y,pmin,pmax,nfreq=1000,fap_threshold=1e-7,maxvars=100,plotfile='/dev/null'):
    nstars = y.shape[0]
    npoints = y.shape[1]
    logp = np.linspace(np.log10(pmin), np.log10(pmax), nfreq)
    freq = 2*np.pi/(10.0**logp)
    variables = []
    periods = []
    faps = []
    nvars = 0
    for star in range(nstars):
        if nvars < maxvars:
            qq = np.where(np.isfinite(y[star,:]))[0]
            if len(qq)>100:
#                print star
#                print dates[qq]
#                print y[star,qq]
                lnp = spectral.lombscargle(dates[qq], (y[star,qq]-np.mean(y[star,qq]))/np.std(y[star,qq]), freq)
                lnpmax = np.max(lnp)
                p = 2*np.pi/freq[np.where(lnp == lnpmax)[0][0]]
                #            fap = 1.0 - (1.0 - (1.0-2.0*lnpmax/npoints)**( (npoints-3)/2.0 ) )**nfreq
                fap = ((npoints-3)/2.0)*np.exp(-lnpmax)
                if fap < fap_threshold and not(np.abs(p-1.0) < 0.01) and not(np.abs(p-0.5) < 0.01) and not(np.abs(p-0.33333) < 0.01):
                    print star,p,np.max(lnp),fap
                    variables.append(star)
                    periods.append(p)
                    faps.append(fap)
                    plt.figure()
                    plt.subplot(3,1,1)
                    plt.plot(dates[qq],y[star,qq],'b.')
                    ymax = np.percentile(y[star,qq],99)
                    ymin = np.percentile(y[star,qq],1)
                    plt.ylim([ymin,ymax])
                    ax=plt.gca()
                    ax.set_ylim(ax.get_ylim()[::-1])
                    lfap = np.log10(fap)
                    plt.title(str(star)+'     P = '+"%6.3f"%p+' d      $log_{10}$ FAP = '+"%4.1f"%lfap)
                    plt.xlabel('Date')
                    plt.ylabel('Magnitude')
                    plt.subplot(3,1,2)
                    cycle = dates[qq]/p
                    phase = cycle - np.floor(cycle)
                    plt.plot(phase,y[star,qq],'b.',phase+1,y[star,qq],'b.')
                    plt.ylim([ymin,ymax])
                    ax=plt.gca()
                    ax.set_ylim(ax.get_ylim()[::-1])
                    plt.xlabel('PHASE')
                    plt.ylabel('Magnitude')
                    plt.subplot(3,1,3)
                    plt.plot(logp,lnp,'r-')
                    plt.xlabel('$log_{10}$ Period (d)')
                    plt.ylabel('LNP')
                    plt.savefig('var%(star)05d.png'%vars(),orientation='portrait',papertype='a4')
                    plt.close()
                    nvars += 1
    return variables, periods, faps

        
   
def transit_search(dates,data,err,pmin,pmax,nfreq,min_intransit_points=4,max_intransit_points=20,threshold=30,maxvars=100,plotfile='/dev/null',renormalize=True,true_period =None):
    nstars = data.shape[0]
    ndates = data.shape[1]
    xphase = np.linspace(0,1,1000)
    vars = []
    periods = []
    n_transit_points = max_intransit_points-min_intransit_points+1
    C = sparse.lil_matrix((n_transit_points*ndates,ndates))
    Nout = np.zeros(n_transit_points*ndates)
    for i in range(n_transit_points):
        for j in range(ndates):
            Nout[i*ndates+j] = ndates - i - min_intransit_points
            C[i*ndates+j,j:np.min([j+i+min_intransit_points,ndates])] = 1
            if j+i+min_intransit_points > ndates:
                C[i*ndates+j,:(j+i+min_intransit_points)%ndates] = 1
    renorm = 1.0
    RNout = 1.0/Nout
    S = C.tocsr()
    logp = np.linspace(np.log10(pmin), np.log10(pmax), nfreq)
    trial_periods = 10**logp
    nvars = 0
#    pdf = PdfPages(plotfile)
    for star in range(nstars):
        if nvars < maxvars:
            w = 1.0/err[star,:]**2
#            m = sum(w*data[star,:])/sum(w)
            m = np.median(data[star,:])
            npts = ndates
#            d = (data[star,:]-m)/err[star,:]
#            md = np.std(d)
#            p = np.where(d < 3*md)
#            if p[0].shape[0]:
#                npts = p[0].shape[0]
#                m = sum(w[p]*data[star,p].flatten())/sum(w[p])
#            else:
#                npts = ndates
            if renormalize:
                renorm = sum(w*(data[star,:].flatten()-m)**2)/(npts-1)
            t = np.zeros(nfreq)
            for i in range(nfreq):
                cycle = dates/trial_periods[i]
                phase = cycle - np.floor(cycle)
                ind = np.argsort(phase)
                d = data[star,ind]
                w = 1.0/(err[star,ind]**2)
                n = (d-m) * w
                sw = S.dot(w)
                sn = S.dot(n)
                t[i] = 0.5 * np.max(sn**2/sw) / renorm
            tmax = np.max(t)
            pw = np.where(t == tmax)
            if pw[0].shape[0] and (tmax > threshold):
                p = trial_periods[pw[0][0]]
                print star,p,tmax,np.std(t),np.mean(t),(tmax-np.mean(t))/np.std(t)
                vars.append(star)
                periods.append(p)
                plt.figure()
                plt.subplot(2,1,1)
                if true_period:
                    p = true_period
                cycle = dates/p
                phase = cycle - np.floor(cycle)
                plt.plot(phase,data[star,:],'b.',phase+1,data[star,:],'b.')
                ax=plt.gca()
                ax.set_ylim(ax.get_ylim()[::-1])
                plt.title(str(star)+'     P = '+"%6.3f"%p+' d')
                plt.xlabel('PHASE')
                plt.ylabel('Magnitude')
                plt.subplot(2,1,2)
                plt.plot(logp,t,'r-')
                plt.xlabel('$log_{10}$ Period (d)')
                plt.ylabel('T')
                plt.savefig('tran_'+str(star)+'.png')
#                pdf.savefig()
                plt.close()
                nvars += 1
#    pdf.close()
    return vars, periods



def insert_transit(dates,mags,period,phase_start,phase_duration,amplitude):
    cycle = dates/period
    phase = cycle - np.floor(cycle)
    p = np.where((phase > phase_start) & (phase < phase_start+phase_duration))
    new_mags = mags
    if p[0].shape[0]:
        new_mags[p] += amplitude
    return new_mags



def plotddp(k,dates,logp,sn,lnprob,mags,mi,eigen,ev):

    ph = np.linspace(0,2,500)
    cycle = dates/10**logp[mi]
    phase = cycle - np.floor(cycle)

    tt = np.linspace(dates[0],dates[-1:],5000)
    ttp = 2*np.pi*tt/10**logp[mi];
    
    plt.subplot(5,1,1)
    plt.plot(logp,np.exp(lnprob-np.max(lnprob)),'r-')
    plt.title(str(k)+'    P = '+str(10**logp[mi])+'    S/N = '+str(sn))
    plt.ylabel('Prob(P)')
    plt.xlabel('log P')

    plt.subplot(5,1,2)
    q1 = np.where(abs(mags-np.mean(mags))<3*np.std(mags))[0]
    npts = q1.shape[0]
    colour1 = np.arange(npts)/np.float(npts)
    plt.scatter(phase[q1],mags[q1],s=2,edgecolor='',c=colour1,cmap=mpl.cm.winter)
    plt.scatter(phase[q1]+1,mags[q1],s=2,edgecolor='',c=colour1,cmap=mpl.cm.winter)
    print ph.shape
    ddd = np.mean(mags)+ev[mi,-2]*np.sin(2*np.pi*ph)+ev[mi,-1]*np.cos(2*np.pi*ph)
    plt.plot(ph,np.mean(mags)+ev[mi,-2]*np.sin(2*np.pi*ph)+ev[mi,-1]*np.cos(2*np.pi*ph),'r-')
    plt.ylabel('Mag')
    plt.xlabel('Phase')
    ax=plt.gca()
    ax.set_xlim([0,2])
    ymax = np.percentile(mags[q1],99)
    ymin = np.percentile(mags[q1],1)
    plt.ylim([ymin,ymax])
    ax.set_ylim(ax.get_ylim()[::-1])
    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    plt.subplot(5,1,3)
    dd = mags-np.dot(eigen[:,:-2],ev[mi,:-2].T).T
#    dd = d.reshape(d.shape[1],)
    q2 = np.where(abs(dd-np.mean(dd))<3*np.std(dd))[0]
    npts = q2.shape[0]
    colour2 = np.arange(npts)/np.float(npts)
    plt.scatter(phase[q2],dd[q2],s=2,edgecolor='',c=colour2,cmap=mpl.cm.winter)
    plt.scatter(phase[q2]+1,dd[q2],s=2,edgecolor='',c=colour2,cmap=mpl.cm.winter)
    plt.plot(ph,np.mean(mags)+ev[mi,-2]*np.sin(2*np.pi*ph)+ev[mi,-1]*np.cos(2*np.pi*ph),'r-')
    plt.ylabel('Detrended Mag')
    plt.xlabel('Phase')
    ax=plt.gca()
    ax.set_xlim([0,2])
    ymaxd = np.percentile(mags[q1],99)
    ymind = np.percentile(mags[q1],1)
    plt.ylim([ymind,ymaxd])
    ax.set_ylim(ax.get_ylim()[::-1])
    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    plt.subplot(5,1,4)
    plt.scatter(dates[q1],mags[q1],s=2,edgecolor='',c=colour1,cmap=mpl.cm.winter)
    plt.plot(tt,np.mean(mags)+ev[mi,-2]*np.sin(ttp)+ev[mi,-1]*np.cos(ttp),'r-')
    plt.ylabel('Mag')
    plt.xlabel('Date')
    ax=plt.gca()
    plt.ylim([ymin,ymax])
    ax.set_ylim(ax.get_ylim()[::-1])
    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    plt.subplot(5,1,5)
    plt.scatter(dates[q2],dd[q2],s=2,edgecolor='',c=colour2,cmap=mpl.cm.winter)
    plt.plot(tt,np.mean(mags)+ev[mi,-2]*np.sin(ttp)+ev[mi,-1]*np.cos(ttp),'r-')
    plt.ylabel('Detrended Mag')
    plt.xlabel('Date')
    ax=plt.gca()
    plt.ylim([ymind,ymaxd])
    ax.set_ylim(ax.get_ylim()[::-1])
    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)



def ddp_search(dates,data,err,pmin,pmax,nfreq,npc=10,nlc=None,BIC_threshold=7,recompute_eigen=True):
    from sklearn.decomposition import PCA
    from scipy import linalg
    if nlc:
        npts = data.shape[1]
    else:
        nlc, npts = data.shape
    print 'npts =',npts
    p = np.where(err<1.e-4)
    if p[0].any():
        err[p] = 1.e-4
    mdata = data - np.mean(data, axis=1)[:,None]
    ndata = mdata / np.std(mdata, axis=1)[:,None]
    logp = np.linspace(np.log10(pmin), np.log10(pmax), nfreq)
    p = 10.0**logp
    q = np.where(np.abs(p-1.0)>0.02)[0]
    logp = logp[q]
    freq = 1/(10.0**logp)
    nfreq = freq.shape[0]
    amp = np.zeros(nfreq)
    phase = np.zeros(nfreq)
    err_amp = np.zeros(nfreq)
    lnprob = np.zeros(nfreq)
    ev = np.zeros((nfreq,npc+2))
    print 'Computing principal components'
    eigen = PCA(n_components = npc).fit(mdata.T).transform(mdata.T)
    print 'Finished computing principal components'    
    for k in np.arange(nlc):
        if recompute_eigen:
            odata = np.vstack((mdata[:k,:],mdata[k+1:,:]))
            eigen = PCA(n_components = npc).fit(odata.T).transform(odata.T)
        eigen_e = eigen/err[k,:].reshape(npts,1)
        C = np.diag(err[k,:]**2)
        Cd = mdata[k,:]/err[k,:]
        for i,f in enumerate(freq):
            sb = np.sin(2*np.pi*f*dates)/err[k,:]
            cb = np.cos(2*np.pi*f*dates)/err[k,:]
#            A = np.hstack((eigen,sb.reshape(npts,1),cb.reshape(npts,1)))/err[k,:].reshape(npts,1)
            A = np.hstack((eigen_e,sb.reshape(npts,1),cb.reshape(npts,1)))

            try:
                M = linalg.inv(np.dot(A.T,A))
                w = np.dot(M,np.dot(A.T,Cd))
                chi2 = np.sum((Cd-np.dot(A,w))**2)
                W0 = M[:-2,:-2]
                U0 = M[-2:,-2:]
                V = M[:-2,-2:]
                U = U0 - np.dot(V.T,linalg.solve(W0,V))
                lnprob[i] = np.log(2*np.pi)-0.5*linalg.det(W0)-0.5*np.dot(w[-2:].T,np.dot(U,w[-2:]))-0.5*chi2
#                vs = U[0,0]
#                vc = U[1,1]
#                vsc =U[0,1]
#                cs = w[-2]
#                cc = w[-1]
#                amp2 = cs**2+cc**2
#                amp[i] = np.sqrt(amp2)
#                phase[i] = np.arctan2(cc,cs)
#                err_amp[i] = np.sqrt(cs**2*vs/amp2+cc**2*vc/amp2+2*cs*cc*vsc/amp2)
                ev[i,:] = w
            except ValueError:
                amp[i] = 0
                err_amp[i] = 1
                phase[i] = 0
                lnprob[i] = -1e6
                break
        p = np.where(lnprob <= -1e5)
        q = np.where(lnprob > -1e5)
        if p[0].any() and q[0].any():
            lnprob[p] = np.min(lnprob[q])
        mb = np.max(lnprob)
        p = np.arange(nfreq)
        for i in range(5):
            p = np.where(abs(lnprob[p]-np.mean(lnprob[p]))<3*np.std(lnprob[p]))
        sn = (mb-np.mean(lnprob[p]))/np.std(lnprob[p])
        print 'star',k,'  lnProb =',mb,'  S/N =',sn
        if sn > BIC_threshold:
            p = np.where(lnprob == mb)
            print 'Detected period',10.0**logp[p],'for star',k,'with lnProb',mb
#            print 'Coefficients',w
            plt.figure(figsize=(8.27, 11.69), dpi=100)
            plotddp(k,dates,logp,sn,lnprob,data[k,:],p[0][0],A*err[k,:].reshape(npts,1),ev)
            plt.savefig('vard%(k)05d.pdf'%vars(),orientation='portrait',papertype='a4')
            plt.close()



def slot_acf(t,x,x_err,delta_tau,n):
    """ Compute the slot autocorrelation function for timeseries x(t) at lags
    k*delta_tau for k in 0..n"""
    
    nx = len(x)

    # Renormalise uncertainties
    x_var = x_err**2
    xmean = np.sum(x/x_var)/np.sum(1.0/x_var)
    x -= xmean
    chi2 = np.sum(x**2/x_var)
    scale = np.sqrt(chi2/nx)
    print 'scale:',scale
    x_err *= scale
    x_err_inv = 1.0/x_err
    x_norm = x*x_err_inv
    
    Ti, Tj = np.meshgrid(t,t)
    Tau = Tj - Ti

    Xi, Xj = np.meshgrid(x_norm,x_norm)
    Xj = np.tril(Xj,k=-1)

    Xi_err_inv, Xj_err_inv = np.meshgrid(x_err_inv,x_err_inv)
    Xj_err_inv = np.tril(Xj_err_inv,k=-1)

    acf = np.zeros(n)

    Taud = Tau/delta_tau
    
    for k in range(n):
        b = np.zeros((nx,nx),dtype=bool)
        b[np.abs(Taud-k)<0.5] = 1
        b = np.tril(b,k=-1)
        Xjb = Xj*b
        Xj_err_inv_b = Xj_err_inv*b
        #acf[k] = np.sum(np.diag(np.dot(Xi.T,Xj*b)))/np.sum(b)
        #acf[k] = np.sum([np.dot(x[m]*np.ones(nx),Xjb[:,m]) for m in range(nx)]) / np.sum([np.dot(x_err_inv[m]*np.ones(nx),Xj_err_inv_b[:,m]) for m in range(nx)])
        acf[k] = np.sum([np.dot(x_norm[m]*np.ones(nx),Xjb[:,m]) for m in range(nx)]) / np.sum(b)

    return acf

   
def gaussian_kernel_acf(t,x,x_err,delta_tau,n):
    """ Compute the autocorrelation function for timeseries x(t) at lags
    k*delta_tau for k in 0..n using a gaussian kernel."""
    
    for i in range(5):
        x = (x-np.median(x))/mad_std(x)
        q = np.where(np.abs(x)<3.5)
        x = x[q]
        t = t[q]
        
    nx = len(x)

    x = (x-np.mean(x))/np.std(x)
    
    Ti, Tj = np.meshgrid(t,t)
    Tau = Tj - Ti
    
    Xi, Xj = np.meshgrid(x,x)
    Xj = np.tril(Xj,k=-1)

    acf = np.zeros(n)

    for k in range(n):
        h = k*delta_tau
        b = np.exp(-Tau**2/(2*h**2))/np.sqrt(2*np.pi*h)
        b = np.tril(b,k=-1)
        Xjb = Xj*b
        acf[k] = np.sum([np.dot(x[m]*np.ones(nx),Xjb[:,m]) for m in range(nx)])/np.sum(b)

    return acf

