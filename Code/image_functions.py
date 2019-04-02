import os
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve, convolve2d
from astropy.modeling import models, fitting


def positional_shift(R,T):
    Rc = R[10:-10,10:-10]
    Tc = T[10:-10,10:-10]
    c = fftconvolve(Rc, Tc[::-1, ::-1])
    cind = np.where(c == np.max(c))
    print cind
    csmall = c[cind[0][0]-10:cind[0][0]+10,cind[1][0]-10:cind[1][0]+10]
    csmall = c[cind[0][0]-6:cind[0][0]+7,cind[1][0]-6:cind[1][0]+7]
    X,Y = np.indices(csmall.shape)
    total =csmall.sum()
    dx = (X*csmall).sum()/total - 6 + cind[0][0] - c.shape[0]/2.0
    dy = (Y*csmall).sum()/total - 6 + cind[1][0] - c.shape[1]/2.0
    return dx, dy





def register(R,T,params):
    #Tc = T.data*T.mask
    #Rc = R.data*R.mask
    Tc_data = T.data.copy()
    Tc_data[T.mask==0] = np.median(Tc_data[T.mask==1])
    Rc_data = R.data.copy()
    Rc_data[R.mask==0] = np.median(Rc_data[R.mask==1])
    if isinstance(params.fwhm_section,np.ndarray):
        w = params.fwhm_section
        Tc = Tc_data[w[2]:w[3],w[0]:w[1]].copy()
        Rc = Rc_data[w[2]:w[3],w[0]:w[1]].copy()
    else:
        Tc = Tc_data
        Rc = Rc_data
    #nx, ny = R.shape
    #z = np.arange(-3,4)
    #saturated_pixels = np.where(R.mask==0)
    #for k in range(saturated_pixels[0].size):
    #    p = np.array([z+saturated_pixels[0][k],z+saturated_pixels[1][k]])
    #    px, py = np.meshgrid(p[0],p[1])
    #    q = np.where((px>=0) & (px<R.data.shape[0]) & (py>=0) & (py<R.data.shape[1]))
    #    Rc[saturated_pixels[0][k],saturated_pixels[1][k]]= np.median(R.data[px[q],py[q]])
    #saturated_pixels = np.where(T.mask==0)
    #for k in range(saturated_pixels[0].size):
    #    p = np.array([z+saturated_pixels[0][k],z+saturated_pixels[1][k]])
    #    px, py = np.meshgrid(p[0],p[1])
    #    q = np.where((px>=0) & (px<R.data.shape[0]) & (py>=0) & (py<R.data.shape[1]))
    #    Tc[saturated_pixels[0][k],saturated_pixels[1][k]]= np.median(T.data[px[q],py[q]])
    Rcm = Rc - np.median(Rc)
    Tcm = Tc - np.median(Tc)
    c = fftconvolve(Rcm, Tcm[::-1, ::-1])
    print c.shape
    kernel = np.ones((3,3))
    c = convolve2d(c,kernel,mode='same')
    print c.shape
    cind = np.where(c == np.max(c))
    print np.max(c)
    print cind
    print Rc.shape
    try:
        xshift = cind[0][0]-Rc.shape[0]+1
    except IndexError:
        print 'Warning:',T.fullname, 'failed to register.'
        return None, None, None
    yshift = cind[1][0]-Rc.shape[1]+1
    imint = max(0,-xshift)
    imaxt = min(R.shape[0],R.shape[0]-xshift)
    jmint = max(0,-yshift)
    jmaxt = min(R.shape[1],R.shape[1]-yshift)
    iminr = max(0,xshift)
    imaxr = min(R.shape[0],R.shape[0]+xshift)
    jminr = max(0,yshift)
    jmaxr = min(R.shape[1],R.shape[1]+yshift)
    RT = np.zeros(R.shape)
    RT[iminr:imaxr,jminr:jmaxr] = T.data[imint:imaxt,jmint:jmaxt]
    mask = np.ones(R.shape,dtype=bool)
    mask[iminr:imaxr,jminr:jmaxr] = 0
    inv_variance = 1.0/(RT/params.gain +
                        (params.readnoise/params.gain)**2) + mask*1.0
    RM = np.zeros(R.shape,dtype=bool)
    RM[iminr:imaxr,jminr:jmaxr] = T.mask[imint:imaxt,jmint:jmaxt]
    return RT, RM, inv_variance


def compute_bleed_mask(d,radius,params):
    print 'Computing bleed mask'
    kernel = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [2,2,2,2,2,2,2,2,2,2],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
    rad2 = radius*radius
    mask = np.ones_like(d,dtype=bool)
    dc = convolve2d(d,kernel.T,mode='same')
    rad = int(np.ceil(radius))
    z = np.arange(2*rad+1)-rad
    x,y = np.meshgrid(z,z)
    p = np.array(np.where(x**2 + y**2 < rad2))
    bad_pixels = np.where(np.abs(dc)>1.1*params.pixel_max)
    zp0 = z[p[0]]
    zp1 = z[p[1]]
    sp0 = bad_pixels[0][:,np.newaxis]
    sp1 = bad_pixels[1][:,np.newaxis]
    q0 = zp0 + sp0
    q1 = zp1 + sp1
    q0 = q0.flatten()
    q1 = q1.flatten()
    s = np.asarray(np.where((q0>=0) & (q0<d.shape[0]) &
                            (q1>=0) & (q1<d.shape[1])))[0]
    mask[q0[s],q1[s]] = 0
    for i in range(mask.shape[1]):
        if np.sum(mask[:,i]) < 0.85*mask.shape[0]:
            mask[:,i] = 0
    return mask


def compute_bleed_mask2(d,params):

    mask = np.ones_like(d,dtype=bool)
    if (params.bleed_mask_multiplier_above == 0) and (params.bleed_mask_multiplier_below == 0):
        return mask
    for kernel_len in [2,3,5,7,10,15,23,34,51,77]:
            lkernel = np.vstack((np.zeros(kernel_len),np.ones(kernel_len),np.zeros(kernel_len)))
            lkernel /= np.sum(lkernel)
            dl = convolve2d(d,lkernel.T,mode='same')
            sy, sx  = np.where(dl>params.pixel_max)
            for q in range(len(sx)):
                ymin = max(0,int(sy[q]-params.bleed_mask_multiplier_below*kernel_len))
                ymax = min(d.shape[1],int(sy[q]+params.bleed_mask_multiplier_above*kernel_len))
                mask[ymin:ymax,sx[q]] = 0
    return mask


def compute_saturated_pixel_mask(im,params):
    radius = params.mask_radius
    rad2 = radius*radius
    rad = int(np.ceil(radius))
    z = np.arange(2*rad+1)-rad
    x,y = np.meshgrid(z,z)
    p = np.array(np.where(x**2 + y**2 < rad2))
    mask = np.ones(im.shape,dtype=bool)
    saturated_pixels = np.where((im > params.pixel_max) |
                                (im <= params.pixel_min))
    zp0 = z[p[0]]
    zp1 = z[p[1]]
    sp0 = saturated_pixels[0][:,np.newaxis]
    sp1 = saturated_pixels[1][:,np.newaxis]
    q0 = zp0 + sp0
    q1 = zp1 + sp1
    q0 = q0.flatten()
    q1 = q1.flatten()
#    q = np.array([[],[]])
#    for k in range(saturated_pixels[0].size):
#        q = np.column_stack([q,np.array([zp0+saturated_pixels[0][k],
#                                         zp1+saturated_pixels[1][k]])])
#        q1.append([r for r in zp0+saturated_pixels[0][k]])
#        q2.append([r for r in zp1+saturated_pixels[1][k]])
#    q = np.array([np.array(q1).flatten(),np.array(q2).flatten()])
    s = np.asarray(np.where((q0>=0) & (q0<im.shape[0]) &
                            (q1>=0) & (q1<im.shape[1])))[0]
    mask[q0[s],q1[s]] = 0            
    return mask

def compute_saturated_pixel_mask_2(im1,im2,radius,params):
    rad2 = radius*radius
    rad = int(np.ceil(radius))
    z = np.arange(2*rad+1)-rad
    x,y = np.meshgrid(z,z)
    p = np.array(np.where(x**2 + y**2 < rad2))
    mask = np.ones(im1.shape,dtype=bool)
    saturated_pixels = np.where((im1 > params.pixel_max) |
                                (im1 <= params.pixel_min) |
                                (im2 > params.pixel_max) |
                                (im2 <= params.pixel_min))
    for k in range(saturated_pixels[0].size):
        q = np.array([z[p[0]]+saturated_pixels[0][k],z[p[1]]+saturated_pixels[1][k]])
        s = np.asarray(np.where((q[0]>=0) & (q[0]<im1.shape[0]) &
                                (q[1]>=0) & (q[1]<im1.shape[1])))[0]
        mask[q[0,s],q[1,s]] = 0            
    return mask


def compute_kernel_saturation_mask(image,params):
    cimage = convolve2d(image,params.pixel_saturation_kernel,mode='same')
    rad2 = params.mask_radius**2
    rad = int(np.ceil(params.mask_radius))
    z = np.arange(2*rad+1)-rad
    x,y = np.meshgrid(z,z)
    p = np.array(np.where(x**2 + y**2 < rad2))
    mask = np.ones(image.shape,dtype=bool)
    saturated_pixels = np.where(cimage > params.pixel_saturation_kernel_max)
    for k in range(saturated_pixels[0].size):
        q = np.array([z[p[0]]+saturated_pixels[0][k],z[p[1]]+saturated_pixels[1][k]])
        s = np.asarray(np.where((q[0]>=0) & (q[0]<image.shape[0]) &
                                (q[1]>=0) & (q[1]<image.shape[1])))[0]
        mask[q[0,s],q[1,s]] = 0            
    return mask


def cosmic_ray_clean(data,params):
    import cosmics
    c = cosmics.cosmicsimage(data, gain=params.gain,
                             readnoise=params.readnoise, sigclip=20,
                             sigfrac=0.6, objlim=10)
    c.run(maxiter = 3)
    return c.cleanarray



def kappa_clip(mask,norm,threshold):
    not_finished = True
    bmask = np.ones(norm.shape,dtype=bool)
    count = 0
    while not_finished and count < 10:
        nm = bmask*mask*norm
        p = np.where(np.abs(nm)>0.0001)
        sp = np.std(norm[p])
        t = np.where(np.abs(norm) > threshold*sp)
        if t:
            print 'Rejecting',t[0].shape[0],'pixels'
            bmask[t] = 0
            count += 1
        else:
            not_finished = False
    return bmask



def boxcar_blur(im):
    d = np.zeros(im.shape)
    m1 = im.shape[0] - 2
    m2 = im.shape[1] - 2
    for i in range(3):
        for j in range(3):
            d[1:m1+1,1:m2+1] += im[i:i+m1,j:j+m2]
    d /= 9.0
    return d


def convolve_undersample(im):
    from scipy.ndimage.filters import convolve
    x = np.arange(3)-1
    xx,yy = np.meshgrid(x,x)
    kernel = 0.25*(np.ones([3,3])-abs(xx*0.5))*(np.ones([3,3])-abs(yy*0.5))
    c = convolve(im,kernel)
    return c


def convolve_gauss(im,fwhm):
    from scipy.ndimage.filters import convolve
    sigma = fwhm/(2*np.sqrt(2*np.log(2.0)))
    nk = 1 + 2*int(4*sigma)
    x = np.arange(nk)-nk/2
    xx,yy = np.meshgrid(x,x)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel /= np.sum(kernel)
    c = convolve(im,kernel)
    return c


def convolve_disk(im,radius):
    from scipy.ndimage.filters import convolve
    radius = int(radius)
    diameter = 2*radius + 1
    x = np.arange(diameter)-radius
    xx,yy = np.meshgrid(x,x)
    kernel = np.zeros((diameter,diameter))
    kernel[xx**2+yy**2<=radius**2] = 1.0
    kernel /= np.sum(kernel)
    fp_im = im*1.0
    c = convolve(fp_im,kernel)
    return c


def apply_photometric_scale(d,c,pdeg):
    p = np.zeros(d.shape)
    (m,n) = d.shape
    eta = (range(n)-0.5*(n-1)*np.ones(n))/(n-1)
    xi = (range(m)-0.5*(m-1)*np.ones(m))/(m-1)
    x,y = np.meshgrid(eta,xi)
    i = 0
    for l in range(pdeg+1):
        for m in range(pdeg-l+1):
            t = (x**l)*(y**m)
            p += c[i]*t
            i += 1
    q = d/p
    return q


def undo_photometric_scale(d,c,pdeg,size=None,position=(0,0)):
    md,nd = d.shape
    if size:
        (m,n) = size
    else:
        (m,n) = d.shape
    p = np.zeros([md,nd])
    eta = (range(n)-0.5*(n-1)*np.ones(n))/(n-1)
    xi = (range(m)-0.5*(m-1)*np.ones(m))/(m-1)
    x0,y0 = np.meshgrid(eta,xi)
    x = x0[position[0]:position[0]+md,position[1]:position[1]+nd]
    y = y0[position[0]:position[0]+md,position[1]:position[1]+nd]
    i = 0
    for l in range(pdeg+1):
        for m in range(pdeg-l+1):
            t = (x**l)*(y**m)
            p += c[i]*t
            i += 1
    q = d*p
    return q



def compute_fwhm(f,params,width=20,seeing_file='seeing',image_name=False):


  
    g_width = None

    if image_name:
        fname = f
    else:
        fname = f.name

    if os.path.exists(params.loc_output+os.path.sep+seeing_file):
        for line in open(params.loc_output+os.path.sep+seeing_file,'r'):
            sline = line.split()
            if sline[0] == fname:
                g_width = float(sline[1])
                g_roundness = float(sline[2])
                bgnd = float(sline[3])
                signal = float(sline[4])
                break
    
    if g_width is None:
        if isinstance(params.fwhm_section,np.ndarray):
            w = params.fwhm_section
            image = f.data[w[2]:w[3],w[0]:w[1]].copy()
            mask = f.mask[w[2]:w[3],w[0]:w[1]].copy()
        else:
            image = f.data.copy()
            mask = f.mask.copy()
        print image.shape
        print mask.shape
        bgnd = np.percentile(image[mask==1],30)
        image[mask==0] = bgnd
        image -= bgnd
        signal = image.sum()/image.size
        c = fftconvolve(image, image[::-1, ::-1])
        xcen = c.shape[0]/2
        ycen = c.shape[1]/2
        c_small = c[xcen-20:xcen+20,ycen-20:ycen+20]
        c_small -= np.min(c_small)
        xsize, ysize = c_small.shape
        xcen = c_small.shape[0]/2
        ycen = c_small.shape[1]/2
        y, x = np.mgrid[:xsize, :ysize]
        g_init = models.Gaussian2D(amplitude=c_small[xcen,ycen],x_stddev=1,y_stddev=1,x_mean=xcen,y_mean=ycen)
        fit_g = fitting.LevMarLSQFitter()
        g=fit_g(g_init,x,y,c_small)
        gx =  g.x_stddev.value
        gy =  g.y_stddev.value
        g_width = np.mean((gx,gy))/np.sqrt(2.0)
        g_roundness = np.max((gx,gy))/np.min((gx,gy))
    #x1 = int(round(c.shape[0]*0.5))
        #x2 = int(round(c.shape[0]*0.5+width))
        #y1 = int(round(c.shape[1]*0.5))
        #xx = np.arange(x2-x1+1)
        #xnew = np.linspace(0,x2-x1,1000)
        #fint = interp1d(xx,c[x1:x2+1,y1]-np.min(c[x1:x2+1,y1]),kind='cubic')
        #ynew = fint(xnew)
        #ymax = max(ynew)
        #for i,y in enumerate(ynew):
        #    if y<ymax/2:
        #        fw = i*(xnew[1]-xnew[0])
        #        break
        #if not(fw):
        #    fw = 6.0
        p = open(seeing_file,'a')
        p.write(f.name+'  '+str(g_width)+'  '+str(g_roundness)+'  '+str(bgnd)+'  '+str(signal)+'\n')
        p.close()
        
    return g_width, g_roundness, bgnd, signal


def subtract_sky(image,params):
    from scipy.linalg import lu_solve, lu_factor, LinAlgError
    print 'subtracting sky'
    if params.sky_subtract_mode == 'percent':
        image2 = image.copy()
        if params.pixel_min > 0.1:
            p = np.where(image2 > params.pixel_min)
            const = np.percentile(image2[p],params.sky_subtract_percent)
        else:
            const = np.percentile(image2,params.sky_subtract_percent)
        image2 -= const
        print 'subtracting sky, constant =',const
        return image2
    else:
        degree = params.sky_degree
        (ni,mi) = image.shape
        sxlen = image.shape[0]/5.0
        sylen = image.shape[1]/5.0
        x = np.zeros(25)
        y = np.zeros(25)
        z = np.zeros(25)
        k = 0
        for i in range(5):
            for j in range(5):
                section = image[int(i*sxlen):int((i+1)*sxlen),
                                int(j*sylen):int((j+1)*sylen)].ravel()
                z[k] = np.min(section[section>params.pixel_min])
                x[k] = ((i+0.5)*sxlen-0.5*(ni-1))/(ni-1)
                y[k] = ((j+0.5)*sylen-0.5*(mi-1))/(mi-1)
                print x[k],y[k],z[k]
                k += 1
        ncoeffs = (degree+1)*(degree+2)/2
        bf = np.zeros([ncoeffs,k])
        m = 0
        for i in range(degree+1):
            for j in range(degree+1-i):
                bf[m,:] = (x[:k]**i) * (y[:k]**j)
                m += 1
        alpha = np.zeros([ncoeffs,ncoeffs])
        beta = np.zeros(ncoeffs)
        for i in range(ncoeffs):
            for j in range(ncoeffs):
                alpha[i,j] = np.sum(bf[i,:]*bf[j,:])
            beta[i] = np.sum(z[:k]*bf[i,:])
        try:
            lu, piv = lu_factor(alpha)
        except LinAlgError:
            print 'LU decomposition failed in subtract_sky'
            return image
        c = lu_solve((lu,piv),beta).astype(np.float32).copy()
        x = (range(ni)-0.5*(ni-1)*np.ones(ni))/(ni-1)
        y = (range(mi)-0.5*(mi-1)*np.ones(mi))/(mi-1)
        xx, yy = np.meshgrid(y,x)
        m = 0
        sky_image = np.zeros_like(image)
        print 'coeffs = ',c
        print 'range x y:',np.min(x),np.max(x),np.min(y),np.max(y)
        for i in range(degree+1):
            for j in range(degree+1-i):
                sky_image += c[m] * (xx**i) * (yy**j)
                m += 1
        sky_image[sky_image<0.0] = 0.0
        image2 = image - sky_image
        return image2
    
            
def mask_cluster(im,mask,params):
    cim = convolve_gauss(im,20)
    p = np.where(cim == np.max(cim))
    xmax = p[0][0]
    ymax = p[1][0]
    x = np.arange(im.shape[0])
    y = np.arange(im.shape[1])
    xx,yy = np.meshgrid(x,y)
    rad2 = params.cluster_mask_radius**2
    q = np.where((xx-xmax)**2 + (yy-ymax)**2 < rad2)
    mask[q] = 0
    return mask
    
    

def define_kernel_pixels_fft(ref,target,rad,INNER_RADIUS=7,threshold=3.0):
    from numpy.fft import fft2, ifft2
    from astropy.stats import mad_std
    nx, ny = ref.image.shape
    x = np.concatenate((np.arange(nx/2),np.arange(-nx/2,0)))
    y = np.concatenate((np.arange(ny/2),np.arange(-ny/2,0)))
    fr = fft2(ref.image)
    ft = fft2(target.image)
    fk = ft/fr
    k = ifft2(fk)
    nk = k/k.max()
    std_nk = mad_std(nk)
    kp = np.where(np.abs(nk)>threshold)
    print 'kernel radius',rad
    crad = int(np.ceil(rad))
    rad2 = rad*rad
    inner_rad2 = INNER_RADIUS*INNER_RADIUS
    kCount = 1
    for p in range(kp[0].shape[0]):
        i = x[kp[0][p]]
        j = y[kp[1][p]]
        r2 = i*i + j*j
        if (r2 < rad2) and ((i,j) != (0,0)):
            if (r2 < inner_rad2):
                kCount += 1
            else:
                if (i/3 == i/3.0) and (j/3 == j/3.0):
                    kCount += 1
    kInd = np.zeros([kCount,2],dtype=np.int32)
    kExtended = np.zeros(kCount,dtype=np.int32)
    kInd[0] = [0,0]
    k = 1
    for p in range(kp[0].shape[0]):
        i = x[kp[0][p]]
        j = y[kp[1][p]]
        r2 = i*i + j*j
        if (r2 < rad2) and ((i,j) != (0,0)):
            if (r2 < inner_rad2):
                kInd[k] = [i,j]
                k += 1
            else:
                if (i/3 == i/3.0) and (j/3 == j/3.0):
                    kInd[k] = [i,j]
                    kExtended[k] = 1
                    k += 1
    n_extend = np.sum(kExtended)
    print kCount-n_extend,'modified delta basis functions'
    print n_extend,'extended basis functions'
    for k in range(kCount):
        print kInd[k]
    return kInd, kExtended


def define_kernel_pixels(rad,INNER_RADIUS=7):
    print 'kernel radius',rad
    crad = int(np.ceil(rad))
    rad2 = rad*rad
    inner_rad2 = INNER_RADIUS*INNER_RADIUS
    kCount = 0
    for i in range(-crad,crad):
        for j in range(-crad,crad):
            r2 = i*i + j*j
            if (r2 < rad2):
                if (r2 < inner_rad2):
                    kCount += 1
                else:
                    if (i/3 == i/3.0) and (j/3 == j/3.0):
                        kCount += 1
    kInd = np.zeros([kCount,2],dtype=np.int32)
    kExtended = np.zeros(kCount,dtype=np.int32)
    kInd[0] = [0,0]
    k = 1
    for i in range(-crad,crad):
        for j in range(-crad,crad):
            r2 = i*i + j*j
            if (r2 < rad2) and ((i,j) != (0,0)):
                if (r2 < inner_rad2):
                    kInd[k] = [i,j]
                    k += 1
                else:
                    if (i/3 == i/3.0) and (j/3 == j/3.0):
                        kInd[k] = [i,j]
                        kExtended[k] = 1
                        k += 1
    n_extend = np.sum(kExtended)
    print kCount-n_extend,'modified delta basis functions'
    print n_extend,'extended basis functions'
    return kInd, kExtended

