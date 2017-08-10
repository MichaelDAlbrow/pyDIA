import sys
import os
import numpy as np
from astropy.io import fits
from pyraf import iraf

from io_functions import read_fits_file, write_image
from image_functions import compute_saturated_pixel_mask, subtract_sky


def transform_coeffs(deg,dx,xx,yy):
    a = np.zeros((deg+1,deg+1))
    nterms = (deg+1)*(deg+2)/2
    M = np.zeros((nterms,nterms))
    v = np.zeros(nterms)
    i = 0
    for m in range(deg+1):
        for n in range(deg+1-m):
            v[i] = np.sum(dx* xx**m * yy**n)
            j = 0
            for p in range(deg+1):
                for q in range(deg+1-p):
                    M[i,j] = np.sum(xx**(m+p) * yy**(n+q))
                    j += 1
            i += 1
    c = np.linalg.solve(M,v)
    i = 0
    for m in range(deg+1):
        for n in range(deg+1-m):
            a[m,n] = c[i]
            i += 1
    return a


def compute_xy_shift(pos1,pos2,threshold,dx=0.0,dy=0.0,degree=0):

    x1 = pos1[:,0]
    y1 = pos1[:,1]
    x2 = pos2[:,0]
    y2 = pos2[:,1]
    xx = (x1 - np.mean(x1))/np.mean(x1)
    yy = (y1 - np.mean(y1))/np.mean(y1)
    print 'Matching positions for',len(x1),'stars'

    match = np.zeros_like(x1,dtype=np.int32)
    deltax = np.zeros_like(x1)
    deltay = np.zeros_like(x1)

    for deg in range(degree+1):

        a = np.zeros((deg+1,deg+1))
        b = np.zeros((deg+1,deg+1))

        if deg == 0:
            a[0,0] = dx
            b[0,0] = dy
        else:
            for m in range(deg):
                for n in range(deg-m):
                    a[m,n] = a_prev[m,n]
                    b[m,n] = b_prev[m,n]
        
        for scale in range(21,0,-4):

            xoffset = np.zeros_like(x1)
            yoffset = np.zeros_like(x1)
            for m in range(deg+1):
                for n in range(deg+1-m):
                    xoffset += a[m,n]*(xx**m)*(yy**n)
                    yoffset += b[m,n]*(xx**m)*(yy**n)
                    
            for j1 in range(len(x1)):
                r2 = (x1[j1]-x2-xoffset[j1])**2 + (y1[j1]-y2-yoffset[j1])**2
                mm = np.where(r2 == np.min(r2))
                try:
                    match[j1] = np.where(r2 == np.min(r2))[0][0]
                except:
                    print r2
                    print np.min(np.sqrt(r2))
                    print mm
                    print mm[0]
                    sys.exit(0)
                deltax[j1] = x1[j1] - x2[match[j1]] - xoffset[j1]
                deltay[j1] = y1[j1] - y2[match[j1]] - yoffset[j1]
                
            deltar = np.sqrt(deltax**2 + deltay**2)
            good = np.where(deltar<scale*threshold)[0]

            dx = x1 - x2[match]
            dy = y1 - y2[match]

            a = transform_coeffs(deg,dx[good],xx[good],yy[good])
            b = transform_coeffs(deg,dy[good],xx[good],yy[good])

            print 'degree', deg, 'using',good.shape[0],'stars'
            print 'threshold = ',scale*threshold,'pixels'
            print 'a = ',a
            print 'b = ',b
            print 'std = ',np.std(deltar),'(all)   ',np.std(deltar[good]),'(matched)'


        a_prev = a
        b_prev = b

    return a, b




def detect_stars(f,params):
    print 'Detecting stars in',f.name
    fp = params.loc_output+os.path.sep
    fn = f.fullname
    iraf.digiphot()
    iraf.daophot()
    print 'FWHM = ',f.fw
    nstars = 0
    thresh = 100
    while (nstars < 2*params.nstamps) and (thresh > 1.5):
        print 'thresh = ',thresh
        for d in ['temp.stars','temp.phot']:
            if os.path.exists(fp+d):
                os.system('/bin/rm '+fp+d)
        iraf.daofind(image=fn,output=fp+'temp.stars',interactive='no',verify='no',
                     threshold=thresh,sigma=30,fwhmpsf=f.fw,
                     datamin=params.pixel_min,datamax=params.pixel_max,
                     epadu=params.gain,readnoise=params.readnoise,
                     noise='poisson')
        iraf.phot(image=fn,output=fp+'temp.phot',coords=fp+'temp.stars',interactive='no',
                  verify='no',
                  sigma=30,fwhmpsf=f.fw,datamin=params.pixel_min,
                  datamax=params.pixel_max,epadu=params.gain,
                  readnoise=params.readnoise,noise='poisson',Stdout='/dev/null')
        nstars = 0
        if os.path.exists(fp+'temp.phot'):
            iraf.psort(infiles=fp+'temp.phot',field='MAG')   
            iraf.prenumber(infile=fp+'temp.phot')
            s = iraf.pdump(infiles=fp+'temp.phot',Stdout=1,fields='ID,XCENTER,YCENTER,MAG',
                           expr='yes')
            stars = np.zeros([len(s),3])
            i = 0
            for line in s:
                mag = line.split()[3]
                if not(mag == 'INDEF'):
                    stars[i,:] = np.array(map(float,line.split()[1:4]))
                    i += 1
            nstars = i
        thresh = thresh*0.5
    if nstars == 0:
        print 'Error: could not detect stars in',fn
        return None
    stars = stars[:i,:].copy()
    sys.old_stdout = sys.stdout
    return stars


def choose_stamps(f,params):
    mask = compute_saturated_pixel_mask(f.image,6,params)
    stars = detect_stars(f,params)
    (xmax,ymax) = f.image.shape
    n_good = 0
    snum = np.zeros(params.nstamps).astype(np.int)
    md = params.stamp_edge_distance
    q = np.where((stars[:,0] > md) & (stars[:,0] < xmax-md) &
                 (stars[:,1] > md) & (stars[:,1] < ymax-md))
    if len(q[0]) >= params.nstamps:
        gstars = stars[q]
    else:
        print 'Warning: using stamps close to edge of detector'
        gstars = stars
    md = params.stamp_half_width
    i = 0
    while (n_good < params.nstamps) & (i<gstars.shape[0]):
        if ((gstars[i,0] > md) & (gstars[i,0] < xmax-md) & (gstars[i,1] > md) &
            (gstars[i,1] < ymax-md)):
            mstamp = mask[gstars[i,0]-md:gstars[i,0]+md,gstars[i,1]-md:gstars[i,1]+md]
            q = np.where(mstamp<1)
            if len(q[0]) == 0:
                snum[n_good] = i
                n_good += 1
        i += 1
    if n_good < params.nstamps:
        print 'Warning: stamps may contain saturated pixels'
        stamps = gstars[:params.nstamps,:]
    else:
        stamps = gstars[snum]
    return stamps


def rewrite_psg(file1,file2):
    min_separation = 100.0 
    q = open(file2,'w')
    lastgroup = -1
    for line in open(file1,'r'):
        if line[0] == '#':
            q.write(line)
        else:
            group = int(line.split()[1])
            if group > lastgroup:
                lastgroup = group
                x0 = float(line.split()[2])
                y0 = float(line.split()[2])
            else:
                x = float(line.split()[2])
                y = float(line.split()[2])
                separation = np.sqrt((x-x0)**2 + (y-y0)**2)
                if separation < min_separation:
                    min_separation = separation
                q.write(line)
    q.close()
    return int(min_separation)




def compute_psf_image(params,g,psf_deg=1,psf_rad=8,
                      star_file='phot.mags',psf_image='psf.fits',edge_dist=5):
    iraf.digiphot()
    iraf.daophot()
    fp = params.loc_output+os.path.sep

    f_im = g.image*g.mask
    f = fp+'temp.ref.fits'
    write_image(f_im,f)

    g.fw = np.max([1.5,g.fw])

    logfile = fp+'psf.log'

    fd = fits.getdata(f)
    xmax = fd.shape[0] - edge_dist
    ymax = fd.shape[1] - edge_dist
    

    for d in ['temp.stars','temp.phot','temp.phot1','temp.phot2','temp.pst',
              'temp.opst','temp.opst2',
              'temp.psf.fits','temp.psf1.fits','temp.psf2.fits','temp.psg',
              'temp.psg2','temp.psg3','temp.psg5','temp.rej','temp.rej2',
              'temp.sub.fits','temp.sub1.fits',
              'temp.sub2.fits','temp.opst1','temp.opst3','temp.rej3',
              'temp.nst','temp.stars1','ref.mags',psf_image,'temp.als',
              'temp.als2']:
            if os.path.exists(fp+d):
                os.remove(fp+d)


    # locate stars
    iraf.daofind(image=f,output=fp+'temp.stars',interactive='no',verify='no',
                 threshold=3,sigma=params.star_detect_sigma,fwhmpsf=g.fw,
                 datamin=1,datamax=params.pixel_max,
                 epadu=params.gain,readnoise=params.readnoise,
                 noise='poisson')

    if params.star_file:
        als_recenter = 'no'
        all_template_stars = np.genfromtxt(params.star_file)
        all_new_stars = np.genfromtxt(fp+'temp.stars')
        
        if all_new_stars.shape[0] > params.star_file_number_match:
            new_stars = all_new_stars[all_new_stars[:,2].argsort()][:params.star_file_number_match]
        else:
            new_stars = all_new_stars

        if all_template_stars.shape[0] > params.star_file_number_match:
            template_stars = all_template_stars[all_template_stars[:,3].argsort()][:params.star_file_number_match]
        else:
            template_stars = all_template_stars

        tx, ty = compute_xy_shift(new_stars,template_stars[:,1:3],0.5,
                                  degree=params.star_file_transform_degree)

        if params.star_file_has_magnitudes:
            star_positions = all_template_stars[:,1:4]
            xx = (star_positions[:,0]-np.mean(new_stars[:,0]))/np.mean(new_stars[:,0])
            yy = (star_positions[:,1]-np.mean(new_stars[:,1]))/np.mean(new_stars[:,1])
            for m in range(params.star_file_transform_degree+1):
                for n in range(params.star_file_transform_degree+1-m):
                    star_positions[:,0] += tx[m,n]* xx**m * yy**n
                    star_positions[:,1] += ty[m,n]* xx**m * yy**n
            np.savetxt(fp+'temp.stars.1',star_positions,fmt='%10.3f %10.3f %10.3f')
        else:
            star_positions = all_template_stars[:,1:3]
            xx = (star_positions[:,0]-np.mean(new_stars[:,0]))/np.mean(new_stars[:,0])
            yy = (star_positions[:,1]-np.mean(new_stars[:,1]))/np.mean(new_stars[:,1])
            for m in range(params.star_file_transform_degree+1):
                for n in range(params.star_file_transform_degree+1-m):
                    star_positions[:,0] += tx[m,n]* xx**m * yy**n
                    star_positions[:,1] += ty[m,n]* xx**m * yy**n
            np.savetxt(fp+'temp.stars.1',star_positions,fmt='%10.3f %10.3f')
        all_template_stars[:,1] = star_positions[:,0]
        all_template_stars[:,2] = star_positions[:,1]
            
    else:
        
        als_recenter = 'yes'
        star_positions = np.genfromtxt(fp+'temp.stars')
        np.savetxt(fp+'temp.stars.1',star_positions[:,:2],fmt='%10.3f %10.3f')

    iraf.phot(image=f,output=fp+'temp.phot',coords=fp+'temp.stars.1',interactive='no',
              verify='no',
              sigma=params.star_detect_sigma,fwhmpsf=g.fw,apertures=g.fw,
              datamin=1,
              datamax=2*params.pixel_max,epadu=params.gain,annulus=3*g.fw,
              dannulus=3.0,
              readnoise=params.readnoise,noise='poisson')

    print 'fw = ',g.fw
    #fw = np.max([4.0,fw])
    #print 'fw = ',fw


    # select PSF stars
    iraf.pstselect(image=f,photfile=fp+'temp.phot',pstfile=fp+'temp.pst',maxnpsf=40,
                   interactive='no',verify='no',datamin=1,fitrad=2.0,
                   datamax=params.pixel_max,epadu=params.gain,psfrad=3*np.max([g.fw,1.8]),
                   readnoise=params.readnoise,noise='poisson')

    if params.star_file and params.star_file_has_magnitudes:

        # We don't need to do the photometry - only make the PSF

        # Initial PSF estimate to generate PSF groups
        iraf.psf(image=f,photfile=fp+'temp.phot',pstfile=fp+'temp.pst',psfimage=fp+'temp.psf',
                 function=params.psf_profile_type,opstfile=fp+'temp.opst',
                 groupfile=fp+'temp.psg',
                 interactive='no',
                 verify='no',varorder=0 ,psfrad=2*np.max([g.fw,1.8]),
                 datamin=-10000,datamax=0.95*params.pixel_max,
                 scale=1.0)

        # construct a file of the psf neighbour stars
        slist = []
        psf_stars = np.loadtxt(fp+'temp.opst',usecols=(0,1,2))

        for star in range(psf_stars.shape[0]):

            xp = psf_stars[star,1]
            yp = psf_stars[star,2]
            xmin = np.max([np.int(xp-10*g.fw),0])
            xmax = np.min([np.int(xp+10*g.fw),f_im.shape[0]])
            ymin = np.max([np.int(yp-10*g.fw),0])
            ymax = np.min([np.int(yp+10*g.fw),f_im.shape[1]])

            p = star_positions[np.logical_and(np.logical_and(star_positions[:,0]>xmin,
                                                             star_positions[:,0]<xmax),
                                              np.logical_and(star_positions[:,1]>ymin,
                                                             star_positions[:,1]<ymax))]
            slist.append(p)

        group_stars = np.concatenate(slist)
        np.savetxt(fp+'temp.nst',group_stars,fmt='%10.3f %10.3f %10.3f')
        
        
        # subtract PSF star neighbours
        iraf.substar(image=f,photfile=fp+'temp.nst',psfimage=fp+'temp.psf',
                     exfile=fp+'temp.opst',fitrad=2.0,
                     subimage=fp+'temp.sub1',verify='no',datamin=1,
                     datamax=params.pixel_max,epadu=params.gain,
                     readnoise=params.readnoise,noise='poisson')
        
        # final PSF
        iraf.psf(image=fp+'temp.sub1',photfile=fp+'temp.phot',pstfile=fp+'temp.opst',
                 psfimage=psf_image,psfrad=5*g.fw,
                 function=params.psf_profile_type,opstfile=fp+'temp.opst2',
                 groupfile=fp+'temp.psg2',
                 interactive='no',
                 verify='no',varorder=0,
                 datamin=1,datamax=0.95*params.pixel_max,
                 scale=1.0)

        np.savetxt(fp+'ref.mags',all_template_stars,fmt='%7d %10.3f %10.3f %10.3f')
        stars = all_template_stars

    else:




        # initial PSF estimate
        iraf.psf(image=f,photfile=fp+'temp.phot',pstfile=fp+'temp.pst',psfimage=fp+'temp.psf',
                 function=params.psf_profile_type,opstfile=fp+'temp.opst',
                 groupfile=fp+'temp.psg1',
                 interactive='no',
                 verify='no',varorder=0 ,psfrad=5*g.fw,
                 datamin=1,datamax=0.95*params.pixel_max,
                 scale=1.0)


        # separation distance of near neighbours
        separation = np.max([rewrite_psg(fp+'temp.psg1',fp+'temp.psg2'),3])
        print 'separation = ',separation

        # subtract all stars using truncated PSF
        iraf.allstar(image=f,photfile=fp+'temp.phot',psfimage=fp+'temp.psf',
                     allstarfile=fp+'temp.als',rejfile='',
                     subimage=fp+'temp.sub',verify='no',psfrad=3*g.fw,fitrad=2.0,
                     recenter='yes',groupsky='yes',fitsky='yes',sannulus=7,wsannulus=10,
                     datamin=1,datamax=params.pixel_max,
                     epadu=params.gain,readnoise=params.readnoise,
                     noise='poisson')

        if params.star_file:

            os.system('cp '+fp+'temp.phot '+fp+'temp.phot2') 

        else:
        
            # locate new stars
            iraf.daofind(image=fp+'temp.sub',output=fp+'temp.stars1',interactive='no',verify='no',
                         threshold=3,sigma=params.star_detect_sigma,fwhmpsf=3*g.fw,
                         datamin=1,datamax=params.pixel_max,
                         epadu=params.gain,readnoise=params.readnoise,
                         noise='poisson')


            # magnitudes for new stars
            iraf.phot(image=fp+'temp.sub',output=fp+'temp.phot1',coords=fp+'temp.stars1',
                      interactive='no',
                      verify='no',sigma=params.star_detect_sigma,
                      fwhmpsf=g.fw,datamin=1,
                      datamax=params.pixel_max,epadu=params.gain,
                      readnoise=params.readnoise,noise='poisson')

            # join star lists together
            iraf.pconcat(infiles=fp+'temp.phot,'+fp+'temp.phot1',outfile=fp+'temp.phot2')

        # new PSF estimate to generate PSF groups
        iraf.psf(image=f,photfile=fp+'temp.phot2',pstfile=fp+'temp.pst',psfimage=fp+'temp.psf2',
                 function=params.psf_profile_type,opstfile=fp+'temp.opst2',
                 groupfile=fp+'temp.psg3',
                 interactive='no',
                 verify='no',varorder=0 ,psfrad=5*g.fw,
                 datamin=-10000,datamax=0.95*params.pixel_max,
                 scale=1.0)

        # magnitudes for PSF group stars
        iraf.nstar(image=f,groupfile=fp+'temp.psg3',psfimage=fp+'temp.psf2',
                   nstarfile=fp+'temp.nst',
                   rejfile='',verify='no',psfrad=5*g.fw,fitrad=2.0,
                   recenter='no',
                   groupsky='yes',fitsky='yes',sannulus=7,wsannulus=10,
                   datamin=1,datamax=params.pixel_max,
                   epadu=params.gain,readnoise=params.readnoise,noise='poisson')

        # subtract PSF star neighbours
        iraf.substar(image=f,photfile=fp+'temp.nst',psfimage=fp+'temp.psf2',
                     exfile=fp+'temp.opst2',fitrad=2.0,
                     subimage=fp+'temp.sub1',verify='no',datamin=1,
                     datamax=params.pixel_max,epadu=params.gain,
                     readnoise=params.readnoise,noise='poisson')
        
        # final PSF
        iraf.psf(image=fp+'temp.sub1',photfile=fp+'temp.phot2',
                 pstfile=fp+'temp.opst2',
                 psfimage=psf_image,psfrad=5*g.fw,
                 function=params.psf_profile_type,opstfile=fp+'temp.opst3',
                 groupfile=fp+'temp.psg5',
                 interactive='no',
                 verify='no',varorder=0,
                 datamin=1,datamax=0.95*params.pixel_max,
                 scale=1.0)

        # final photometry

        
        iraf.allstar(image=g.fullname,photfile=fp+'temp.phot2',psfimage=psf_image,
                     allstarfile=fp+'temp.als2',rejfile='',
                     subimage=fp+'temp.sub2',verify='no',psfrad=5*g.fw,
                     recenter=als_recenter,groupsky='yes',fitsky='yes',sannulus=7,
                     wsannulus=10,fitrad=2.0,
                     datamin=params.pixel_min,datamax=params.pixel_max,
                     epadu=params.gain,readnoise=params.readnoise,
                     noise='poisson')

        psfmag = 10.0
        for line in open(fp+'temp.als2','r'):
            sline = line.split()
            if sline[1] == 'PSFMAG':
                psfmag = float(sline[3])
                break

        if params.star_file:
            
            iraf.psort(infiles=fp+'temp.als2',field='ID')
            os.system('cp '+fp+'temp.als2 '+fp+'temp.als3') 

        else:
        
            selection = 'XCE >= '+str(edge_dist)+' && XCE <= '+str(xmax)+' && YCE >= '+str(edge_dist)+' && YCE <= '+str(ymax)+' && MAG != INDEF'
            iraf.pselect(infiles=fp+'temp.als2',outfiles=fp+'temp.als3',expr=selection)
            iraf.psort(infiles=fp+'temp.als3',field='MAG')   
            iraf.prenumber(infile=fp+'temp.als3')
            
        s = iraf.pdump(infiles=fp+'temp.als3',Stdout=1,
                       fields='ID,XCENTER,YCENTER,MAG,MERR,SHARPNESS,CHI',expr='yes')
        sf = [k.replace('INDEF','-1') for k in s]
        stars = np.zeros([len(sf),3])
        for i, line in enumerate(sf):
            stars[i,:] = np.array(map(float,sf[i].split()[1:4]))

        s = iraf.pdump(infiles=fp+'temp.als3',Stdout=1,
                       fields='ID,XCENTER,YCENTER,MAG,MERR,SHARPNESS,CHI',expr='yes')
        sf = [k.replace('INDEF','-1') for k in s]
        with open(fp+'ref.mags','w') as fid:
            for s in sf:
                fid.write(s+'\n')

    return stars



def group_stars_ccd(params,star_positions,reference):
    print 'grouping stars'
    d, h = read_fits_file(reference)
    ccd_size = d.shape
    print d.shape
    xpos = np.abs(star_positions[:,0])
    ypos = np.abs(star_positions[:,1])
    g_size = params.ccd_group_size
    n_groups_x = (ccd_size[1]-1)/g_size + 1
    n_groups_y = (ccd_size[0]-1)/g_size + 1
    print np.min(xpos), np.min(ypos)
    print np.max(xpos), np.max(ypos)
    print n_groups_x, n_groups_y
    indx = (xpos*0).astype(np.int)
    c = 0
    k = 0
    mposx = np.zeros(n_groups_x*n_groups_y)
    mposy = np.zeros(n_groups_x*n_groups_y)
    g_bound = np.zeros(n_groups_x*n_groups_y).astype(np.int)
    for i in range(n_groups_x):
        for j in range(n_groups_y):
            print 'group',i,j,i*g_size,(i+1)*g_size,j*g_size,(j+1)*g_size
            mposx[k] = (i+0.5)*g_size
            mposy[k] = (j+0.5)*g_size
            p = np.where((xpos>=i*g_size) & (xpos<(i+1)*g_size) &
                         (ypos>=j*g_size) & (ypos<(j+1)*g_size))[0]
            if p.shape[0]:
                pn = p.shape[0]
                indx[c:c+pn] = p
                c += pn
                print k, pn, c
            g_bound[k] = c
            k += 1
    return indx,g_bound,mposx,mposy
                
            
