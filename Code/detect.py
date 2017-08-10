import os
import sys
from astropy.io import fits
import fnmatch
import numpy as np
from scipy.ndimage import label, center_of_mass, fourier_ellipsoid, fourier_gaussian, gaussian_filter
from io_functions import *
from image_functions import *
from photometry_functions import *
import c_interface_functions as ci
from data_structures import Observation

def gauss(x, p):
    A, x0, sigma = p
    return A*numpy.exp(-(x-x0)**2/(2.*sigma**2))


def final_variable_photometry(files,params,coords=None,
                              coord_file=None,psf_file=None):
    
    if params.use_GPU:
        from cuda_interface_functions import *
    else:
        from c_interface_functions import *

    if not(psf_file):
        psf_file = params.loc_output+os.path.sep+'psf.fits'
    if not(os.path.exists(psf_file)):
        print 'Error: PSF file',psf_file,'not found'
        sys.exit(1)
        
    if coords==None:
        if coord_file:
            coords = np.loadtxt(coord_file)
        else:
            print 'Error in final_variable_photometry: no coordinates provided.'
            sys.exit(1)

    flux = np.zeros([len(files),coords.shape[0]])*np.nan
    dflux = np.zeros([len(files),coords.shape[0]])*np.nan
    
    star_group_boundaries = None
    detector_mean_positions_x = None
    detector_mean_positions_y = None
    star_unsort_index = None
            
    if not(params.use_GPU):
        
        star_sort_index,star_group_boundaries,detector_mean_positions_x,detector_mean_positions_y = group_stars_ccd(params,coords,params.loc_output+os.path.sep+'ref.fits')
        scoords = coords[star_sort_index]
        star_unsort_index = np.argsort(star_sort_index)

        for i,f in enumerate(files):
            print f
            basename = os.path.basename(f)
            dfile = params.loc_output+os.path.sep+'d_'+basename
            nfile = params.loc_output+os.path.sep+'n_'+basename
            mfile = params.loc_output+os.path.sep+'z_'+basename
            ktable = params.loc_output+os.path.sep+'k_'+basename
            if os.path.exists(dfile) and os.path.exists(nfile) and os.path.exists(ktable):
                kernelIndex, extendedBasis, c, params = \
                             read_kernel_table(ktable,params)
                kernelRadius = np.max(kernelIndex[:,0])+1
                if np.sum(extendedBasis) > 0:
                    kernelRadius += 1
                diff, h = read_fits_file(dfile)
                norm, h = read_fits_file(nfile)
                mask, h = read_fits_file(mfile)
                inv_var = (norm/diff)**2 + (1-mask)
                diff = undo_photometric_scale(diff,c,params.pdeg)
                
                sflux, sdflux = photom_all_stars(diff,inv_var,coords,
                                                         psf_file,c,kernelIndex,
                                                         extendedBasis,
                                                         kernelRadius,params,
                                                         star_group_boundaries,
                                                         detector_mean_positions_x,
                                                         detector_mean_positions_y)

                flux[i,:] = sflux[star_unsort_index].copy()
                dflux[i,:] = sdflux[star_unsort_index].copy()
                    
    else:

        for i,f in enumerate(files):
            print f
            basename = os.path.basename(f)
            dfile = params.loc_output+os.path.sep+'d_'+basename
            nfile = params.loc_output+os.path.sep+'n_'+basename
            mfile = params.loc_output+os.path.sep+'z_'+basename
            ktable = params.loc_output+os.path.sep+'k_'+basename
            if os.path.exists(dfile) and os.path.exists(nfile) and os.path.exists(ktable):
                kernelIndex, extendedBasis, c, params = \
                             read_kernel_table(ktable,params)
                kernelRadius = np.max(kernelIndex[:,0])+1
                if np.sum(extendedBasis) > 0:
                    kernelRadius += 1
                diff, h = read_fits_file(dfile)
                norm, h = read_fits_file(nfile)
                mask, h = read_fits_file(mfile)
                inv_var = (norm/diff)**2 + (1-mask)
                diff = undo_photometric_scale(diff,c,params.pdeg)
                
                flux[i,:], dflux[i,:] = photom_all_stars(diff,inv_var,coords,
                                                         psf_file,c,kernelIndex,
                                                         extendedBasis,
                                                         kernelRadius,params)


                    
    return flux, dflux


def detect_variables(params,psf_file=None,ccoords=None,time_sigma=4.0,star_coords=None):

    if params.use_GPU:
        from cuda_interface_functions import *
    else:
        from c_interface_functions import *

    #
    # Check that the PSF exists
    #
    if not(psf_file):
        psf_file = params.loc_output+os.path.sep+'psf.fits'
    if not(os.path.exists(psf_file)):
        print 'Error: PSF file',psf_file,'not found'
        sys.exit(1)

    #
    # Read the PSF
    #
    psf,psf_hdr = fits.getdata(psf_file,0,header='true')
    psf_height = psf_hdr['PSFHEIGH']
    psf_sigma_x = psf_hdr['PAR1']*0.8493218
    psf_sigma_y = psf_hdr['PAR2']*0.8493218
    psf_x = psf_hdr['PSFX']
    psf_y = psf_hdr['PSFY']
    psf_size = psf.shape[1]

    #
    # Determine our list of files
    #
    all_files = os.listdir(params.loc_data)
    all_files.sort()
    filenames = []
    nfiles = 0
    for f in all_files:
        if fnmatch.fnmatch(f,params.name_pattern):
            basename = os.path.basename(f)
            dfile = params.loc_output+os.path.sep+'d_'+basename
            nfile = params.loc_output+os.path.sep+'n_'+basename
            ktable = params.loc_output+os.path.sep+'k_'+basename
            if os.path.exists(dfile) and os.path.exists(nfile) and os.path.exists(ktable):
                filenames.append(f)
                nfiles += 1

    if (ccoords==None):

        #
        #  3-d array for image stack
        #
        d = fits.getdata(params.loc_output+os.path.sep+'d_'+filenames[0],
                         header=False)
        xs, ys = d.shape

        good_centroids = []

        j = 0

        while j < len(filenames):

            stack_size = np.min([len(filenames)-j,200]) 
            stackd = np.zeros([stack_size,xs,ys])
            stackn = np.zeros([stack_size,xs,ys])
            print 'len_filenames, j, stack_size', len(filenames), j, stack_size

            nstack = 0
            while (j<len(filenames)) and (nstack < stack_size):

                f = filenames[j]
                j += 1
                
                #
                # Read difference image and kernel table
                #
                basename = os.path.basename(f)
                dfile = params.loc_output+os.path.sep+'d_'+basename
                nfile = params.loc_output+os.path.sep+'n_'+basename
                ktable = params.loc_output+os.path.sep+'k_'+basename
                print dfile, ktable
                if os.path.exists(dfile) and os.path.exists(nfile) and \
                       os.path.exists(ktable):
                    d = fits.getdata(dfile,header=False)
                    n = fits.getdata(nfile,header=False)
                    print np.nanstd(d),np.nanstd(n)
                    if np.nanstd(n) < params.diff_std_threshold:
                        kernelIndex, extendedBasis, c, params = \
                                     read_kernel_table(ktable,params)
                        kernelRadius = np.max(kernelIndex[:,0])+1
                        if np.sum(extendedBasis) > 0:
                            kernelRadius += 1

                        #
                        # Convolve reference PSF with kernel
                        # Convolve difference image with convolved PSF
                        #
                        stackd[nstack,:,:], stackn[nstack,:,:] = convolve_image_with_psf(\
                            psf_file,d,n,c,kernelIndex,extendedBasis,kernelRadius,
                            params)
                        nstack += 1
                        

            #
            # Filter by time
            #
            stackn = gaussian_filter(stackn[:nstack,:,:],sigma=(time_sigma,0.0,0.0))
            stackn[np.isnan(stackn)] = 0.0


            #
            # Detect positive and negative peaks in stack of noise-normalised
            # difference images
            #
            stackn_thresh = stackn[:nstack,:,:].copy()
            n_std = np.std(stackn_thresh)
            stackn_thresh = np.abs(stackn_thresh)
            print 'Standard deviation of norm stack = ',n_std    
            stackn_thresh[stackn_thresh<params.detect_threshold*n_std] = 0
            labeled_image, number_of_objects = label(stackn_thresh)
            if number_of_objects == 0:
                print 'No variable objects detected'
                return
            else:
                centroids = center_of_mass(stackn_thresh, labeled_image, \
                                           np.arange(1, number_of_objects + 1))
            print 'centroids'
            dr = 20
            for cen in centroids:
                t, y, x = cen
                if (x>dr) & (x<xs-dr) & (y>dr) & (y<ys-dr):
                    print cen
                    if star_coords is None:
                        good_centroids.append(cen)
                    else:
                        if (x-star_coords[0])**2 + (y-star_coords[1])**2 < star_coords[2]**2:
                            good_centroids.append(cen)
                    stackn_thresh = -stackn[:nstack,:,:].copy()
                    labeled_image, number_of_objects = label(stackn_thresh)
                    centroids = center_of_mass(stackn_thresh, labeled_image, \
                                               np.arange(1, number_of_objects + 1))
                    for cen in centroids:
                        t, y, x = cen
                        if (x>dr) & (x<xs-dr) & (y>dr) & (y<ys-dr):
                            print cen
                            if star_coords is None:
                                good_centroids.append(cen)
                            else:
                                if (x-star_coords[0])**2 + (y-star_coords[1])**2 < star_coords[2]**2:
                                    good_centroids.append(cen)

            print 'j',j
            if j<len(filenames):
                j -= 30
            print 'j',j
                    


        #
        # Remove repeat coordinates
        #
        print 'final centroids'
        final_centroids = []
        dr = 4
        for i,cen in enumerate(good_centroids):
            x1 = cen[2]
            y1 = cen[1]
            repeat = False
            for j in range(i+1,len(good_centroids)):
                x2 = good_centroids[j][2]
                y2 = good_centroids[j][1]
                if (x1-x2)**2 + (y1-y2)**2 < dr**2:
                    repeat = True
            if not(repeat):
                final_centroids.append(cen)
                print cen
            
        coords = np.zeros([len(final_centroids),2])
        for i,cen in enumerate(final_centroids):
            coords[i,0] = cen[2]
            coords[i,1] = cen[1]
        
        if params.star_file_is_one_based:
            coords += 1.0

        np.savetxt(params.loc_output+os.path.sep+'variables.coords',coords)

        if len(final_centroids) > 0:

            #
            # Do photometry at centroid positions
            #
            print 'Doing photometry at centroid positions'
            flux = np.zeros([len(filenames),len(final_centroids)])*np.nan
            dflux = np.zeros([len(filenames),len(final_centroids)])*np.nan

            star_group_boundaries = None
            detector_mean_positions_x = None
            detector_mean_positions_y = None
            star_unsort_index = None
            
            if not(params.use_GPU):
                star_sort_index,star_group_boundaries,detector_mean_positions_x,detector_mean_positions_y = group_stars_ccd(params,coords,params.loc_output+os.path.sep+'ref.fits')
                scoords = coords[star_sort_index]
                star_unsort_index = np.argsort(star_sort_index)

                for i,f in enumerate(filenames):
                    basename = os.path.basename(f)
                    dfile = params.loc_output+os.path.sep+'d_'+basename
                    nfile = params.loc_output+os.path.sep+'n_'+basename
                    mfile = params.loc_output+os.path.sep+'z_'+basename
                    ktable = params.loc_output+os.path.sep+'k_'+basename
                    if os.path.exists(dfile) and os.path.exists(nfile) and \
                           os.path.exists(ktable):
                        diff, h = read_fits_file(dfile)
                        norm, h = read_fits_file(nfile)
                        mask, h = read_fits_file(mfile)
                        inv_var = (norm/diff)**2 + (1-mask)
                        diff = undo_photometric_scale(diff,c,params.pdeg)
                    
                        sflux, sdflux  = photom_all_stars(diff,inv_var,scoords,
                                                          psf_file,c,kernelIndex,
                                                          extendedBasis,
                                                          kernelRadius,
                                                          params,
                                                          star_group_boundaries,
                                                          detector_mean_positions_x,
                                                          detector_mean_positions_y)
                        
                        
                        flux[i,:] = sflux[star_unsort_index].copy()
                        dflux[i,:] = sdflux[star_unsort_index].copy()

            else:
                
                for i,f in enumerate(filenames):
                    basename = os.path.basename(f)
                    dfile = params.loc_output+os.path.sep+'d_'+basename
                    nfile = params.loc_output+os.path.sep+'n_'+basename
                    mfile = params.loc_output+os.path.sep+'z_'+basename
                    ktable = params.loc_output+os.path.sep+'k_'+basename
                    if os.path.exists(dfile) and os.path.exists(nfile) and \
                           os.path.exists(ktable):
                        diff, h = read_fits_file(dfile)
                        norm, h = read_fits_file(nfile)
                        mask, h = read_fits_file(mfile)
                        inv_var = (norm/diff)**2 + (1-mask)
                        diff = undo_photometric_scale(diff,c,params.pdeg)
                    
                        flux[i,:], dflux[i,:] = photom_all_stars(diff,inv_var,coords,
                                                                 psf_file,c,kernelIndex,
                                                                 extendedBasis,
                                                                 kernelRadius,
                                                                 params)

            for i in range(len(final_centroids)):
                np.savetxt(params.loc_output+os.path.sep+'var%03d.flux'%i,
                           np.vstack((flux[:,i],dflux[:,i])).T)

            #
            # Photometry with coordinate refinement
            #
            print 'Refining coordinates'
            ccoords = coords.copy()
            for i in range(len(final_centroids)):
                ccoords[i,:], flux, dflux = do_photometry_variables(filenames,params,
                                                                    coords[i,:],
                                                                    psf_file=psf_file)
                np.savetxt(params.loc_output+os.path.sep+'cvar%03d.flux'%i,
                       np.vstack((flux,dflux)).T)

            np.savetxt(params.loc_output+os.path.sep+'variables.ccoords',ccoords)

    #
    # Final photometry using DIA routines
    #
    print 'Final photometry'
    flux, dflux = final_variable_photometry(filenames,params,
                                            coords=ccoords,
                                            psf_file=psf_file)

    for i in range(ccoords.shape[0]):
        np.savetxt(params.loc_output+os.path.sep+'fvar%03d.flux'%i,
                   np.vstack((flux[:,i],dflux[:,i])).T)

    #
    # Final photometry of reference image
    #

    #
    # Do photometry of the reference image
    #
    frad = params.psf_fit_radius
    params.psf_fit_radius = 2.1
    ssmode = params.sky_subtract_mode
    params.sky_subtract_mode = 'percent'
    reference_image = 'ref.fits'
    ref = Observation(params.loc_output+os.path.sep+reference_image,params)
    ref.image = ref.data
    ref.inv_variance += 1 - ref.mask
    ktable = params.loc_output+os.path.sep+'k_'+os.path.basename(reference_image)
    kernelIndex, extendedBasis, c, params = read_kernel_table(ktable,params)
    kernelRadius = np.max(kernelIndex[:,0])+1
    if np.sum(extendedBasis) > 0:
        kernelRadius += 1
    print 'kernelIndex', kernelIndex
    print 'extendedBasis',extendedBasis
    print 'coeffs', c
    print 'kernelRadius',kernelRadius
    flux, dflux = photom_all_stars(ref.image,ref.inv_variance,ccoords,
                                   psf_file,c,kernelIndex,extendedBasis,
                                   kernelRadius,params)
    
    for i in range(ccoords.shape[0]):
        np.savetxt(params.loc_output+os.path.sep+'fvar%03d.ref.flux'%i,
                   np.array([flux[i],dflux[i]]))
    params.psf_fit_radius = frad
    params.sky_subtract_mode = ssmode


def do_photometry_variables(files,params,position,extname='vflux',
                            psf_file=None,reference_image='ref.fits',iterations=10):
    
    #
    # Get image size
    #
    dtarget = params.loc_output+os.path.sep+'d_'+ \
                  os.path.basename(files[0])
    d, h = read_fits_file(dtarget)
    imsize = d.shape
    
    #
    # Check that the PSF exists
    #
    if not(psf_file):
        psf_file = params.loc_output+os.path.sep+'psf.fits'
    if not(os.path.exists(psf_file)):
        print 'Error: PSF file',psf_file,'not found'
        sys.exit(1)

    #position[0] += 2.0
    

    #
    # Read and store difference images
    #
    nfiles = len(files)
    xpos = int(np.floor(position[0]+0.5)-16)
    ypos = int(np.floor(position[1]+0.5)-16)
    dd = np.zeros([nfiles,32,32])
    vv = np.zeros([nfiles,32,32])
    nn = np.zeros([nfiles,32,32])
    good = np.zeros(nfiles, dtype=bool)
    slice = (xpos,xpos+32,ypos,ypos+32)
    print 'slice',slice
    
    for j,f in enumerate(files):

        print 'Reading',f
        target = f
        dtarget = params.loc_output+os.path.sep+'d_'+ \
                  os.path.basename(target)
        ntarget = params.loc_output+os.path.sep+'n_'+ \
                  os.path.basename(target)
        ztarget = params.loc_output+os.path.sep+'z_'+ \
                  os.path.basename(target)
        ktable = params.loc_output+os.path.sep+'k_'+ \
                 os.path.basename(target)
                
        if os.path.exists(dtarget) and os.path.exists(ntarget) and \
               os.path.exists(ktable):
            norm, h = read_fits_file(ntarget,slice=slice)
            diff, h = read_fits_file(dtarget,slice=slice)
            mask, h = read_fits_file(ztarget,slice=slice)
            kernelIndex, extendedBasis, c, params = \
                         read_kernel_table(ktable,params)
            diff[abs(diff) > 1.2*params.pixel_max] = np.nan
            inv_var = (norm/diff)**2 + (1-mask)
            diff[np.isnan(diff)] = 0.0
            diff = undo_photometric_scale(diff,c,params.pdeg,
                                          size=imsize,position=(xpos,ypos))
            
            dd[j,:,:] = diff
            nn[j,:,:] = norm
            vv[j,:,:] = inv_var



    ngood = 0
    good_threshold = 6.0
    while (ngood < 12) and (good_threshold > 3.0):
        ngood = 0
        print 'good_threshold =',good_threshold
        for j,f in enumerate(files):
            snorm = np.std(nn[j,:,:])
            print 'j,snorm, max = ',j,snorm, np.max(np.abs(nn[j,14:17,14:17]))
            if not(np.isinf(vv[j,:,:]).any() or np.isnan(vv[j,:,:]).any() or np.isnan(dd[j,:,:]).any()) and (np.nanstd(nn[j,:,:]) < params.diff_std_threshold) and (np.max(np.abs(nn[j,14:17,14:17])) > good_threshold*snorm):
                good[j] = True
                ngood += 1
        good_threshold *= 0.9

    if ngood < 6:
        print 'Convergence failed'
        return position, np.zeros(nfiles), np.zeros(nfiles)

    #
    # Iterate
    #
    position0 = position.copy()
    position1 = position.copy()
    position = np.array([position[0]-int(np.floor(position[0]+0.5))+16,
                         position[1]-int(np.floor(position[1]+0.5))+16])
    for i in range(iterations):
        
        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0
        flux = np.zeros(nfiles)
        dflux = np.zeros(nfiles)

        #
        # Process difference images
        #
        for j,f in enumerate(files):

            print i,j,f
            ktable = params.loc_output+os.path.sep+'k_'+ \
                     os.path.basename(f)

            if os.path.exists(ktable) and good[j]:
                
                kernelIndex, extendedBasis, c, params = \
                             read_kernel_table(ktable,params)
                kernelRadius = np.max(kernelIndex[:,0])+1
                if np.sum(extendedBasis) > 0:
                    kernelRadius += 1

                flux[j], dflux[j], sjx1, sjx2, sjy1, sjy2 = \
                         ci.photom_variable_star(dd[j,:,:],vv[j,:,:],position,
                                                 position0,psf_file,c,kernelIndex,
                                                 extendedBasis,kernelRadius,
                                                 params)


                #print sjx1/sjx2, sjy1/sjy2, sjx1,sjx2,sjy1,sjy2
                #if good[j] & (abs(sjx2)>10) & (abs(sjy2)>10) & (sjx1/sjx2 < 4) & (sjy1/sjy2 < 4):
                if (sjx1/sjx2 < 2) & (sjy1/sjy2 < 2):
                    x1 += sjx1
                    x2 += sjx2
                    y1 += sjy1
                    y2 += sjy2

        if (abs(x2) > 1.e-6) and (abs(y2) > 1.e-6):
            dx = -x1/x2
            dy = -y1/y2
        else:
            dx = 0.0
            dy = 0.0
        print x1,x2,dx
        print y1,y2,dy
        if (abs(dx) > 5.0) | (abs(dy) > 5.0) | np.isnan(dx) | np.isnan(dy):
            break
        
        x0 = position0[0]+dx
        y0 = position0[1]+dy
        x = position[0]+dx
        y = position[1]+dy

        print position,'    +    ',dx,',',dy,'    =    ',x,y
        print position0,'    +    ',dx,',',dy,'    =    ',x0,y0

        position0[0] = x0
        position0[1] = y0
        position[0] = x
        position[1] = y

        if (position1[0]-position0[0])**2 + (position1[1]-position0[1])**2 > 25.0:
            position0 = position1
            print 'Convergence failed'


    return position0, flux, dflux
