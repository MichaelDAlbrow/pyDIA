import numpy as np
from image_functions import convolve_undersample

#
#  Load the C library versions of the CUDA functions
#
import os
import ctypes
from numpy.ctypeslib import ndpointer
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'c_functions_dp.so'
lib = ctypes.cdll.LoadLibrary(dllabspath)

cu_convolve_image_psf = lib.cu_convolve_image_psf
cu_photom = lib.cu_photom
cu_photom_converge = lib.cu_photom_converge
cu_compute_model = lib.cu_compute_model
cu_compute_vector = lib.cu_compute_vector
cu_compute_matrix = lib.cu_compute_matrix
cu_compute_vector_stamps = lib.cu_compute_vector_stamps
cu_compute_matrix_stamps = lib.cu_compute_matrix_stamps


#
#  Specify the ctypes data types for the C function calls
#

cu_convolve_image_psf.restype = None
cu_convolve_image_psf.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int,
                                  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

cu_photom.restype = None
cu_photom.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int,
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ctypes.c_long, ctypes.c_int, ctypes.c_int,
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                      ctypes.c_int]

cu_photom_converge.restype = None
cu_photom_converge.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int,
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ctypes.c_float,ctypes.c_float,
                                ctypes.c_float,ctypes.c_float,
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ctypes.c_int, ctypes.c_int,
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]


cu_compute_model.restype = None
cu_compute_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ctypes.c_int,
                             ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int,
                             ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]


cu_compute_vector.restype = None
cu_compute_vector.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]


cu_compute_matrix.restype = None
cu_compute_matrix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]


cu_compute_vector_stamps.restype = None
cu_compute_vector_stamps.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int,
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]


cu_compute_matrix_stamps.restype = None
cu_compute_matrix_stamps.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int,
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]


                              




def compute_matrix_and_vector_cuda(R,RB,T,Vinv,mask,kernelIndex,extendedBasis,
                                   kernelRadius,params,stamp_positions=None):

    # Create a numpy array for matrix H
    dp = (params.pdeg+1)*(params.pdeg+2)/2
    ds = (params.sdeg+1)*(params.sdeg+2)/2
    db = (params.bdeg+1)*(params.bdeg+2)/2
    hs = (kernelIndex.shape[0]-1)*ds+dp+db
    
    H = np.zeros([hs,hs]).astype(np.float32).copy()
    V = np.zeros(hs).astype(np.float32).copy()

    # Fill the elements of H
    print hs,' * ',hs,' elements'
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.use_stamps:
        posx = np.float32(stamp_positions[:params.nstamps,0].copy()-1.0)
        posy = np.float32(stamp_positions[:params.nstamps,1].copy()-1.0)
        cu_compute_matrix_stamps(params.pdeg,
                                 params.sdeg,
                                 params.bdeg,
                                 R.shape[1],
                                 R.shape[0],
                                 params.nstamps,
                                 params.stamp_half_width,
                                 posx,
                                 posy,
                                 k0,
                                 k1,
                                 extendedBasis,
                                 kernelIndex.shape[0],
                                 np.int(kernelRadius),
                                 H, 256, hs, hs,
                                 np.float32(R), np.float32(RB), np.float32(Vinv),
                                 np.float32(mask))
    else:
        cu_compute_matrix(params.pdeg, params.sdeg,
                          params.bdeg,
                          R.shape[1], R.shape[0],
                          k0,
                          k1,
                          extendedBasis,
                          kernelIndex.shape[0], np.int(kernelRadius),
                          H, 256, hs, hs,
                          np.float32(R),np.float32(RB),np.float32(Vinv),np.float32(mask))

    # Fill the elements of V
    if params.use_stamps:
        cu_compute_vector_stamps(params.pdeg, params.sdeg,
                                 params.bdeg,
                                 R.shape[1], R.shape[0], params.nstamps,
                                 params.stamp_half_width,
                                 posx,
                                 posy,
                                 k0,
                                 k1,
                                 extendedBasis,
                                 kernelIndex.shape[0], np.int(kernelRadius),
                                 V, 256, hs, np.float32(R), np.float32(RB),
                                 np.float32(T), np.float32(Vinv), np.float32(mask))
    else:
        cu_compute_vector(params.pdeg, params.sdeg,
                          params.bdeg,
                          R.shape[1], R.shape[0],
                          k0,
                          k1,
                          extendedBasis,
                          kernelIndex.shape[0], np.int(kernelRadius),
                          V, 256, hs, np.float32(R), np.float32(RB), np.float32(T),
                          np.float32(Vinv), np.float32(mask))
    return H, V, (R, RB)


def compute_model_cuda(image_size,(R,RB),c,kernelIndex,extendedBasis,params):
    

    # Create a numpy array for the model M
    M = np.zeros(image_size).astype(np.float32).copy()
    
    # Call the cuda function to perform the convolution
    blockDim = (256,1,1)
    gridDim = (image_size[1],image_size[0])+(1,)
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    cu_compute_model(params.pdeg, params.sdeg,
                     params.bdeg, k0,
                     k1, extendedBasis,
                     kernelIndex.shape[0],
                     c, M, image_size[1], image_size[0], np.float32(R), np.float32(RB))
    return M


def photom_all_stars(diff,inv_variance,positions,psf_image,c,kernelIndex,
                     extendedBasis,kernelRadius,params,
                     star_group_boundaries,
                     detector_mean_positions_x,detector_mean_positions_y,subtract_stars=False):
    
    from astropy.io import fits
    # Read the PSF
    psf,psf_hdr = fits.getdata(psf_image,0,header='true')
    psf_height = psf_hdr['PSFHEIGH']
    psf_x = psf_hdr['PSFX']
    psf_y = psf_hdr['PSFY']
    psf_size = psf.shape[1]
    psf_fit_rad = params.psf_fit_radius
    if params.psf_profile_type == 'gaussian':
        psf_sigma_x = psf_hdr['PAR1']*0.8493218
        psf_sigma_y = psf_hdr['PAR2']*0.8493218
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float32)
        print 'psf_parameters',psf_parameters
        profile_type = 0
    elif params.psf_profile_type == 'moffat25':
        print 'params.psf_profile_type moffat25 not working yet. Exiting.'
        sys.exit(0)
        psf_sigma_x = psf_hdr['PAR1']
        psf_sigma_y = psf_hdr['PAR2']
        psf_sigma_xy = psf_hdr['PAR3']
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float32)
        print 'psf_parameters',psf_parameters
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)

    
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.star_file_is_one_based:
        posx = np.float32(positions[:,0]-1.0)
        posy = np.float32(positions[:,1]-1.0)
    else:
        posx = np.float32(positions[:,0])
        posy = np.float32(positions[:,1])
        
    #psf_0 = convolve_undersample(psf[0]).astype(np.float32).copy()
    #psf_xd = convolve_undersample(psf[1]).astype(np.float32).copy()*0.0
    #psf_yd = convolve_undersample(psf[2]).astype(np.float32).copy()*0.0
    #psf_0 = psf[0].astype(np.float32).copy()
    #psf_xd = psf[1].astype(np.float32).copy()*0.0
    #psf_yd = psf[2].astype(np.float32).copy()*0.0
    psf_0 = psf.astype(np.float32)
    psf_xd = np.zeros_like(psf_0,dtype=np.float32)
    psf_yd = np.zeros_like(psf_0,dtype=np.float32)
    nstars = positions.shape[0]
    flux = np.zeros(nstars,dtype=np.float32)
    dflux = np.zeros(nstars,dtype=np.float32)

 
    subtract_flag = 0
    if subtract_stars:
      subtract_flag = 1

    print 'nstars', nstars
    print 'flux', flux.shape
    print 'dflux', dflux.shape

    cu_photom(np.int(profile_type), diff.shape[1], diff.shape[0], params.pdeg,
              params.sdeg, c.shape[0], kernelIndex.shape[0],
              np.int(kernelRadius), k0,
              k1, extendedBasis,
              psf_parameters, psf_0, psf_xd, psf_yd,
              posx, posy, c, flux, dflux, long(nstars), 16, 16, np.float32(diff),
              np.float32(inv_variance),np.int32(star_group_boundaries),
              np.float32(detector_mean_positions_x),
              np.float32(detector_mean_positions_y),star_group_boundaries.shape[0],subtract_flag)
    
    return flux, dflux
    

def convolve_image_with_psf(psf_image,image1,image2,c,kernelIndex,extendedBasis,
                            kernelRadius,params):

    from astropy.io import fits

    # Read the PSF
    psf,psf_hdr = fits.getdata(psf_image,0,header='true')
    psf_height = psf_hdr['PSFHEIGH']
    psf_x = psf_hdr['PSFX']
    psf_y = psf_hdr['PSFY']
    psf_size = psf.shape[1]
    psf_fit_rad = params.psf_fit_radius
    if params.psf_profile_type == 'gaussian':
        psf_sigma_x = psf_hdr['PAR1']*0.8493218
        psf_sigma_y = psf_hdr['PAR2']*0.8493218
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float32)
        profile_type = 0
    elif params.psf_profile_type == 'moffat25':
        print 'params.psf_profile_type moffat25 not working yet. Exiting.'
        sys.exit(0)
        psf_sigma_x = psf_hdr['PAR1']
        psf_sigma_y = psf_hdr['PAR2']
        psf_sigma_xy = psf_hdr['PAR3']
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float32)
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)


    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    #psf_0 = convolve_undersample(psf[0]).astype(np.float32).copy()
    #psf_xd = convolve_undersample(psf[1]).astype(np.float32).copy()*0.0
    #psf_yd = convolve_undersample(psf[2]).astype(np.float32).copy()*0.0
    psf_0 = psf.astype(np.float32).copy()


    
    psf_xd = psf.astype(np.float32).copy()*0.0
    psf_yd = psf.astype(np.float32).copy()*0.0

    image_section_size = 32
    convolved_image1 = (0.0*image1).astype(np.float32)
    convolved_image2 = (0.0*image1).astype(np.float32)

    cu_convolve_image_psf(np.int(profile_type),image1.shape[1], image1.shape[0],
                          np.int(image_section_size),
                          np.int(image_section_size), params.pdeg,
                          params.sdeg,c.shape[0],
                          kernelIndex.shape[0],np.int(kernelRadius),k0,
                          k1,extendedBasis,psf_parameters,psf_0,
                          psf_xd,psf_yd,c,convolved_image1,
                          convolved_image2, np.float32(image1),
                          np.float32(image2))

    return convolved_image1, convolved_image2





def photom_variable_star(diff,inv_variance,position,position0,psf_image,c,kernelIndex,
                          extendedBasis,kernelRadius,params):

    from astropy.io import fits
    # Read the PSF
    psf,psf_hdr = fits.getdata(psf_image,0,header='true')
    psf_height = psf_hdr['PSFHEIGH']
    psf_sigma_x = psf_hdr['PAR1']*0.8493218
    psf_sigma_y = psf_hdr['PAR2']*0.8493218
    psf_x = psf_hdr['PSFX']
    psf_y = psf_hdr['PSFY']
    psf_size = psf.shape[1]
    psf_fit_rad = params.psf_fit_radius
    psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                               psf_y,psf_fit_rad,params.gain]).astype(np.float32)

    if params.psf_profile_type == 'gaussian':
        psf_sigma_x = psf_hdr['PAR1']*0.8493218
        psf_sigma_y = psf_hdr['PAR2']*0.8493218
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float32)
        profile_type = 0
    elif params.psf_profile_type == 'moffat25':
        print 'params.psf_profile_type moffat25 not working yet. Exiting.'
        sys.exit(0)
        psf_sigma_x = psf_hdr['PAR1']
        psf_sigma_y = psf_hdr['PAR2']
        psf_sigma_xy = psf_hdr['PAR3']
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float32)
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)
    
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.star_file_is_one_based:
        posx0 = np.float32(position0[0]-1.0)
        posy0 = np.float32(position0[1]-1.0)
        posx = np.float32(position[0]-1.0)
        posy = np.float32(position[1]-1.0)
    else:
        posx0 = np.float32(position0[0].copy())
        posy0 = np.float32(position0[1].copy())
        posx = np.float32(position[0].copy())
        posy = np.float32(position[1].copy())

    #psf_0 = convolve_undersample(psf[0]).astype(np.float32).copy()
    #psf_xd = convolve_undersample(psf[1]).astype(np.float32).copy()*0.0
    #psf_yd = convolve_undersample(psf[2]).astype(np.float32).copy()*0.0
    psf_0 = psf[0].astype(np.float32).copy()
    psf_xd = psf[1].astype(np.float32).copy()*0.0
    psf_yd = psf[2].astype(np.float32).copy()*0.0
    flux = np.array([np.float32(0.0)])
    dflux = np.array([np.float32(0.0)])
    x1 = np.float32(posy.copy() * 0.0)
    x2 = np.float32(posy.copy() * 0.0)
    y1 = np.float32(posy.copy() * 0.0)
    y2 = np.float32(posy.copy() * 0.0)
    sjx1 = np.array([np.float32(posy.copy() * 0.0)])
    sjx2 = np.array([np.float32(posy.copy() * 0.0)])
    sjy1 = np.array([np.float32(posy.copy() * 0.0)])
    sjy2 = np.array([np.float32(posy.copy() * 0.0)])

    cu_photom_converge(profile_type,diff.shape[1], diff.shape[0], params.pdeg,
                       params.sdeg, c.shape[0], kernelIndex.shape[0],
                       np.int(kernelRadius), k0,
                       k1, extendedBasis,
                       psf_parameters, psf_0, psf_xd, psf_yd,
                       posx, posy, posx0, posy0, c, flux, dflux, sjx1, sjx2, sjy1,
                       sjy2, 
                       16, 16, np.float32(diff),
                       np.float32(inv_variance))

        
    return flux, dflux, sjx1, sjx2, sjy1, sjy2
