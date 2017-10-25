import numpy as np
import fnmatch
import io_functions as IO 
import image_functions as IM

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import linalg as sp_linalg


#
#  Load the C library versions of the CUDA functions
#
import os
import ctypes
from numpy.ctypeslib import ndpointer
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'c_functions.so'
lib = ctypes.cdll.LoadLibrary(dllabspath)

cu_convolve_image_psf = lib.cu_convolve_image_psf
cu_photom = lib.cu_photom
cu_make_residual = lib.cu_make_residual
cu_multi_photom = lib.cu_multi_photom
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
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

cu_photom.restype = None
cu_photom.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int,
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_long, ctypes.c_int, ctypes.c_int,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int]

cu_make_residual.restype = None
cu_make_residual.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int,
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_long, ctypes.c_int, ctypes.c_int,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int]


cu_multi_photom.restype = None
cu_multi_photom.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int,
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_long, ctypes.c_int, ctypes.c_int,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int, 
                      ctypes.POINTER(ctypes.c_int),
                      ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                      ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                      ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int]


cu_photom_converge.restype = None
cu_photom_converge.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, 
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_double,
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_double,ctypes.c_double,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_double,ctypes.c_int]


cu_compute_model.restype = None
cu_compute_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ctypes.c_int,
                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int,
                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


cu_compute_vector.restype = None
cu_compute_vector.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


cu_compute_matrix.restype = None
cu_compute_matrix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


cu_compute_vector_stamps.restype = None
cu_compute_vector_stamps.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int,
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


cu_compute_matrix_stamps.restype = None
cu_compute_matrix_stamps.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int,
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


                              




def compute_matrix_and_vector_cuda(R,RB,T,Vinv,mask,kernelIndex,extendedBasis,
                                   kernelRadius,params,stamp_positions=None):

    # Create a numpy array for matrix H
    dp = (params.pdeg+1)*(params.pdeg+2)/2
    ds = (params.sdeg+1)*(params.sdeg+2)/2
    db = (params.bdeg+1)*(params.bdeg+2)/2
    hs = (kernelIndex.shape[0]-1)*ds+dp+db
    
    H = np.zeros([hs,hs]).astype(np.float64).copy()
    V = np.zeros(hs).astype(np.float64).copy()

    # Fill the elements of H
    print hs,' * ',hs,' elements'
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.use_stamps:
        posx = np.float64(stamp_positions[:params.nstamps,0].copy()-1.0)
        posy = np.float64(stamp_positions[:params.nstamps,1].copy()-1.0)
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
                                 np.float64(R), np.float64(RB), np.float64(Vinv),
                                 np.float64(mask))
    else:
        cu_compute_matrix(params.pdeg, params.sdeg,
                          params.bdeg,
                          R.shape[1], R.shape[0],
                          k0,
                          k1,
                          extendedBasis,
                          kernelIndex.shape[0], np.int(kernelRadius),
                          H, 256, hs, hs,
                          np.float64(R),np.float64(RB),np.float64(Vinv),np.float64(mask))

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
                                 V, 256, hs, np.float64(R), np.float64(RB),
                                 np.float64(T), np.float64(Vinv), np.float64(mask))
    else:
        cu_compute_vector(params.pdeg, params.sdeg,
                          params.bdeg,
                          R.shape[1], R.shape[0],
                          k0,
                          k1,
                          extendedBasis,
                          kernelIndex.shape[0], np.int(kernelRadius),
                          V, 256, hs, np.float64(R), np.float64(RB), np.float64(T),
                          np.float64(Vinv), np.float64(mask))
    return H, V, (R, RB)


def compute_model_cuda(image_size,(R,RB),c,kernelIndex,extendedBasis,params):
    

    # Create a numpy array for the model M
    M = np.zeros(image_size).astype(np.float64).copy()
    
    # Call the cuda function to perform the convolution
    blockDim = (256,1,1)
    gridDim = (image_size[1],image_size[0])+(1,)
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    c64 = c.astype(np.float64).copy()
    cu_compute_model(params.pdeg, params.sdeg,
                     params.bdeg, k0,
                     k1, extendedBasis,
                     kernelIndex.shape[0],
                     c64, M, image_size[1], image_size[0], np.float64(R), np.float64(RB))
    return M




def photom_all_stars(diff,inv_variance,positions,psf_image,c,kernelIndex,
                     extendedBasis,kernelRadius,params,
                     star_group_boundaries,
                     detector_mean_positions_x,detector_mean_positions_y):
    
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
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float64)
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
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float64)
        print 'psf_parameters',psf_parameters
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)

    
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.star_file_is_one_based:
        posx = np.float64(positions[:,0]-1.0)
        posy = np.float64(positions[:,1]-1.0)
    else:
        posx = np.float64(positions[:,0])
        posy = np.float64(positions[:,1])
        
    #psf_0 = convolve_undersample(psf[0]).astype(np.float64).copy()
    #psf_xd = convolve_undersample(psf[1]).astype(np.float64).copy()*0.0
    #psf_yd = convolve_undersample(psf[2]).astype(np.float64).copy()*0.0
    #psf_0 = psf[0].astype(np.float64).copy()
    #psf_xd = psf[1].astype(np.float64).copy()*0.0
    #psf_yd = psf[2].astype(np.float64).copy()*0.0
    psf_0 = psf.astype(np.float64)
    psf_xd = np.zeros_like(psf_0,dtype=np.float64)
    psf_yd = np.zeros_like(psf_0,dtype=np.float64)
    nstars = positions.shape[0]
    flux = np.zeros(nstars,dtype=np.float64)
    dflux = np.zeros(nstars,dtype=np.float64)
    c64 = c.astype(np.float64).copy()

    print 'nstars', nstars
    print 'flux', flux.shape
    print 'dflux', dflux.shape

    cu_photom(np.int(profile_type), diff.shape[1], diff.shape[0], params.pdeg,
              params.sdeg, c.shape[0], kernelIndex.shape[0],
              np.int(kernelRadius), k0,
              k1, extendedBasis,
              psf_parameters, psf_0, psf_xd, psf_yd,
              posx, posy, c64, flux, dflux, long(nstars), 16, 16, np.float64(diff),
              np.float64(inv_variance),np.int32(star_group_boundaries),
              np.float64(detector_mean_positions_x),
              np.float64(detector_mean_positions_y),star_group_boundaries.shape[0])
    
    return flux, dflux
    




def photom_all_stars_simultaneous(diff,inv_variance,positions,psf_image,c,kernelIndex,
                                  extendedBasis,kernelRadius,params,
                                  star_group_boundaries,
                                  detector_mean_positions_x,detector_mean_positions_y):
    
    from astropy.io import fits
    # Read the PSF
    psf,psf_hdr = fits.getdata(psf_image,0,header='true')
    psf_height = psf_hdr['PSFHEIGH']
    psf_x = psf_hdr['PSFX']
    psf_y = psf_hdr['PSFY']
    psf_size = psf.shape[1]
    psf_fit_rad = params.psf_fit_radius
    #psf_fit_rad = 3.1
    if params.psf_profile_type == 'gaussian':
        psf_sigma_x = psf_hdr['PAR1']*0.8493218
        psf_sigma_y = psf_hdr['PAR2']*0.8493218
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float64)
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
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float64)
        print 'psf_parameters',psf_parameters
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)

    
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.star_file_is_one_based:
        posx = np.float64(positions[:,0]-1.0)
        posy = np.float64(positions[:,1]-1.0)
    else:
        posx = np.float64(positions[:,0])
        posy = np.float64(positions[:,1])
        
    psf_0 = psf.astype(np.float64)
    psf_xd = np.zeros_like(psf_0,dtype=np.float64)
    psf_yd = np.zeros_like(psf_0,dtype=np.float64)
    nstars = positions.shape[0]
    flux = np.zeros(nstars+1,dtype=np.float64)
    dflux = np.zeros(nstars+1,dtype=np.float64)
    c64 = c.astype(np.float64).copy()

    print 'nstars', nstars
    print 'flux', flux.shape
    print 'dflux', dflux.shape

    i_index = ctypes.POINTER(ctypes.c_int)()
    j_index = ctypes.POINTER(ctypes.c_int)()
    value = ctypes.POINTER(ctypes.c_double)()
    n_entries = ctypes.c_int()

    rvec = np.zeros(nstars).astype(np.float64).copy()

    for iteration in range(1):

      cu_multi_photom(np.int(profile_type), diff.shape[1], diff.shape[0], params.pdeg,
              params.sdeg, c.shape[0], kernelIndex.shape[0],
              np.int(kernelRadius), k0,
              k1, extendedBasis,
              psf_parameters, psf_0, psf_xd, psf_yd,
              posx, posy, c64, long(nstars), 16, 16, np.float64(diff),
              np.float64(inv_variance),np.int32(star_group_boundaries),
              np.float64(detector_mean_positions_x),
              np.float64(detector_mean_positions_y),star_group_boundaries.shape[0],ctypes.byref(n_entries),
              ctypes.byref(i_index), ctypes.byref(j_index), ctypes.byref(value), rvec, flux[:nstars], iteration)


      n_e = np.int32(n_entries)

      buf_from_mem = ctypes.pythonapi.PyBuffer_FromMemory
      buf_from_mem.restype = ctypes.py_object
      
      buffer = buf_from_mem(i_index, n_e*np.dtype(np.int32).itemsize)
      i_ind = np.frombuffer(buffer, np.int32)

      buffer = buf_from_mem(j_index, n_e*np.dtype(np.int32).itemsize)
      j_ind = np.frombuffer(buffer, np.int32)

      buffer = buf_from_mem(value, n_e*np.dtype(np.float64).itemsize)
      val = np.frombuffer(buffer, np.float64)


      #for row in range(20):
      #  print 'Row', row
      #  q = np.where(i_ind == row)
      #  print q
      #  for qq in q[0]:
      #    print j_ind[qq], val[qq]
      #  print rvec[row]


      A = csc_matrix((val,(i_ind,j_ind)),shape=(nstars, nstars))

      flux = np.float64(sp_linalg.spsolve(A, rvec))
      dflux = sp_linalg.spsolve(A, np.ones_like(rvec))
    
      print 'flux =', flux
      print 'dflux =', dflux

      cdiff = np.float64(diff).copy()

      #cu_make_residual(np.int(profile_type), diff.shape[1], diff.shape[0], params.pdeg,
      #        params.sdeg, c.shape[0], kernelIndex.shape[0],
      #        np.int(kernelRadius), k0,
      #        k1, extendedBasis,
      #        psf_parameters, psf_0, psf_xd, psf_yd,
      #        posx, posy, c64, flux, dflux, long(nstars), 16, 16, cdiff,
      #        np.float64(inv_variance),np.int32(star_group_boundaries),
      #        np.float64(detector_mean_positions_x),
      #        np.float64(detector_mean_positions_y),star_group_boundaries.shape[0])

      #IO.write_image(cdiff,'dref.fits')

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
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float64)
        profile_type = 0
    elif params.psf_profile_type == 'moffat25':
        print 'params.psf_profile_type moffat25 not working yet. Exiting.'
        sys.exit(0)
        psf_sigma_x = psf_hdr['PAR1']
        psf_sigma_y = psf_hdr['PAR2']
        psf_sigma_xy = psf_hdr['PAR3']
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float64)
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)


    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    #psf_0 = convolve_undersample(psf[0]).astype(np.float64).copy()
    #psf_xd = convolve_undersample(psf[1]).astype(np.float64).copy()*0.0
    #psf_yd = convolve_undersample(psf[2]).astype(np.float64).copy()*0.0
    psf_0 = psf.astype(np.float64).copy()

    c64 = c.astype(np.float64).copy()

    
    psf_xd = psf.astype(np.float64).copy()*0.0
    psf_yd = psf.astype(np.float64).copy()*0.0

    image_section_size = 32
    convolved_image1 = (0.0*image1).astype(np.float64)
    convolved_image2 = (0.0*image1).astype(np.float64)

    cu_convolve_image_psf(np.int(profile_type),image1.shape[1], image1.shape[0],
                          np.int(image_section_size),
                          np.int(image_section_size), params.pdeg,
                          params.sdeg,c.shape[0],
                          kernelIndex.shape[0],np.int(kernelRadius),k0,
                          k1,extendedBasis,psf_parameters,psf_0,
                          psf_xd,psf_yd,c64,convolved_image1,
                          convolved_image2, np.float64(image1),
                          np.float64(image2))

    return convolved_image1, convolved_image2





def photom_variable_star(x0,y0,params,patch_half_width=15,converge=True,save_stamps=False,stamp_prefix='mosaic'):

    from astropy.io import fits

    def save_mosaic(stack,nfiles,patch_size,name):
        stamps_per_row = int(np.sqrt(nfiles))
        nrows = (nfiles-1)/stamps_per_row+1;
        mx = stamps_per_row*(patch_size+1)+1
        my = nrows*(patch_size+1)+1
        mosaic = np.ones((my,mx))*1000.0
        for i in range(nfiles):
          mosaic[(i/stamps_per_row)*(patch_size+1)+1:(i/stamps_per_row+1)*(patch_size+1), \
                  (i%stamps_per_row)*(patch_size+1)+1:(i%stamps_per_row+1)*(patch_size+1)] \
                  = stack[i,:,:]
        IO.write_image(mosaic,name)


    ix0 = np.int32(x0+0.5)
    iy0 = np.int32(y0+0.5)

    x_patch = x0 - ix0 + patch_half_width
    y_patch = y0 - iy0 + patch_half_width

    patch_size = 2*patch_half_width+1
    patch_slice = (ix0-patch_half_width, ix0+patch_half_width+1, iy0-patch_half_width, iy0+patch_half_width+1)

    # Obtain a list of files

    all_files = os.listdir(params.loc_data)
    all_files.sort()
    filenames = []
    nfiles = 0

    for f in all_files:

        if fnmatch.fnmatch(f,params.name_pattern):

            basename = os.path.basename(f)
            dfile = params.loc_output+os.path.sep+'d_'+basename
            ktable = params.loc_output+os.path.sep+'k_'+basename

            if os.path.exists(dfile) and os.path.exists(ktable):

                nfiles += 1
                filenames.append(f)

    # Load the kernel tables
    # Load the difference images into a data cube

    dates = np.zeros(nfiles)
    norm_std = np.zeros(nfiles,dtype=np.float64)
    diff_std = np.zeros(nfiles,dtype=np.float64)
    n_kernel = np.zeros(nfiles,dtype=np.int32)
    n_coeffs = np.zeros(nfiles,dtype=np.int32)
    kindex_x = np.arange(0,dtype=np.int32)
    kindex_y = np.arange(0,dtype=np.int32)
    kindex_ext = np.arange(0,dtype=np.int32)
    coeffs = np.arange(0,dtype=np.float64)

    d_image_stack = np.zeros((nfiles,patch_size,patch_size),dtype=np.float64)
    inv_var_image_stack = np.zeros((nfiles,patch_size,patch_size),dtype=np.float64)

    filenames.sort()

    for i, f in enumerate(filenames):

        basename = os.path.basename(f)
        ktable = params.loc_output+os.path.sep+'k_'+basename
        kernelIndex, extendedBasis, c, params = IO.read_kernel_table(ktable,params)
        coeffs = np.hstack((coeffs,c))
        kindex_x = np.hstack((kindex_x,kernelIndex[:,0].T))
        kindex_y = np.hstack((kindex_y,kernelIndex[:,1].T))
        kindex_ext = np.hstack((kindex_ext,extendedBasis))
        n_kernel[i] = kernelIndex.shape[0]
        n_coeffs[i] = c.shape[0]
        dates[i] = IO.get_date(params.loc_data+os.path.sep+basename,key=params.datekey)-2450000

        dfile = params.loc_output+os.path.sep+'d_'+basename
        nfile = params.loc_output+os.path.sep+'n_'+basename
        diff, _ = IO.read_fits_file(dfile)
        diff_sc = IM.undo_photometric_scale(diff,c,params.pdeg)
        d_image_stack[i,:,:] = diff_sc[patch_slice[2]:patch_slice[3],patch_slice[0]:patch_slice[1]]
        norm, _ = IO.read_fits_file(nfile,slice=patch_slice)
        inv_var_image_stack[i,:,:] = (norm/d_image_stack[i,:,:])**2
        diff_std[i] = np.std(diff)
        d_image_stack[i,:,:] -= np.median(d_image_stack[i,:,:])

    if save_stamps:
      save_mosaic(d_image_stack,nfiles,patch_size,params.loc_output+os.path.sep+stamp_prefix+'.fits')

    qd1 = np.arange(len(filenames))
    for iter in range(10):
        qd = np.where(diff_std[qd1]<np.mean(diff_std[qd1])+3*np.std(diff_std[qd1]))
        qd1 = qd1[qd]

    print 'mean(diff) :',np.mean(diff_std[qd1])
    print 'std(diff) :',np.std(diff_std[qd1])
    print '1-sig threshold:', np.mean(diff_std[qd1])+1*np.std(diff_std[qd1])
    print '3-sig threshold:', np.mean(diff_std[qd1])+3*np.std(diff_std[qd1])

    print '1-sig diff reject:',np.where(diff_std>np.mean(diff_std[qd1])+1*np.std(diff_std[qd1]))
    print '3-sig diff reject:',np.where(diff_std>np.mean(diff_std[qd1])+3*np.std(diff_std[qd1]))

    threshold = np.mean(diff_std[qd1])+3*np.std(diff_std[qd1])

    # Read the PSF

    psf_image = params.loc_output+os.path.sep+'psf.fits'
    psf,psf_hdr = fits.getdata(psf_image,0,header='true')
    psf_height = psf_hdr['PSFHEIGH']
    psf_sigma_x = psf_hdr['PAR1']*0.8493218
    psf_sigma_y = psf_hdr['PAR2']*0.8493218
    psf_x = psf_hdr['PSFX']
    psf_y = psf_hdr['PSFY']
    psf_size = psf.shape[1]
    psf_fit_rad = params.psf_fit_radius
    psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                               psf_y,psf_fit_rad,params.gain]).astype(np.float64)

    if params.psf_profile_type == 'gaussian':
        psf_sigma_x = psf_hdr['PAR1']*0.8493218
        psf_sigma_y = psf_hdr['PAR2']*0.8493218
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,psf_fit_rad,params.gain]).astype(np.float64)
        profile_type = 0
    elif params.psf_profile_type == 'moffat25':
        print 'params.psf_profile_type moffat25 not working yet. Exiting.'
        sys.exit(0)
        psf_sigma_x = psf_hdr['PAR1']
        psf_sigma_y = psf_hdr['PAR2']
        psf_sigma_xy = psf_hdr['PAR3']
        psf_parameters = np.array([psf_size,psf_height,psf_sigma_x,psf_sigma_y,psf_x,
                                   psf_y,
                                   psf_fit_rad,params.gain,psf_sigma_xy]).astype(np.float64)
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)

    psf_0 = psf.astype(np.float64).copy()
    psf_xd = psf.astype(np.float64).copy()*0.0
    psf_yd = psf.astype(np.float64).copy()*0.0
    flux = np.zeros(nfiles,dtype=np.float64)
    dflux = np.zeros(nfiles,dtype=np.float64)

    x0_arr = np.atleast_1d(np.array([x0],dtype=np.float64))
    y0_arr = np.atleast_1d(np.array([y0],dtype=np.float64))

    cu_photom_converge(profile_type, patch_half_width, params.pdeg, params.sdeg, nfiles, 
                        n_kernel, kindex_x, kindex_y, kindex_ext, n_coeffs, coeffs.astype(np.float64),
                        psf_parameters, psf_0, psf_xd, psf_yd,
                        np.float64(d_image_stack.ravel()), inv_var_image_stack, diff_std, np.float64(threshold),
                        x0_arr, y0_arr, x_patch, y_patch, diff.shape[1], diff.shape[0], 16, 16, flux, dflux, np.float64(params.gain),np.int32(converge))

    if save_stamps:
        save_mosaic(d_image_stack,nfiles,patch_size,params.loc_output+os.path.sep+'p'+stamp_prefix+'.fits')

    return dates, flux, dflux, diff_std/threshold, x0_arr[0], y0_arr[0]

