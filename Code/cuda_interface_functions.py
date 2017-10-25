import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from cuda_functions import cu_matrix_kernel
from image_functions import convolve_undersample
import sys


def numpy3d_to_array(np_array, allow_surface_bind=False, layered=True):    

    d, h, w = np_array.shape

    descr = cuda.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = cuda.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    if allow_surface_bind:
        descr.flags = cuda.array3d_flags.SURFACE_LDST

    if layered:
        descr.flags = cuda.array3d_flags.ARRAY3D_LAYERED

    device_array = cuda.Array(descr)

    copy = cuda.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return device_array


def array_to_numpy3d(cuda_array):

    descriptor = cuda_array.get_descriptor_3d()

    w = descriptor.width
    h = descriptor.height
    d = descriptor.depth

    shape = d, h, w

    dtype = array_format_to_dtype(descriptor.format)

    numpy_array=np.zeros(shape, dtype)

    copy = cuda.Memcpy3D()
    copy.set_src_array(cuda_array)
    copy.set_dst_host(numpy_array)

    itemsize = numpy_array.dtype.itemsize

    copy.width_in_bytes = copy.src_pitch = w*itemsize
    copy.src_height = copy.height = h
    copy.depth = d

    copy()

    return numpy_array


def compute_matrix_and_vector_cuda(R,RB,T,Vinv,mask,kernelIndex,extendedBasis,
                                   kernelRadius,params,stamp_positions=None):

    # Import CUDA function to compute the matrix
    cu_compute_matrix = cu_matrix_kernel.get_function('cu_compute_matrix')
    cu_compute_vector = cu_matrix_kernel.get_function('cu_compute_vector')
    cu_compute_matrix_stamps = cu_matrix_kernel.get_function('cu_compute_matrix_stamps')
    cu_compute_vector_stamps = cu_matrix_kernel.get_function('cu_compute_vector_stamps')
    
    # Copy the reference, target and inverse variance images to
    # GPU texture memory
    RTV = np.array([R,RB,T,Vinv,mask]).astype(np.float32).copy()
    RTV_cuda = numpy3d_to_array(RTV)
    texref = cu_matrix_kernel.get_texref("tex")
    texref.set_array(RTV_cuda)
    texref.set_filter_mode(cuda.filter_mode.POINT)

    # Create a numpy array for matrix H
    dp = (params.pdeg+1)*(params.pdeg+2)/2
    ds = (params.sdeg+1)*(params.sdeg+2)/2
    db = (params.bdeg+1)*(params.bdeg+2)/2
    hs = (kernelIndex.shape[0]-1)*ds+dp+db
    
    H = np.zeros([hs,hs]).astype(np.float32).copy()
    V = np.zeros(hs).astype(np.float32).copy()

    # Fill the elements of H
    print hs,' * ',hs,' elements'
    blockDim = (256,1,1)
    gridDim = (hs,hs,1)
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    if params.use_stamps:
        posx = np.float32(stamp_positions[:params.nstamps,0].copy()-1.0)
        posy = np.float32(stamp_positions[:params.nstamps,1].copy()-1.0)
        cu_compute_matrix_stamps(np.int32(params.pdeg),
                                 np.int32(params.sdeg),
                                 np.int32(params.bdeg),
                                 np.int32(R.shape[1]),
                                 np.int32(R.shape[0]),
                                 np.int32(params.nstamps),
                                 np.int32(params.stamp_half_width),
                                 cuda.In(posx),
                                 cuda.In(posy),
                                 cuda.In(k0),
                                 cuda.In(k1),
                                 cuda.In(extendedBasis),
                                 np.int32(kernelIndex.shape[0]),
                                 np.int32(kernelRadius),
                                 cuda.Out(H),
                                 block=blockDim,grid=gridDim,texrefs=[texref])
    else:
        cu_compute_matrix(np.int32(params.pdeg),np.int32(params.sdeg),
                          np.int32(params.bdeg),
                          np.int32(R.shape[1]),np.int32(R.shape[0]),
                          cuda.In(k0),
                          cuda.In(k1),
                          cuda.In(extendedBasis),
                          np.int32(kernelIndex.shape[0]),np.int32(kernelRadius),
                          cuda.Out(H),block=blockDim,grid=gridDim,
                          texrefs=[texref])

    # Fill the elements of V
    blockDim = (256,1,1)
    gridDim = (hs,1,1)
    if params.use_stamps:
        cu_compute_vector_stamps(np.int32(params.pdeg),np.int32(params.sdeg),
                                 np.int32(params.bdeg),
                                 np.int32(R.shape[1]),np.int32(R.shape[0]),np.int32(params.nstamps),
                                 np.int32(params.stamp_half_width),
                                 cuda.In(posx),
                                 cuda.In(posy),
                                 cuda.In(k0),
                                 cuda.In(k1),
                                 cuda.In(extendedBasis),
                                 np.int32(kernelIndex.shape[0]),np.int32(kernelRadius),
                                 cuda.Out(V),block=blockDim,grid=gridDim,
                                 texrefs=[texref])
    else:
        cu_compute_vector(np.int32(params.pdeg),np.int32(params.sdeg),
                          np.int32(params.bdeg),
                          np.int32(R.shape[1]),np.int32(R.shape[0]),
                          cuda.In(k0),
                          cuda.In(k1),
                          cuda.In(extendedBasis),
                          np.int32(kernelIndex.shape[0]),np.int32(kernelRadius),
                          cuda.Out(V),block=blockDim,grid=gridDim,
                          texrefs=[texref])
    return H, V, texref


def compute_model_cuda(image_size,texref,c,kernelIndex,extendedBasis,params):
    
    # Import CUDA function to perform the convolution
    cu_compute_model = cu_matrix_kernel.get_function('cu_compute_model')

    # Create a numpy array for the model M
    M = np.zeros(image_size).astype(np.float32).copy()
    
    # Call the cuda function to perform the convolution
    blockDim = (256,1,1)
    gridDim = (image_size[1],image_size[0])+(1,)
    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    cu_compute_model(np.int32(params.pdeg),np.int32(params.sdeg),
                     np.int32(params.bdeg),cuda.In(k0),
                     cuda.In(k1), cuda.In(extendedBasis),
                     np.int32(kernelIndex.shape[0]),
                     cuda.In(c),cuda.Out(M),
                     block=blockDim,grid=gridDim,texrefs=[texref])
    return M


def photom_all_stars(diff,inv_variance,positions,psf_image,c,kernelIndex,
                     extendedBasis,kernelRadius,params,
                     star_group_boundaries=None,
                     detector_mean_positions_x=None,detector_mean_positions_y=None,star_sky=None):
    
    from astropy.io import fits
    # Read the PSF
    psf,psf_hdr = fits.getdata(psf_image,0,header='true')
    print 'CIF psf_shape',psf.shape
    print 'CIF psf_sum = ',np.sum(psf)
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
        print 'psf_parameters',psf_parameters
        profile_type = 1
    else:
        print 'params.psf_profile_type undefined'
        sys.exit(0)
    
    # Copy the difference and inverse variance images into GPU texture memory
    RR = np.array([diff,inv_variance]).astype(np.float32).copy()
    diff_cuda = numpy3d_to_array(RR)
    texref = cu_matrix_kernel.get_texref("tex")
    texref.set_array(diff_cuda)
    texref.set_filter_mode(cuda.filter_mode.POINT)
    
    # Call the CUDA function to perform the photometry.
    # Each block is one star.
    # Each thread is one column of the PSF, but 32 threads per warp
    nstars = positions.shape[0]
    gridDim = (int(nstars),1,1)
    blockDim = (16,16,1)

    k0 = kernelIndex[:,0].astype(np.int32).copy()
    k1 = kernelIndex[:,1].astype(np.int32).copy()
    positions = positions.reshape(-1,2)
    if params.star_file_is_one_based:
        posx = np.float32(positions[:,0].copy()-1.0)
        posy = np.float32(positions[:,1].copy()-1.0)
    else:
        posx = np.float32(positions[:,0].copy())
        posy = np.float32(positions[:,1].copy())


    #psf_0 = convolve_undersample(psf[0]).astype(np.float32).copy()
    #psf_xd = convolve_undersample(psf[1]).astype(np.float32).copy()*0.0
    #psf_yd = convolve_undersample(psf[2]).astype(np.float32).copy()*0.0
    #psf_0 = psf[0].astype(np.float32).copy()
    #psf_xd = psf[1].astype(np.float32).copy()*0.0
    #psf_yd = psf[2].astype(np.float32).copy()*0.0
    psf_0 = psf.astype(np.float32).copy()
    psf_xd = psf.astype(np.float32).copy()*0.0
    psf_yd = psf.astype(np.float32).copy()*0.0
    flux = np.float32(posy.copy() * 0.0);
    dflux = np.float32(posy.copy() * 0.0);
    star_sky = np.float32(star_sky);

    cu_photom = cu_matrix_kernel.get_function('cu_photom')

    try:
        cu_photom(np.int32(profile_type),np.int32(diff.shape[0]), np.int32(diff.shape[1]),
                  np.int32(params.pdeg),
                  np.int32(params.sdeg),np.int32(c.shape[0]),np.int32(kernelIndex.shape[0]),
                  np.int32(kernelRadius),cuda.In(k0),
                  cuda.In(k1),cuda.In(extendedBasis),
                  cuda.In(psf_parameters),cuda.In(psf_0),cuda.In(psf_xd),cuda.In(psf_yd),
                  cuda.In(posx),cuda.In(posy),cuda.In(c),cuda.Out(flux),cuda.Out(dflux),cuda.In(star_sky),
                  block=blockDim,grid=gridDim,
                  texrefs=[texref])
    except:
        print 'Call to cu_photom failed.'
        print 'psf_parameters', psf_parameters
        print 'size of posx, posy:', posx.shape, posy.shape
        print 'Parameters:'
        for par in dir(params):
            print par, getattr(params, par)
        print

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
    
    # Copy the images into GPU texture memory
    nx, ny = image1.shape
    RR = np.array([image1,image2]).astype(np.float32).copy()
    image_cuda = numpy3d_to_array(RR)
    texref = cu_matrix_kernel.get_texref("tex")
    texref.set_array(image_cuda)
    texref.set_filter_mode(cuda.filter_mode.POINT)

    # Call the CUDA function to perform the double convolution.
    # Each block is one image section.
    # Each thread is one pixel of the PSF, but 32 threads per warp

    cu_convolve = cu_matrix_kernel.get_function('convolve_image_psf')

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
    gridDim = (int((nx-1)/image_section_size+1),int((ny-1)/image_section_size+1),1)
    blockDim = (16,16,1)

    cu_convolve(np.int32(profile_type),np.int32(nx), np.int32(ny),
                np.int32(image_section_size),
                np.int32(image_section_size), np.int32(params.pdeg),
                np.int32(params.sdeg),np.int32(c.shape[0]),
                np.int32(kernelIndex.shape[0]),np.int32(kernelRadius),cuda.In(k0),
                cuda.In(k1),cuda.In(extendedBasis),cuda.In(psf_parameters),cuda.In(psf_0),
                cuda.In(psf_xd),cuda.In(psf_yd),cuda.In(c),cuda.Out(convolved_image1),
                cuda.Out(convolved_image2),block=blockDim,grid=gridDim,texrefs=[texref])

    return convolved_image1, convolved_image2





