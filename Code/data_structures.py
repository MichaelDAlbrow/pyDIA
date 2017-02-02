import os
import numpy as np
import image_functions as im
import io_functions as io

#
# Fundamental data structures
#

class EmptyBase(object): pass


class Observation(object):
    "Container for all observation attributes"
    
    def get_data(self):
        if not(isinstance(self._data,np.ndarray)):
            self._data, _ = io.read_fits_file(self.fullname)
            self.data_median = np.median(self._data)
            self.shape = self._data.shape
        return self._data
    
    def set_data(self,value):
        self._data = value

    def del_data(self):
        self._data = None

    data = property(get_data,set_data,del_data)


    def get_image(self):
        if not(isinstance(self._image,np.ndarray)):
            image_name = os.path.join(self.output_dir,'r_'+self.name)
            self._image, _ = io.read_fits_file(image_name)
        return self._image

    def set_image(self,value):
        self._image = value
        image_name = os.path.join(self.output_dir,'r_'+self.name)
        io.write_image(self._image,image_name)

    def del_image(self):
        self._image = None

    image = property(get_image,set_image,del_image)

    
    def get_mask(self):
        if not(isinstance(self._mask,np.ndarray)):
            mask_name = os.path.join(self.output_dir,'sm_'+self.name)
            self._mask, _ = io.read_fits_file(mask_name)
        return self._mask

    def set_mask(self,value):
        self._mask = value
        mask_name = os.path.join(self.output_dir,'sm_'+self.name)
        io.write_image(self._mask,mask_name)

    def del_mask(self):
        self._mask = None

    mask = property(get_mask,set_mask,del_mask)

    
    def get_inv_variance(self):
        if not(isinstance(self._inv_variance,np.ndarray)):
            inv_variance_name = os.path.join(self.output_dir,'sm_'+self.name)
            self._inv_variance, _ = io.read_fits_file(inv_variance_name)
        return self._inv_variance

    def set_inv_variance(self,value):
        self._inv_variance = value
        inv_variance_name = os.path.join(self.output_dir,'iv_'+self.name)
        io.write_image(self._inv_variance,inv_variance_name)

    def del_inv_variance(self):
        self._inv_variance = None

    inv_variance = property(get_inv_variance,set_inv_variance,del_inv_variance)

    
    def __init__(self,filename,params):
        self.fullname = filename
        self.name = os.path.basename(filename)
        self.output_dir = params.loc_output
        self._data = None
        self._image = None
        self._mask = None
        self.mask = im.compute_saturated_pixel_mask(self.data,5,params) * \
                    im.compute_bleed_mask(self.data,3,params)
        self.inv_variance = 1.0/(self.data/params.gain +
                                (params.readnoise/params.gain)**2) + self.mask
        if params.subtract_sky:
            self.data = im.subtract_sky(self.data,params)
        self.fw, self.sky, self.signal = -1.0, -1.0, -1.0
        if 20 < self.data_median < 0.5*params.pixel_max:
            self.fw, self.sky, self.signal = im.compute_fwhm(self,params,
                                                             seeing_file=
                                                             params.loc_output+
                                                             os.path.sep+'seeing')
        del self.mask
        del self.inv_variance

    def register(self,reg,params):
        print self.name
        self._image, self._mask, self._inv_variance = im.register(reg,self,
                                                                  params)
        rf = os.path.join(self.output_dir,'r_'+self.name)
        io.write_image(self._image,rf)
        rf = os.path.join(self.output_dir,'sm_'+self.name)
        io.write_image(self._mask,rf)
        rf = os.path.join(self.output_dir,'iv_'+self.name)
        io.write_image(self._inv_variance,rf)
        del self.mask
        del self.data
        del self.inv_variance


class Parameters:
    "Container for parameters"
    def __init__(self):
        self.bdeg = 0
        self.ccd_group_size = 100
        self.cluster_mask_radius = 50
        self.datekey = 'MJD-OBS'
        self.detect_threshold = 4.0
        self.diff_std_threshold = 10.0
        self.do_photometry = True
        self.fft_kernel_threshold = 3.0
        self.fwhm_mult = 6.5
        self.fwhm_section = None
        self.gain = 1.0
        self.image_list_file = 'images'
        self.iterations = 1
        self.kernel_maximum_radius = 20.0
        self.kernel_minimum_radius = 5.0
        self.loc_data = '.'
        self.loc_output = '.'
        self.mask_cluster = False
        self.min_ref_images = 3
        self.n_parallel = 1
        self.name_pattern = '*.fits'
        self.nstamps = 200
        self.pdeg = 0
        self.pixel_max = 50000
        self.pixel_min = 0.0
        self.pixel_rejection_threshold = 3.0
        self.psf_fit_radius = 3.0
        self.psf_profile_type = 'gaussian'
        self.readnoise = 1.0
        self.ref_image_list = 'ref.images'
        self.ref_include_file = None
        self.ref_exclude_file = None
        self.reference_min_seeing = 1.3
        self.reference_seeing_factor = 1.01
        self.reference_sky_factor = 1.3
        self.registration_image = None
        self.sdeg = 0
        self.sky_degree = 0
        self.sky_subtract_mode = 'percent'
        self.sky_subtract_percent = 0.01
        self.stamp_edge_distance = 40
        self.stamp_half_width = 20
        self.star_detect_sigma = 12
        self.star_file = None
        self.star_file_has_magnitudes = False
        self.star_file_is_one_based = True
        self.star_file_number_match = 10000
        self.star_file_transform_degree = 2
        self.star_reference_image = None
        self.subtract_sky = True
        self.use_fft_kernel_pixels = False
        self.use_GPU = True
        self.use_stamps = False


