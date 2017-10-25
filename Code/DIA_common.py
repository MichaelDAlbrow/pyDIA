#  pyDIA
#
#  This software implements the difference-imaging algorithm of Bramich et al. (2010)
#  with mixed-resolution delta basis functions. It uses an NVIDIA GPU to do the heavy
#  processing. 
#
# Subroutines deconvolve3_rows, deconvolve3_columns, resolve_coeffs_2d and
# interpolate_2d are taken from the Gwiddion software for scanning probe
# microscopy (http://gwyddion.net/), which is distributed under the GNU General
# Public License.
#
# All remaining code is Copyright (C) 2014, 2015  Michael Albrow
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import os
import time
import fnmatch
import itertools
from multiprocessing import Pool

import numpy as np
import data_structures as DS
import io_functions as IO 
import image_functions as IM
import photometry_functions as PH

import c_interface_functions as CIF

def difference_image(ref,target,params,stamp_positions=None,
					 psf_image=None,star_positions=None,
					 star_group_boundaries=None,detector_mean_positions_x=None,
					 detector_mean_positions_y=None,star_sky=None):

	from scipy.linalg import lu_solve, lu_factor, LinAlgError

	start = time.time()
	print 'difference_image', ref.name, target.name

	#
	# Set the kernel size based on the difference in seeing from the reference
	#
	#kernelRadius = min(params.kernel_maximum_radius,
	#                   max(params.kernel_minimum_radius,
	#                       np.abs(target.fw-ref.fw)*params.fwhm_mult))
	kernelRadius = min(params.kernel_maximum_radius,
					   max(params.kernel_minimum_radius,
						   np.sqrt(np.abs(target.fw**2-ref.fw**2))*params.fwhm_mult))

	#
	# Mask saturated pixels
	#
	#print 'Masking ',target.name,time.time()-start
	#smask = compute_saturated_pixel_mask(target.image,kernelRadius,params)


	#
	# Define the kernel basis functions
	#
	print 'Defining kernel pixels',time.time()-start
	if params.use_fft_kernel_pixels:
		kernelIndex, extendedBasis = IM.define_kernel_pixels_fft(ref,target,kernelRadius+2,INNER_RADIUS=20,threshold=params.fft_kernel_threshold)
	else:
		kernelIndex, extendedBasis = IM.define_kernel_pixels(kernelRadius)
	nKernel = kernelIndex.shape[0]

	#
	# We dont want to use bad pixels in either the target or reference image
	#
	smask = target.mask * ref.mask
	bmask = np.ones(smask.shape,dtype=bool)

	g = DS.EmptyBase()

	for iteration in range(params.iterations):
	
		print 'Computing matrix',time.time()-start

		tmask = bmask * smask

		#
		# Compute the matrix and vector
		#
		H, V, texref = CI.compute_matrix_and_vector_cuda(ref.image,ref.blur,
														target.image,target.inv_variance,
													  	tmask,kernelIndex,
													  	extendedBasis,
													  	kernelRadius,
													  	params,
													  	stamp_positions=stamp_positions)
		

		#
		# Solve the matrix equation to find the kernel coefficients
		#
		print 'Solving matrix equation', time.time()-start
		try:
			lu, piv = lu_factor(H)
			c = lu_solve((lu,piv),V).astype(np.float32).copy()
		except (LinAlgError,ValueError):
			print 'LU decomposition failed'
			g.model = None
			g.flux = None
			g.diff = None
			print 'H'
			print H
			sys.stdout.flush()
			return g

		#
		# Compute the model image
		#
		print 'Computing model',time.time()-start
		g.model = CI.compute_model_cuda(ref.image.shape,texref,c,kernelIndex,extendedBasis,params)

		#
		# Compute the difference image
		#
		difference = (target.image - g.model)
		g.norm = difference*np.sqrt(target.inv_variance)

		#
		# Recompute the variance image from the model
		#
		target.inv_variance = 1.0/(g.model/params.gain +
								   (params.readnoise/params.gain)**2) + (1-smask)
		mp = np.where(tmask == 0)
		if len(mp[0]) > 0:
			target.inv_variance[mp] = 1.e-12

		#
		# Mask pixels that disagree with the model
		#
		if iteration > 2:
			bmask = IM.kappa_clip(smask,g.norm,params.pixel_rejection_threshold)
			
		print 'Iteration',iteration,'completed',time.time()-start

	#
	# Delete the target image array to save memory
	#
	del target.image

	#
	# Save the kernel coefficients to a file
	#
	if params.do_photometry and psf_image:
		kf = params.loc_output+os.path.sep+'k_'+os.path.basename(target.name)
		IO.write_kernel_table(kf,kernelIndex,extendedBasis,c,params)

	g.norm = difference*np.sqrt(target.inv_variance)
	g.variance = 1.0/target.inv_variance
	g.mask = tmask

	#
	# Do the photometry if requested
	#
	g.flux = None
	if params.do_photometry and psf_image:
		print 'star_positions', star_positions.shape
		print 'star_group_boundaries', star_group_boundaries
		if ref.name == target.name:
			sky_image, _ = IO.read_fits_file(params.loc_output+os.path.sep+'temp.sub2.fits')
			phot_target = ref.image - sky_image
			g.flux, g.dflux = CIF.photom_all_stars_simultaneous(phot_target,target.inv_variance,star_positions,
													psf_image,c,kernelIndex,extendedBasis,kernelRadius,params,
                                 					star_group_boundaries,
                                  					detector_mean_positions_x,detector_mean_positions_y)
 		else:
			phot_target = difference
			g.flux, g.dflux = CI.photom_all_stars(phot_target,target.inv_variance,star_positions,
													psf_image,c,kernelIndex,extendedBasis,
													kernelRadius,params,
													star_group_boundaries,
													detector_mean_positions_x,
													detector_mean_positions_y)

		print 'Photometry completed',time.time()-start

	#
	# Apply the photometric scale factor to the difference image.
	# We don't do this prior to the photometry because the PSF is
	# being convolved by the kernel, which already includes the
	# photometric scale factor.
	#
	g.diff = IM.apply_photometric_scale(difference,c,params.pdeg)
	sys.stdout.flush()
	return g



def process_reference_image(f,args):
	best_seeing_ref, params, stamp_positions = args
	result = difference_image(f,best_seeing_ref,params,
								stamp_positions=stamp_positions)
	del f.image
	del f.mask
	del f.inv_variance
	return result


def process_reference_image_helper(args):
	return process_reference_image(*args)



def make_reference(files,reg,params,reference_image='ref.fits'):
	seeing = {}
	sky = {}
	ref_seeing = 1000

	#
	# Have we specified the files to make the reference with?
	#
	if params.ref_include_file:

		ref_list = []
		for line in open(params.ref_include_file,'r'):
			for f in files:
				if f.name == line.split()[0]:
					ref_list.append(f)
					print f.name, f.fw, f.signal
					if f.fw < ref_seeing:
						ref_sky = f.sky
						ref_seeing = f.fw
						best_seeing_ref = f

	else:

		#
		# We try to choose the best images
		#
		reference_exclude = []
		if params.ref_exclude_file:
			for line in open(params.ref_exclude_file,'r'):
				reference_exclude.append(line.split()[0])

		sig = []
		for f in files:
			sig.append(f.signal)

		sig = np.asarray(sig)
		sigcut = np.mean(sig) - np.std(sig)
		print 'signal: mean, std, cut = ',np.mean(sig),np.std(sig),sigcut


		print 'Searching for best-seeing image'
		for f in files:
			print f.name, f.fw, f.sky, f.signal
			if (f.fw < ref_seeing) and (f.fw > params.reference_min_seeing) and \
						(f.roundness < params.reference_max_roundness) and (f.signal > sigcut) and not(f.name in reference_exclude):
				ref_sky = f.sky
				ref_seeing = f.fw
				best_seeing_ref = f

		ref_list = []
		while len(ref_list) < params.min_ref_images:
			ref_list = []
			print 'Reference FWHM = ',ref_seeing
			print 'Cutoff FWHM for reference = ',params.reference_seeing_factor*ref_seeing
			print 'Combining for reference:'
			for f in files:
				if (f.fw < params.reference_seeing_factor*ref_seeing) and \
								(f.roundness < params.reference_max_roundness) and (f.sky < params.reference_sky_factor*ref_sky) and \
								(f.fw > params.reference_min_seeing) and (f.signal > sigcut) and not(f.name in reference_exclude):
					ref_list.append(f)
					print f.name, f.fw, f.sky, f.signal
			params.reference_seeing_factor *= 1.02

		sig = []
		for f in ref_list:
			sig.append(f.signal)
		sig = np.asarray(sig)
		sigcut = np.mean(sig) - np.std(sig)
		print 'signal: mean, std, cut = ',np.mean(sig),np.std(sig),sigcut
	   
		ref_seeing = 1000
		ref_roundness = 2.0
		for f in ref_list:
			if (f.roundness < ref_roundness) and (f.signal > sigcut):
				ref_sky = f.sky
				ref_seeing = f.fw
				ref_roundness = f.roundness
				best_seeing_ref = f


	#
	# Which ref image has the worst seeing?
	#
	worst_seeing = 0.0
	for f in ref_list:
		if f.fw > worst_seeing:
			worst_seeing = f.fw
			worst_seeing_ref = f


	if params.ref_image_list:
		with open(params.loc_output+os.path.sep+params.ref_image_list,'w') as fid:
			for f in ref_list:
				fid.write(f.name+'  '+str(f.fw)+'  '+str(f.sky)+'  '+str(f.signal)+'\n')

	#
	# Find the locations of the brightest stars to use as stamp positions
	# if required
	#
	stamp_positions = None
	if params.use_stamps:
		stars = PH.choose_stamps(best_seeing_ref,params)
		stamp_positions = stars[:,0:2]


	#
	#  Construct the reference image.
	#
	ref = np.zeros([1,1])
	sum1 = 0
	sum2 = 0

	good_ref_list = []
	
	for f in ref_list:
		f.blur = IM.boxcar_blur(f.image)
		good_ref_list.append(f)
		print 'difference_image:',f.name,best_seeing_ref.name

	if not(params.use_GPU) and (params.n_parallel > 1):
		
		#
		# Use ParallelProcessing to process images in the reference list
		#
	
		pool = Pool(params.n_parallel)
		results = pool.map(process_reference_image_helper,
						   itertools.izip(ref_list,
										  itertools.repeat((best_seeing_ref,
															params,
															stamp_positions))))

		for i, f in enumerate(ref_list):
			f.result = results[i]

	else:

		for f in ref_list:
			f.result = process_reference_image(f,(best_seeing_ref,params,
												  stamp_positions))

	#
	# Remove bad reference models
	#

	rlist = [g for g in good_ref_list]
	for g in rlist:
		if not(isinstance(g.result.diff,np.ndarray)):
			print 'removing',g.name
			good_ref_list.remove(g)

	print 'good reference list:'
	for g in good_ref_list:
		print g.name

	print 'kappa-clipping reference list'
	for iterations in range(5):
		if len(good_ref_list) < 4:
			break
		sd = np.zeros(len(good_ref_list))
		for i, g in enumerate(good_ref_list):
			print g.name, g.result.diff
			sd[i] = np.std(g.result.diff)
		sds = sd.std()
		sdm = sd.mean()
		rlist = [g for g in good_ref_list]
		for g in rlist:
			if np.std(g.result.diff) > (sdm + 2.5*sds):
				print 'removing',g.name
				good_ref_list.remove(g)

	#
	# Combine the good reference models
	#
	g = good_ref_list[0]
	gstack = np.zeros([len(good_ref_list),g.result.model.shape[0],g.result.model.shape[1]])
	print 'final reference list'
	for i,g in enumerate(good_ref_list):
		if isinstance(g.result.model,np.ndarray):
			print g.name, np.std(g.result.diff), np.median(g.result.model)
			IO.write_image(g.result.model,params.loc_output+os.path.sep+'mr_'+g.name)
			gstack[i,:,:] = g.result.model
	rr = np.median(gstack,axis=0)
	IO.write_image(rr,params.loc_output+os.path.sep+reference_image)

	for f in ref_list:
		f.result = None

	return stamp_positions



def process_image(f,args):
	ref,params,stamp_positions,star_positions,star_group_boundaries,star_unsort_index, detector_mean_positions_x,detector_mean_positions_y = args
	dtarget = params.loc_output+os.path.sep+'d_'+f.name
	if not(os.path.exists(dtarget)):
			
		#
		# Compute difference image
		#
		result = difference_image(ref,f,
								  params,
								  stamp_positions=stamp_positions,
								  psf_image=params.loc_output+os.path.sep+
								  'psf.fits',
								  star_positions=star_positions,
								  star_group_boundaries=star_group_boundaries,
								  detector_mean_positions_x=detector_mean_positions_x,
								  detector_mean_positions_y=detector_mean_positions_y)
		del f.image
		del f.mask
		del f.inv_variance

		#
		# Save photometry to a file
		#
		if isinstance(result.flux,np.ndarray):
			if not(params.use_GPU):
				print 'ungrouping fluxes'
				result.flux = result.flux[star_unsort_index].copy()
				result.dflux = result.dflux[star_unsort_index].copy()
			np.savetxt(params.loc_output+os.path.sep+f.name+'.flux',
					   np.vstack((result.flux,result.dflux)).T)
			f.flux = result.flux.copy()
			f.dflux = result.dflux.copy()
				
		#
		# Save output images to files
		#
		if isinstance(result.diff,np.ndarray):
			IO.write_image(result.diff,params.loc_output+os.path.sep+'d_'+f.name)
			IO.write_image(result.model,params.loc_output+os.path.sep+'m_'+f.name)
			IO.write_image(result.norm,params.loc_output+os.path.sep+'n_'+f.name)
			IO.write_image(result.mask,params.loc_output+os.path.sep+'z_'+f.name)
	return 0


def process_image_helper(args):
	return process_image(*args)




def imsub_all_fits(params,reference='ref.fits'):

	#
	# Create the output directory if it doesn't exist
	#
	if not (os.path.exists(params.loc_output)):
		os.mkdir(params.loc_output)

	#
	# The degree of spatial shape changes has to be at least as
	# high as the degree of spatial photometric scale
	#
	if (params.sdeg < params.pdeg):
		print 'Increasing params.sdeg to ',params.pdeg
		params.sdeg = params.pdeg

	#
	# Print out the parameters for this run.
	#
	print 'Parameters:'
	for par in dir(params):
		print par, getattr(params, par)
	print



	#
	# Determine our list of images
	#
	all_files = os.listdir(params.loc_data)
	all_files.sort()
	files = []
	for f in all_files:
		if fnmatch.fnmatch(f,params.name_pattern):
			g = DS.Observation(params.loc_data+os.path.sep+f,params)
			del g.data
			del g.mask
			if g.fw > 0.0:
			   files.append(g)
			print g.name

	if len(files) < 3:
		print 'Only',len(files),'files found matching',params.name_pattern
		print 'Exiting'
		sys.exit(0)

	#
	# Have we specified a registration template?
	#
	if params.registration_image:
		reg = DS.Observation(params.registration_image,params)
	else:
		reg = DS.EmptyBase()
		reg.fw = 999.0
		for f in files:
			if (f.fw < reg.fw) and (f.fw > 1.2):
				reg = f

	print 'Registration image:',reg.name


	#
	# Register images
	#
	for f in files:
		if f == reg:
			f.image = f.data
			rf = params.loc_output+os.path.sep+'r_'+f.name
			IO.write_image(f.image,rf)
		else:
			f.register(reg,params)
			# delete image arrays to save memory
			del f.image
			del f.mask
			del f.inv_variance
		del reg.data
		del reg.image
		del reg.mask
		del reg.inv_variance


	#
	# Write image names and dates to a file
	#
	if params.image_list_file:
		try:
			with open(params.loc_output+os.path.sep+params.image_list_file,'w') as fid:
				for f in files:
					date = None
					if params.datekey:
						date = IO.get_date(params.loc_data+os.path.sep+f.name,
											key=params.datekey)-2450000
					if date:
						fid.write(f.name+'   %10.5f\n'%date)
					else:
						fid.write(f.name)
		except:
			raise

	#
	# Make the photometric reference image if we don't have it.
	# Find stamp positions if required.
	#
	if not(os.path.exists(params.loc_output+os.path.sep+reference)):
		print 'Reg = ',reg.name
		stamp_positions = make_reference(files,reg,params,reference_image=reference)
		ref = DS.Observation(params.loc_output+os.path.sep+reference,params)
		ref.register(reg,params)
	else:
		ref = DS.Observation(params.loc_output+os.path.sep+reference,params)
		ref.register(reg,params)
		stamp_positions = None
		if params.use_stamps:
			stamp_file = params.loc_output+os.path.sep+'stamp_positions'
			if os.path.exists(stamp_file):
				stamp_positions = np.genfromtxt(stamp_file)
			else:
				stars = PF.choose_stamps(ref,params)
				stamp_positions = stars[:,0:2]
				np.savetxt(stamp_file,stamp_positions)

	pm = params.pixel_max
	params.pixel_max *= 0.9
	ref.mask = IM.compute_saturated_pixel_mask(ref.image,4,params)
	params.pixel_max = pm
	ref.blur = IM.boxcar_blur(ref.image)
	if params.mask_cluster:
		ref.mask = IM.mask_cluster(ref.image,ref.mask,params)

	#
	# Detect stars and compute the PSF if we are doing photometry
	#
	star_positions = None
	sky = 0.0
	if params.do_photometry:
		star_file = params.loc_output+os.path.sep+'star_positions'
		psf_file = params.loc_output+os.path.sep+'psf.fits'
		if not(os.path.exists(psf_file)) or not(os.path.exists(star_file)):
			stars = PH.compute_psf_image(params,ref,psf_image=psf_file)
			star_positions = stars[:,0:2]
			star_sky = stars[:,4]
		if os.path.exists(star_file):
			star_positions = np.genfromtxt(star_file)
			star_sky = star_positons[:,0]*0.0;
		else:
			np.savetxt(star_file,star_positions)

	print 'sky =', sky

	#
	# If we have pre-determined star positions
	#
	#if params.star_file:
	#	stars = np.genfromtxt(params.star_file)
	#	star_positions = stars[:,1:3]
	#	if params.star_reference_image:
	#		star_ref, h = IO.read_fits_file(params.star_reference_image)
	#		dy, dx = IM.positional_shift(ref.image,star_ref)
	#		print 'position shift =',dx,dy
	#		star_positions[:,0] += dx
	#		star_positions[:,1] += dy
	#	np.savetxt(star_file,star_positions)

	#
	# If we are using a CPU, group the stars by location
	#
	print 'Group_check'
	print 'params.do_photometry',params.do_photometry
	print 'params.use_GPU',params.use_GPU
	if params.do_photometry:
		star_group_boundaries = None
		detector_mean_positions_x = None
		detector_mean_positions_y = None
		star_unsort_index = None
		star_sort_index,star_group_boundaries,detector_mean_positions_x,detector_mean_positions_y = \
							PH.group_stars_ccd(params,star_positions,params.loc_output+os.path.sep+reference)
		star_positions = star_positions[star_sort_index]
		star_sky = star_sky[star_sort_index]
		star_unsort_index = np.argsort(star_sort_index)
			

	#
	# Do photometry of the reference image
	#
	if params.do_photometry:
		ref_flux_file = params.loc_output+os.path.sep+'ref.flux'
		if not(os.path.exists(ref_flux_file)):
			result = difference_image(ref,ref,params,
									  stamp_positions=stamp_positions,
									  psf_image=psf_file,
									  star_positions=star_positions,
									  star_group_boundaries=star_group_boundaries,
									  detector_mean_positions_x=detector_mean_positions_x,
									  detector_mean_positions_y=detector_mean_positions_y,
									  star_sky=star_sky)
			if isinstance(result.flux,np.ndarray):
				print 'ungrouping fluxes'
				result.flux = result.flux[star_unsort_index].copy()
				result.dflux = result.dflux[star_unsort_index].copy()
				np.savetxt(ref_flux_file,
						   np.vstack((result.flux,result.dflux)).T)

	#
	# Process images
	#

	if params.make_difference_images:

		if not(params.use_GPU) and (params.n_parallel > 1):

			pool = Pool(params.n_parallel)
			pool.map(process_image_helper, itertools.izip(files, itertools.repeat(
					(ref,params,stamp_positions,star_positions,star_group_boundaries,
					 star_unsort_index,detector_mean_positions_x,detector_mean_positions_y))))

		else:

			for f in files:
				process_image(f,(ref,params,stamp_positions,star_positions,
								 star_group_boundaries,star_unsort_index,
								 detector_mean_positions_x,detector_mean_positions_y))

	return files


def do_photometry(params,extname='newflux',star_file='star_positions',
				  psf_file='psf.fits',star_positions=None,reference_image='ref.fits'):

	#
	# Determine our list of files
	#
	all_files = os.listdir(params.loc_data)
	all_files.sort()
	files = []
	for f in all_files:
		if fnmatch.fnmatch(f,params.name_pattern):
			g = DS.Observation(params.loc_data+os.path.sep+f,params)
			if g.fw > 0.0:
			   files.append(g)
			   
	ref = DS.Observation(params.loc_output+os.path.sep+reference_image,params)
	ref.register(ref,params)

	#
	# Detect stars and compute the PSF if necessary
	#
	if params.do_photometry:
		psf_file = params.loc_output+os.path.sep+psf_file
		if os.path.exists(params.star_file):
			star_pos = np.genfromtxt(params.star_file)[:,1:3]
			if not(os.path.exists(psf_file)):
				stars = PH.compute_psf_image(params,ref,psf_image=psf_file)
		else:
			if not(os.path.exists(star_file)):
				stars = PH.compute_psf_image(params,ref,psf_image=psf_file)
				star_pos = stars[:,0:2]
				np.savetxt(star_file,star_pos)
			else:
				star_pos = np.genfromtxt(star_file)
				if not(os.path.exists(psf_file)):
					stars = PH.compute_psf_image(params,ref,psf_image=psf_file)

	#
	# Have we been passed an array of star positions?
	#
	if star_positions == None:
		star_positions = star_pos

	#
	# If we are using a CPU, group the stars by location
	#
	star_group_boundaries = None
	detector_mean_positions_x = None
	detector_mean_positions_y = None
	if not(params.use_GPU):
		star_sort_index,star_group_boundaries,detector_mean_positions_x,detector_mean_positions_y = \
					PH.group_stars_ccd(params,star_positions,params.loc_output+os.path.sep+reference_image)
		star_positions = star_positions[star_sort_index]
		star_unsort_index = np.argsort(star_sort_index)

	#
	# Process the reference image
	#
	print 'Processing',reference_image
	ref = DS.Observation(params.loc_output+os.path.sep+reference_image,params)
	#reg = Observation(params.loc_data+os.path.sep+
	#                  params.registration_image,params)
	ref.register(ref,params)
	smask = IM.compute_saturated_pixel_mask(ref.image,6,params)
	ref.inv_variance += 1 - smask
	ktable = params.loc_output+os.path.sep+'k_'+os.path.basename(reference_image)
	kernelIndex, extendedBasis, c, params = IO.read_kernel_table(ktable,params)
	kernelRadius = np.max(kernelIndex[:,0])+1
	if np.sum(extendedBasis) > 0:
		kernelRadius += 1
	print 'kernelIndex', kernelIndex
	print 'extendedBasis',extendedBasis
	print 'coeffs', c
	print 'kernelRadius',kernelRadius
	phot_target = ref.image
	ref.flux, ref.dflux = PH.photom_all_stars(phot_target,ref.inv_variance,star_positions,
												psf_file,c,kernelIndex,extendedBasis,
										   		kernelRadius,
										   		params,
										   		star_group_boundaries,
										   		detector_mean_positions_x,
										   		detector_mean_positions_y, sky=sky)

	if isinstance(ref.flux,np.ndarray):
		if not(params.use_GPU):
			print 'ungrouping fluxes'
			ref.flux = ref.flux[star_unsort_index].copy()
			ref.dflux = ref.dflux[star_unsort_index].copy()
		np.savetxt(params.loc_output+os.path.sep+reference_image+'.'+extname,
				   np.vstack((ref.flux,ref.dflux)).T)

	#
	# Process difference images
	#
	for f in files:

		if not(os.path.exists(params.loc_output+os.path.sep+f.name+'.'+extname)):

			print 'Processing',f.name
			target = f.name
			dtarget = params.loc_output+os.path.sep+'d_'+os.path.basename(target)
			ntarget = params.loc_output+os.path.sep+'n_'+os.path.basename(target)
			ztarget = params.loc_output+os.path.sep+'z_'+os.path.basename(target)
			ktable = params.loc_output+os.path.sep+'k_'+os.path.basename(target)
			
			if os.path.exists(dtarget) and os.path.exists(ntarget) and os.path.exists(ktable):

				norm, h = IO.read_fits_file(ntarget)
				diff, h = IO.read_fits_file(dtarget)
				mask, h = IO.read_fits_file(ztarget)
				inv_var = (norm/diff)**2 + (1-mask)

				kernelIndex, extendedBasis, c, params = IO.read_kernel_table(ktable,params)
				kernelRadius = np.max(kernelIndex[:,0])+1
				if np.sum(extendedBasis) > 0:
					kernelRadius += 1

				print 'kernelIndex', kernelIndex
				print 'extendedBasis',extendedBasis
				print 'coeffs', c
				print 'kernelRadius',kernelRadius

				diff = IM.undo_photometric_scale(diff,c,params.pdeg)
				
				flux, dflux = PH.photom_all_stars(diff,inv_var,star_positions,
											   		psf_file,c,kernelIndex,extendedBasis,
											   		kernelRadius,params,
											   		star_group_boundaries,
											   		detector_mean_positions_x,
											   		detector_mean_positions_y)

				if isinstance(flux,np.ndarray):
					if not(params.use_GPU):
						print 'ungrouping fluxes'
						flux = flux[star_unsort_index].copy()
						dflux = dflux[star_unsort_index].copy()
					np.savetxt(params.loc_output+os.path.sep+f.name+'.'+extname,
							   np.vstack((flux,dflux)).T)


