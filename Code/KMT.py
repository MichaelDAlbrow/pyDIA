import sys
import os
import numpy as np

import DIA_GPU as DIA
import calibration_functions as cal
import c_interface_functions as CF

def process_KMT_patch(site,coords=None,quality_max=1.25,mag_err_max=0.3,seeing_max=7.0,sky_max=10000,name_pattern_has_site=True,date_header='MIDHJD',
						parameters=None,q_sigma_threshold=1.0,locate_date_range=None,locate_half_width=None):

	params = DIA.DS.Parameters()
	params.gain = 1.5
	params.readnoise = 15.0
	params.pixel_max = 57000
	params.datekey = date_header
	params.use_GPU = True
	params.n_parallel = 1
	params.pdeg = 1
	params.sdeg = 1
	params.bdeg = 1
	params.reference_seeing_factor = 1.01
	params.reference_min_seeing = 1.0
	params.loc_data = 'RAW'
	params.fwhm_mult = 10

	if parameters is not None:
		for par, value in parameters.iteritems():
			try:
				exec('params.'+par+' = '+str(value))
			except NameError:
				exec('params.'+par+' = \''+str(value)+'\'')

	lightcurve_header='      Date   Delta_Flux Err_Delta_Flux     Mag  Err_Mag        Q    FWHM Roundness      Sky    Signal'

	min_ref_images = {'I':20,'V':3}

	median_mag = {}

	for band in ['I','V']:

		prefix = site+band
		params.loc_output = prefix

		params.ref_include_file = prefix+'-ref.list'

		if not(os.path.exists(params.ref_include_file)):
			params.ref_include_file = False

		if band == 'V':
			params.star_file = site+'I/ref.mags'
			params.star_reference_image = site+'I/ref.fits'
			params.registration_image = site+'I/ref.fits'

		if name_pattern_has_site:
			params.name_pattern = site+'*'+band+'*.fits'
		else:
			params.name_pattern = '*'+band+'*.fits'

		params.min_ref_images = min_ref_images[band]

		if not(os.path.exists(params.loc_output)):
			DIA.imsub_all_fits(params)

		if not(os.path.exists(params.loc_output+'/calibration.png')):
			cal.calibrate(params.loc_output)

		if coords is not None:

			if band == 'I':

				x0, y0 = coords
				print 'Starting photometry for',prefix,'at', (x0,y0)
				dates, seeing, roundness, bgnd, signal, flux, dflux, quality, x0, y0 = \
						CF.photom_variable_star(x0,y0,params,save_stamps=True,patch_half_width=20,locate_date_range=locate_date_range,
												locate_half_width=locate_half_width)
				print 'Converged to', (x0,y0)

			else:

				Imags = np.loadtxt(site+'I/ref.mags')
				Vmags = np.loadtxt(site+'V/ref.mags')
				x0 += np.median(Vmags[:,1]-Imags[:,1])
				y0 += np.median(Vmags[:,2]-Imags[:,2])
				print 'Starting photometry for',prefix,'at', (x0,y0)
				dates, seeing, roundness, bgnd, signal, flux, dflux, quality, x0, y0 = \
						CF.photom_variable_star(x0,y0,params,save_stamps=True,patch_half_width=20,converge=False)
				print 'Converged to', (x0,y0)

			print 'Photometry for', site, band, 'at', x0, y0

			refmags = np.loadtxt(params.loc_output+'/ref.mags.calibrated')
			refflux = np.loadtxt(params.loc_output+'/ref.flux.calibrated')
		
			star_dist2 = (refmags[:,1]-x0)**2 + (refmags[:,2]-y0)**2
			star_num = np.argmin(star_dist2)
		
			print 'x0 y0:', x0, y0
			print 'Nearest star', star_num, 'located at', refmags[star_num,1], refmags[star_num,2]
			print 'Reference flux', refflux[star_num,:]
			print 'Reference mag', refmags[star_num,:]

			mag = 25 - 2.5*np.log10(refflux[star_num,0] + flux)
			mag_err = 25 - 2.5*np.log10(refflux[star_num,0] + flux - dflux) - mag 

			np.savetxt(prefix+'-lightcurve.dat',np.vstack((dates,flux,dflux,mag,mag_err,quality,seeing,roundness,bgnd,signal)).T, \
					fmt='%12.5f  %12.4f  %12.4f  %7.4f  % 7.4f  %6.2f  %6.2f  %5.2f  %10.2f  %8.2f', \
					header=lightcurve_header)

			q = np.where( (quality < quality_max) & (mag_err < mag_err_max) & (seeing < seeing_max) & (bgnd < sky_max) )[0]
			median_mag[band] = np.nanmedian(mag[q])

			np.savetxt(prefix+'-lightcurve-filtered.dat',np.vstack((dates[q],flux[q],dflux[q], \
					mag[q],mag_err[q],quality[q],seeing[q],roundness[q],bgnd[q],signal[q])).T, \
					fmt='%12.5f  %12.4f  %12.4f  %7.4f  % 7.4f  %6.2f  %6.2f  %5.2f  %10.2f  %8.2f', \
					header=lightcurve_header)
			
			cal.plot_lightcurve(prefix+'-lightcurve.dat',plotfile=prefix+'-lightcurve.png')
			cal.plot_lightcurve(prefix+'-lightcurve-filtered.dat',plotfile=prefix+'-lightcurve-filtered.png')

		if site in ['A','SSOre']:
			return

	RC = cal.makeCMD(site+'I',site+'V')

	VI, VI_err = cal.source_colour(site+'I-lightcurve-filtered.dat',site+'V-lightcurve-filtered.dat',plotfile=site+'-source-colour.png')

	cal.makeCMD(site+'I',site+'V',plot_density=False,IV=(median_mag['I'],median_mag['V']),RC=RC,source_colour=(VI,VI_err))



