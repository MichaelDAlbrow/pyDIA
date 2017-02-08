import sys
import os

#
# Change this to your pyDIA install location
#
sys.path.append('/Users/mda45/Projects/pyDIA/Test/pyDIA-master')

use_GPU = True

if use_GPU:
    from Code import DIA_GPU as DIA
else:
    from Code import DIA_CPU as DIA

params = DIA.Parameters()
params.use_GPU = use_GPU

from Code import calibration_functions as cal

params.n_parallel = 1
params.gain = 1.9
params.readnoise = 5.0
params.name_pattern = 'A*.fits'
params.datekey = 'PHJDMID'
params.pdeg = 1
params.sdeg = 1
params.bdeg = 1

#rparams.use_stamps = True
#rparams.nstamps = 100
#rparams.stamp_half_width = 15

params.loc_data = 'Images'
params.loc_output = 'Output-gpu'

DIA.imsub_all_fits(params)
cal.calibrate(params.loc_output)

