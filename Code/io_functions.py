import os
from astropy.io import fits
import numpy as np

def get_date(file,key='JD'):
    target = file
    hdulist = fits.open(target)
    try:
        date = hdulist[0].header[key]
    except KeyError:
        date = None
    return date


def read_fits_file(file,slice=None):
    if slice:
        f = fits.open(file,memmap=True)
        data = f[0].section[slice[2]:slice[3],slice[0]:slice[1]]
        hdr = f[0].header
    else:
        data, hdr = fits.getdata(file, header=True)
    return np.float64(data), hdr


def write_image(image,file):
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    try:
        hdu.writeto(file,overwrite=True)
    except IOError:
        print 'Warning - io_functions.write_image: could not write file',file
        pass

def write_kernel_table(file,kernel_index,extended_basis,coeffs,params):
    if os.path.exists(file):
        os.remove(file)
    table1 = fits.TableHDU.from_columns([  \
        fits.Column(name='x',format='I',array=kernel_index[:,0]), \
        fits.Column(name='y',format='I',array=kernel_index[:,1]), \
        fits.Column(name='extended',format='I5',array=extended_basis)])
    table2 = fits.TableHDU.from_columns([  \
        fits.Column(name='Spatial type',format='A4',array=np.array(['PDEG','SDEG','BDEG'])), \
        fits.Column(name='degree',format='I',array=np.array([params.pdeg,params.sdeg,params.bdeg]))])
    table3 = fits.TableHDU.from_columns(  \
        [fits.Column(name='Coefficients',format='E',array=coeffs)])
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList([hdu,table1,table2,table3])
    hdulist.writeto(file)


def read_kernel_table(file,params):
    hdulist = fits.open(file)
    t = hdulist[1].data
    k1 = t.field('x')
    k2 = t.field('y')
    extended_basis = t.field('extended')
    kernel_index = np.array([k1,k2]).T
    t = hdulist[2].data
    deg = t.field('degree')
    t = hdulist[3].data
    coeffs = t.field('Coefficients')
    if ((params.pdeg != deg[0]) or (params.sdeg != deg[1]) or
        (params.bdeg != deg[2])):
        print 'Warning: kernel degrees in',file,'do not match current parameters'
        params.pdeg = deg[0]
        params.sdeg = deg[1]
        params.bdeg = deg[2]
    return kernel_index, extended_basis, coeffs, params

        


