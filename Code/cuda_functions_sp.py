import pycuda.autoinit
from pycuda.compiler import SourceModule


cu_matrix_kernel = SourceModule("""

#include <math.h>
#include <stdio.h>

#include "texture_fetch_functions.h"
#include "texture_types.h"

#define THREADS_PER_BLOCK 256
#define FIT_RADIUS 6

texture<float, cudaTextureType2DLayered, cudaReadModeElementType> tex;


__device__ void deconvolve3_columns(int width,int height,int rowstride,
                          float *data,float *buffer,float a,float b) {
    float *row;
    float q;
    int i, j;

/*
//    if (( height < 2) || (rowstride > width)) {
//      printf("Failure in deconvolve3_rows: height, rowstride, width, a, b = %//d %d %d %f %f\n",height, rowstride, width, a, b );
//		return;
//    }
*/


    if (!height || !width)
        return;
	
    if (height == 1) {
        q = a + 2.0*b;
        for (j = 0; j < width; j++)
            data[j] /= q;
        return;
    }
    if (height == 2) {
        q = a*(a + 2.0*b);
        for (j = 0; j < width; j++) {
            buffer[0] = (a + b)/q*data[j] - b/q*data[rowstride + j];
            data[rowstride + j] = (a + b)/q*data[rowstride + j] - b/q*data[j];
            data[j] = buffer[0];
        }
        return;
    }
	
    /* Special-case first row */
    buffer[0] = a + b;
    /* Inner rows */
    for (i = 1; i < height-1; i++) {
        q = b/buffer[i-1];
        buffer[i] = a - q*b;
        row = data + (i - 1)*rowstride;
        for (j = 0; j < width; j++)
            row[rowstride + j] -= q*row[j];
    }
    /* Special-case last row */
    q = b/buffer[i-1];
    buffer[i] = a + b*(1.0 - q);
    row = data + (i - 1)*rowstride;
    for (j = 0; j < width; j++)
        row[rowstride + j] -= q*row[j];
    /* Go back */
    row += rowstride;
    for (j = 0; j < width; j++)
        row[j] /= buffer[i];
    do {
        i--;
        row = data + i*rowstride;
        for (j = 0; j < width; j++)
            row[j] = (row[j] - b*row[rowstride + j])/buffer[i];
    } while (i > 0);
}


__device__ void deconvolve3_rows(int width,int height,int rowstride,float *data,
                                 float *buffer,float a,float b) {
    float *row;
    float q;
    int i, j;

/*
//    if (( height < 2) || (rowstride > width)) {
//		printf("Failure in deconvolve3_rows\n");
//		return;
//    }
*/


    if (!height || !width)
        return;
	
    if (width == 1) {
        q = a + 2.0*b;
        for (i = 0; i < height; i++)
            data[i*rowstride] /= q;
        return;
    }
    if (width == 2) {
        q = a*(a + 2.0*b);
        for (i = 0; i < height; i++) {
            row = data + i*rowstride;
            buffer[0] = (a + b)/q*row[0] - b/q*row[1];
            row[1] = (a + b)/q*row[1] - b/q*row[0];
            row[0] = buffer[0];
        }
        return;
    }
	
    /* Special-case first item */
    buffer[0] = a + b;
    /* Inner items */
    for (j = 1; j < width-1; j++) {
        q = b/buffer[j-1];
        buffer[j] = a - q*b;
        data[j] -= q*data[j-1];
    }
    /* Special-case last item */
    q = b/buffer[j-1];
    buffer[j] = a + b*(1.0 - q);
    data[j] -= q*data[j-1];
    /* Go back */
    data[j] /= buffer[j];
    do {
        j--;
        data[j] = (data[j] - b*data[j+1])/buffer[j];
    } while (j > 0);
	
    /* Remaining rows */
    for (i = 1; i < height; i++) {
        row = data + i*rowstride;
        /* Forward */
        for (j = 1; j < width-1; j++)
            row[j] -= b*row[j-1]/buffer[j-1];
        row[j] -= b*row[j-1]/buffer[j-1];
        /* Back */
        row[j] /= buffer[j];
        do {
            j--;
            row[j] = (row[j] - b*row[j+1])/buffer[j];
        } while (j > 0);
    }
}




__device__ void resolve_coeffs_2d(int width, int height, int rowstride, float *data) {
    float *buffer;
    int     max;
	
    max = width > height ? width : height;
    buffer = (float *)malloc(max*sizeof(float));
    deconvolve3_rows(width, height, rowstride, data, buffer, 13.0/21.0, 4.0/21.0);
    deconvolve3_columns(width, height, rowstride, data, buffer, 13.0/21.0, 4.0/21.0);
    free(buffer);
}


__device__ float interpolate_2d(float x,float y,int rowstride,float *coeff) {
    float wx[4], wy[4];
    int i, j;
    float v, vx;

/*
//    if (x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0) {
//		printf("interpolate_2d: x or y out of bounds %f %f\n",x,y);
//		return(-1.0);
//    }
*/    
    wx[0] = 4.0/21.0 + (-11.0/21.0 + (0.5 - x/6.0)*x)*x;
    wx[1] = 13.0/21.0 + (1.0/14.0 + (-1.0 + x/2.0)*x)*x;
    wx[2] = 4.0/21.0 + (3.0/7.0 + (0.5 - x/2.0)*x)*x;
    wx[3] = (1.0/42.0 + x*x/6.0)*x;
    wy[0] = 4.0/21.0 + (-11.0/21.0 + (0.5 - y/6.0)*y)*y;
    wy[1] = 13.0/21.0 + (1.0/14.0 + (-1.0 + y/2.0)*y)*y;
    wy[2] = 4.0/21.0 + (3.0/7.0 + (0.5 - y/2.0)*y)*y;
    wy[3] = (1.0/42.0 + y*y/6.0)*y;
	
    v = 0.0;
    for (i = 0; i < 4; i++) {
        vx = 0.0;
        for (j = 0; j < 4; j++)
            vx += coeff[i*rowstride + j]*wx[j];
        v += wy[i]*vx;
    }
	
    return v;
}



__device__ float integrated_profile(int profile_type, int idx, int idy, float xpos,
                                   float ypos, float *psf_parameters, float *lut_0,
                                   float *lut_xd, float *lut_yd) {

    int psf_size;
    float   psf_height, psf_sigma_x, psf_sigma_y, psf_xpos, psf_ypos;
    float   p0;
    int     ip, jp;
    float  pi=3.14159265,fwtosig=0.8493218;
    
    psf_size = (int) psf_parameters[0];
    psf_height = psf_parameters[1];
    psf_sigma_x = psf_parameters[2];
    psf_sigma_y = psf_parameters[3];
    psf_ypos = psf_parameters[4];
    psf_xpos = psf_parameters[5];

    if (profile_type == 0) {
    
       // gaussian

       // PSF at location (Idx,Idy). PSF is centred at (7.5,7.5)
       // Analytic part

       p0 = 0.5*psf_height*pi*fwtosig*fwtosig*
            (erff((idx-7.5+0.5)/(1.41421356*psf_sigma_x)) - 
             erff((idx-7.5-0.5)/(1.41421356*psf_sigma_x))) *
            (erff((idy-7.5+0.5)/(1.41421356*psf_sigma_y)) - 
             erff((idy-7.5-0.5)/(1.41421356*psf_sigma_y)));

       // Index into the lookup table

       ip = psf_size/2 + 2*idx - 15;
       jp = psf_size/2 + 2*idy - 15;
       if ((ip>=0) && (ip<=psf_size-1) && (jp>=0) && (jp<=psf_size-1)) {
          p0 += lut_0[ip+psf_size*jp] + lut_xd[ip+psf_size*jp]*(xpos-psf_xpos) +
                lut_yd[ip+psf_size*jp]*(ypos-psf_ypos);
       }

       return p0;

    } else if (profile_type == 1) {

       //  moffat25
       //  From iraf/noao/digiphot/daophot/daolib/profile.x

       float d[4][4] = {{ 0.0,         0.0,        0.0,        0.0},
                        {-0.28867513,  0.28867513, 0.0,        0.0},
                        {-0.38729833,  0.0,        0.38729833, 0.0},
                        {-0.43056816, -0.16999052, 0.16999052, 0.43056816}};
       float w[4][4] = {{1.0,         0.0,        0.0,        0.0},
                        {0.5,         0.5,        0.0,        0.0},
                        {0.27777778,  0.44444444, 0.27777778, 0.0},
                        {0.17392742,  0.32607258, 0.32607258, 0.17392742}};

       float alpha = 0.3195079;
       float  p1sq, p2sq, p1p2, dx, dy, xy, denom, func, x[4], xsq[4], p1xsq[4];
       float  y, ysq, p2ysq, wt, p4fod, wp4fod, wf;
       int    npt, ix, iy;

       p1sq = psf_parameters[2]*psf_parameters[2];
       p2sq = psf_parameters[3]*psf_parameters[3];
       p1p2 = psf_parameters[2]*psf_parameters[3];
       dx = idx-7.5+0.5;
       dy = idy-7.5+0.5;
       xy = dx * dy;
       
       denom = 1.0 + alpha * (dx*dx/p1sq + dy*dy/p2sq + xy*psf_parameters[4]);
       if (denom > 1.0e4) {
          return 0.0;
       }

       p0 = 0.0;
       func = 1.0 / (p1p2*powf(float(denom),float(2.5)));
       if (func >= 0.046) {
          npt = 4;
       } else if (func >= 0.0022) {
          npt = 3;
       } else if (func >= 0.0001) {
          npt = 2;
       } else if (func >= 1.0e-10) {
          p0 = (2.5 - 1.0) * func;
       }

       if (func >= 0.0001) {
       
          for (ix=0; ix<npt; ix++) {
             x[ix] = dx + d[npt][ix];
             xsq[ix] = x[ix]*x[ix];
             p1xsq[ix] = xsq[ix]/p1sq;
          }

          for (iy=0; iy<npt; iy++) {
             y = dy + d[npt][iy];
             ysq = y*y;
             p2ysq = ysq/p2sq;
             for (ix=0; ix<npt; ix++) {
                wt = w[npt][iy] * w[npt][ix];
                xy = x[ix] * y;
                denom = 1.0 + alpha * (p1xsq[ix] + p2ysq + xy*psf_parameters[4]);
                func = (2.5 - 1.0) / (p1p2 * powf(denom,2.5) );
                p4fod = 2.5 * alpha * func / denom;
                wp4fod = wt * p4fod;
                wf = wt * func;
                p0 += wf;
             }
          }
          
       }

       p0 *= psf_parameters[1];

       // Index into the lookup table

       ip = psf_size/2 + 2*idx - 15;
       jp = psf_size/2 + 2*idy - 15;
       if ((ip>=0) && (ip<=psf_size-1) && (jp>=0) && (jp<=psf_size-1)) {
           p0 += lut_0[ip+psf_size*jp] + lut_xd[ip+psf_size*jp]*(xpos-psf_xpos) +
                 lut_yd[ip+psf_size*jp]*(ypos-psf_ypos);
       }
       
       return p0;

   } else {

      return 0.0;

   }
   
}


__global__ void convolve_image_psf(int profile_type, int nx, int ny, int dx, int dy,
                          int dp, int ds, int n_coeff, int nkernel,
                          int kernel_radius,int *kxindex,
                          int *kyindex, int* ext_basis, float *psf_parameters,
                          float *psf_0, float *psf_xd, float *psf_yd,
                          float *coeff,float *cim1, float* cim2) {

   int     id, txa, tyb, txag, tybg;
   int     np, ns, i, j, ii, ip, jp, ic, ki, a, b;
   int     d1, sidx, l, m, l1, m1, ig, jg;
   int     psf_size, ix, jx;
   float   x, y, p0, p1, p1g, cpsf_pixel, xpos, ypos;
   float   psf_height, psf_sigma_x, psf_sigma_y, psf_sigma_xy, psf_xpos, psf_ypos;
   float   gain,psf_rad,psf_rad2, px, py;
   float   sx2, sy2, sxy2, sx2msy2, sx2psy2; 
   float  psf_norm,dd;
   float  pi=3.14159265,fwtosig=0.8493218;

   __shared__ float psf_sum[256];
   __shared__ float cpsf[256];
   __shared__ float cpix1[256];
   __shared__ float cpix2[256];


   // initialise memory
   id = threadIdx.x+threadIdx.y*16;
   cpsf[id] = 0.0;

   // star position in normalised units
   xpos = blockIdx.x*dx + dx/2;
   ypos = blockIdx.y*dy + dy/2;
   x = (xpos - 0.5*(nx-1))/(nx-1);
   y = (ypos - 0.5*(ny-1))/(ny-1);


   // number of polynomial coefficients per basis function
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;

   // PSF parameters
   psf_size = (int) psf_parameters[0];
   psf_height = psf_parameters[1];
   psf_sigma_x = psf_parameters[2];
   psf_sigma_y = psf_parameters[3];
   psf_ypos = psf_parameters[4];
   psf_xpos = psf_parameters[5];
   psf_rad = psf_parameters[6];
   gain = psf_parameters[7];
   if (psf_rad > 5.0) {
      psf_rad = 5.0;
   }
   psf_rad2 = psf_rad*psf_rad;


   // PSF integral

   __syncthreads();

   psf_sum[id] = 0.0;
   for (i=threadIdx.x+1; i<psf_size-1; i+=blockDim.x) {
     for (j=threadIdx.y+1; j<psf_size-1; j+=blockDim.y) {
       psf_sum[id] += psf_0[i+j*psf_size];
     }
   }

   __syncthreads();

   i = 128;
   while (i != 0) {
     if (id < i) {
       psf_sum[id] += psf_sum[id + i];
     }
     __syncthreads();
     i /= 2;
   }
   __syncthreads();

   if (profile_type == 0) {
      // gaussian
      psf_norm = 0.25*psf_sum[0] + psf_height*2*pi*fwtosig*fwtosig;
   } else if (profile_type == 1) {
      // moffat25
      psf_sigma_xy = psf_parameters[8];
      sx2 = psf_sigma_x*psf_sigma_x;
      sy2 = psf_sigma_y*psf_sigma_y;
      sxy2 = psf_sigma_xy*psf_sigma_xy;
      sx2msy2 = 1.0/sx2 - 1.0/sy2;
      sx2psy2 = 1.0/sx2 + 1.0/sy2;
      px = 1.0/sqrt( sx2psy2 + sqrt(sx2msy2*sx2msy2 + sxy2) );
      py = 1.0/sqrt( sx2psy2 - sqrt(sx2msy2*sx2msy2 + sxy2) );
      psf_norm = 0.25*psf_sum[0] + psf_height*pi*(px*py)/(psf_sigma_x*psf_sigma_y);
   }
   
   // Construct the convolved PSF

   // PSF at location (Idx,Idy). PSF is centred at (7.5,7.5)
   // Analytic part

   p0 = integrated_profile(profile_type, threadIdx.x, threadIdx.y, xpos, ypos,
                           psf_parameters, psf_0, psf_xd, psf_yd);

   __syncthreads();

   cpsf_pixel = 0.0;
    
   // Iterate over coefficients

   for (ic=0; ic<n_coeff; ic++) {
 
      // basis function position
      ki = ic < np ? 0 : (ic-np)/ns + 1;

      if (ki<nkernel) {
       
        a = kxindex[ki];
        b = kyindex[ki];
 
        // Set the polynomial degree for the subvector and the
        // index within the subvector
        if (ki == 0) {
          d1 = dp;
          sidx = ic;
        } else {
          d1 = ds;
          sidx = ic - np - (ki-1)*ns;
        }
       

        // Compute the polynomial index (l,m) values corresponding
        // to the index within the subvector
        l1 = m1 = 0;
        if (d1 > 0) {
          i = 0;
          for (l=0; l<=d1; l++) {
            for (m=0; m<=d1-l; m++) {
              if (i == sidx) {
                l1 = l;
                m1 = m;
              }
              i++;
            }
          }
        }

        // Indices into the PSF

        if (ki > 0) {

          txa = threadIdx.x + a;
          tyb = threadIdx.y + b;
          
          p1 = integrated_profile(profile_type, txa, tyb, xpos, ypos,
                                  psf_parameters, psf_0, psf_xd, psf_yd);

          __syncthreads();


        // If we have an extended basis function, we need to
        // average the PSF over a 3x3 grid
          if (ext_basis[ki]) {

            p1 = 0.0;
            for (ig=-1; ig<2; ig++) {
              for (jg=-1; jg<2; jg++) {
                txag = txa + ig;
                tybg = tyb + jg;
                               
                p1g = integrated_profile(profile_type, txag, tybg, xpos, ypos,
                                         psf_parameters, psf_0, psf_xd, psf_yd);

                __syncthreads();

                p1 += p1g;
              }
            }
            p1 /= 9.0;
           
          }

          cpsf_pixel += coeff[ic]*(p1-p0)*powf(x,l1)*powf(y,m1);

        } else {

          cpsf_pixel += coeff[ic]*p0*powf(x,l1)*powf(y,m1);

        }
      
     }
       
   } //end ic loop

   __syncthreads();

   cpsf[id] = cpsf_pixel/psf_norm;

   __syncthreads();

   // Now convolve the image section with the convolved PSF

   for (i=xpos-dx/2; i<xpos+dx/2; i++) {
     for (j=ypos-dy/2; j<ypos+dy/2; j++) {
       ix = (int)floorf(i+0.5)+threadIdx.x-8.0;
       jx = (int)floorf(j+0.5)+threadIdx.y-8.0;
       cpix1[id] = cpsf[id]*tex2DLayered(tex,ix,jx,0);
       cpix2[id] = cpsf[id]*tex2DLayered(tex,ix,jx,1);

       __syncthreads();
       
       // Parallel sum
       ii = 128;
       while (ii != 0) {
         if (id < ii) {
           cpix1[id] += cpix1[id + ii];
           cpix2[id] += cpix2[id + ii];
         }
         __syncthreads();
         ii /= 2;
       }

       if (id == 0) {
          cim1[i+j*nx] = cpix1[0];
          cim2[i+j*nx] = cpix2[0];
       }
       
       __syncthreads();

     }
   }

   return;

}




__global__ void cu_photom(int profile_type,
                          int nx, int ny, int dp, int ds, int n_coeff, int nkernel,
                          int kernel_radius,int *kxindex,
                          int *kyindex, int* ext_basis, float *psf_parameters,
                          float *psf_0, float *psf_xd, float *psf_yd,
                          float *posx,
                          float *posy, float *coeff, 
                          float *flux, float *dflux, float *star_sky) {

   int     id, txa, tyb, txag, tybg;
   int     np, ns, i, j, ip, jp, ic, ki, a, b;
   int     d1, sidx, l, m, l1, m1, ig, jg;
   int     psf_size, ix, jx;
   float   x, y, p0, p1, p1g, cpsf_pixel, xpos, ypos, dd;
   float   psf_height, psf_sigma_x, psf_sigma_y, psf_sigma_xy, psf_xpos, psf_ypos;
   float   psf_rad, psf_rad2, gain, fl, inv_var, px, py;
   float   sx2, sy2, sxy2, sx2msy2, sx2psy2; 
   float  subx, suby, psf_norm, bgnd, dr2;
   float  pi=3.14159265,fwtosig=0.8493218, RON=5.0;

   __shared__ float psf_sum[256];
   __shared__ float cpsf[256];
   __shared__ float  mpsf[256];
   __shared__ float  fsum1[256];
   __shared__ float  fsum2[256];
   __shared__ float  fsum3[256];
   __shared__ float  fsum4[256];
   __shared__ float  fsum5[256];


   // initialise memory
   id = threadIdx.x+threadIdx.y*16;
   cpsf[id] = 0.0;
   mpsf[id] = 0.0;

   // star position in normalised units
   xpos = posx[blockIdx.x];
   ypos = posy[blockIdx.x];
   x = (xpos - 0.5*(nx-1))/(nx-1);
   y = (ypos - 0.5*(ny-1))/(ny-1);


   // number of polynomial coefficients per basis function
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;

   // PSF parameters
   psf_size = (int) psf_parameters[0];
   psf_height = psf_parameters[1];
   psf_sigma_x = psf_parameters[2];
   psf_sigma_y = psf_parameters[3];
   psf_ypos = psf_parameters[4];
   psf_xpos = psf_parameters[5];
   psf_rad = psf_parameters[6];
   gain = psf_parameters[7];
   if (psf_rad > 5.0) {
     psf_rad = 5.0;
   }
   psf_rad2 = psf_rad*psf_rad;


   // PSF integral

   __syncthreads();

   psf_sum[id] = 0.0;
   for (i=threadIdx.x; i<psf_size; i+=blockDim.x) {
     for (j=threadIdx.y; j<psf_size; j+=blockDim.y) {
       psf_sum[id] += psf_0[i+j*psf_size];
       //if (blockIdx.x == 120) {
       //   printf("i, j, id, psf_0: %d %d %d %f\\n",i,j,id,psf_0[i+j*psf_size]);
       //}
     }
   }

   __syncthreads();

   i = 128;
   while (i != 0) {
     if (id < i) {
       psf_sum[id] += psf_sum[id + i];
     }
     __syncthreads();
     i /= 2;
   }
   __syncthreads();

   if (profile_type == 0) {
      // gaussian
      psf_norm = 0.25*psf_sum[0] + psf_height*2*pi*fwtosig*fwtosig;
      //if ((id == 0) && (blockIdx.x==120)){
      //   printf("psf_sum0, psf_height, psf_norm: %f %f %f\\n",psf_sum[0],psf_height,psf_norm);
      //}
   } else if (profile_type == 1) {
      // moffat25
      psf_sigma_xy = psf_parameters[8];
      sx2 = psf_sigma_x*psf_sigma_x;
      sy2 = psf_sigma_y*psf_sigma_y;
      sxy2 = psf_sigma_xy*psf_sigma_xy;
      sx2msy2 = 1.0/sx2 - 1.0/sy2;
      sx2psy2 = 1.0/sx2 + 1.0/sy2;
      px = 1.0/sqrt( sx2psy2 + sqrt(sx2msy2*sx2msy2 + sxy2) );
      py = 1.0/sqrt( sx2psy2 - sqrt(sx2msy2*sx2msy2 + sxy2) );
      psf_norm = 0.25*psf_sum[0] + psf_height*pi*(px*py)/(psf_sigma_x*psf_sigma_y);
      //if ((id == 0) && (blockIdx.x==120)){
      //   printf("psf_sum0, psf_height, psf_norm: %f %f %f\\n",psf_sum[0],psf_height, psf_norm);
      //}
   }
   
   
   // Construct the convolved PSF

   // PSF at location (Idx,Idy). PSF is centred at (7.5,7.5)
   // Analytic part

   p0 = integrated_profile(profile_type, threadIdx.x, threadIdx.y, xpos, ypos,
                           psf_parameters, psf_0, psf_xd, psf_yd);

   __syncthreads();

// Spatially variable part 
//
//       +
//               psf_xd[ipsf+psf_size*jpsf]*(xpos-psf_xpos) +
//               psf_yd[ipsf+psf_size*jpsf]*(ypos-psf_ypos);
//       }
//


   cpsf_pixel = 0.0;
    
   // Iterate over coefficients

   for (ic=0; ic<n_coeff; ic++) {
 
      // basis function position
      ki = ic < np ? 0 : (ic-np)/ns + 1;

      if (ki<nkernel) {
       
        a = kxindex[ki];
        b = kyindex[ki];
 
        // Set the polynomial degree for the subvector and the
        // index within the subvector
        if (ki == 0) {
          d1 = dp;
          sidx = ic;
        } else {
          d1 = ds;
          sidx = ic - np - (ki-1)*ns;
        }
       

        // Compute the polynomial index (l,m) values corresponding
        // to the index within the subvector
        l1 = m1 = 0;
        if (d1 > 0) {
          i = 0;
          for (l=0; l<=d1; l++) {
            for (m=0; m<=d1-l; m++) {
              if (i == sidx) {
                l1 = l;
                m1 = m;
              }
              i++;
            }
          }
        }

        // Indices into the PSF

        if (ki > 0) {

          txa = threadIdx.x + a;
          tyb = threadIdx.y + b;

          p1 = integrated_profile(profile_type, txa, tyb, xpos, ypos,
                                  psf_parameters, psf_0, psf_xd, psf_yd);

          __syncthreads();


//
//             +
//                     psf_xd[ipsf+psf_size*jpsf]*(xpos-psf_xpos) +
//                     psf_yd[ipsf+psf_size*jpsf]*(ypos-psf_ypos);
//             }
//

        // If we have an extended basis function, we need to
        // average the PSF over a 3x3 grid
          if (ext_basis[ki]) {

            p1 = 0.0;
            for (ig=-1; ig<2; ig++) {
              for (jg=-1; jg<2; jg++) {
                txag = txa + ig;
                tybg = tyb + jg;
                               
                p1g = integrated_profile(profile_type, txag, tybg, xpos, ypos,
                                  psf_parameters, psf_0, psf_xd, psf_yd);

                __syncthreads();


//
//                   +
//                            psf_xd[ipsf+psf_size*jpsf]*(xpos-psf_xpos) +
//                            psf_yd[ipsf+psf_size*jpsf]*(ypos-psf_ypos);
//                   }
//


                p1 += p1g;
              }
            }
            p1 /= 9.0;
           
          }

          cpsf_pixel += coeff[ic]*(p1-p0)*powf(x,l1)*powf(y,m1);

        } else {

          cpsf_pixel += coeff[ic]*p0*powf(x,l1)*powf(y,m1);

        }
      
     }
       
   } //end ic loop

    __syncthreads();

    cpsf[id] = cpsf_pixel/psf_norm;

    __syncthreads();

/* Uncomment to print convolved PSF   
   if ((id == 0) && (blockIdx.x==14)){
     txa = 7;
     tyb = 7;
     ip = psf_size/2 + 2*txa - 15;
     jp = psf_size/2 + 2*tyb - 15;
     if (profile_type == 0) {
       printf("psf_test: %lf %lf %lf %lf\\n",
               0.5*psf_height*pi*fwtosig*fwtosig*
                 (erff((txa-7.5+0.5)/(1.41421356*psf_sigma_x)) - 
                  erff((txa-7.5-0.5)/(1.41421356*psf_sigma_x))) *
                 (erff((tyb-7.5+0.5)/(1.41421356*psf_sigma_y)) - 
                  erff((tyb-7.5-0.5)/(1.41421356*psf_sigma_y))),
              psf_0[ip+psf_size*jp],
              psf_xd[ip+psf_size*jp]*(xpos-psf_xpos),
              psf_yd[ip+psf_size*jp]*(ypos-psf_ypos));
      }    
             
              

     dd = 0.0;
     printf("cpsf\\n");
     for (j=15; j>=0; j--) {
       printf("%2d ",j);
       for (i=0; i<16; i++) {
         printf("%6.4f  ",cpsf[i+j*16]);
         dd += cpsf[i+j*16];
       }
       printf("\\n");
     }
     printf("sum = %f\\n",dd);
     printf("psf lookup table fraction: %f\\n",psf_sum[0]/psf_norm);
   }
   */


   
   __syncthreads();



   // Map the convolved PSF to the subpixel star coordinates
   
   if (id == 0) {
     resolve_coeffs_2d(16,16,16,cpsf);
   }

   __syncthreads();

   mpsf[id] = 0.0;

   subx = ceilf(xpos+0.5+0.00001) - (xpos+0.5);
   suby = ceilf(ypos+0.5+0.00001) - (ypos+0.5);
   if ((threadIdx.x > 1) && (threadIdx.x < 14) &&
       (threadIdx.y > 1) && (threadIdx.y < 14)) {
      mpsf[id] = interpolate_2d(subx,suby,16,&cpsf[threadIdx.x-2+(threadIdx.y-2)*16]);
   }

   __syncthreads();

   // force negative pixels to zero
   mpsf[id] = mpsf[id] > 0.0 ? mpsf[id] : 0.0;
  
   __syncthreads();

   //
   // Normalise mapped PSF
   //  (No - the convolved PSF contain the phot scale)
/*
   cpsf[id] = mpsf[id];
   __syncthreads();
   i = 128;
   while (i != 0) {
     if (id < i) {
       cpsf[id] += cpsf[id + i];
     }
     __syncthreads();
     i /= 2;
   }
   
   mpsf[id] /= cpsf[0];
*/

/* Uncomment to print mapped PSF */
  if ((id == 0) && (blockIdx.x==14)){
     printf("xpos, ypos: %f %f\\n",xpos,ypos);
     printf("subx, suby: %f %f\\n",subx,suby);
     printf("mpsf\\n");
     dd = 0.0;
     for (j=15; j>=0; j--) {
       printf("%2d ",j);
       for (i=0; i<16; i++) {
         printf("%6.4f  ",mpsf[i+j*16]);
         dd += mpsf[i+j*16];
       }
       printf("\\n");
     }
     printf("sum = %f\\n",dd);
   }  
   __syncthreads();

   

   // Fit the mapped PSF to the difference image to compute an
   // optimal flux estimate.
   // Assume the difference image is in tex(:,:,0)
   // and the inverse variance in tex(:,:,1).
   // We need to iterate to get the variance correct
   //

   fl = 0.0;

   for (j=0; j<3; j++) {

     fsum1[id] = 0.0;
     fsum2[id] = 0.0;
     fsum3[id] = 0.0;

     __syncthreads();

     /*
     if ((id == 0) && (blockIdx.x==14)){
         printf("photom, j=%d\\n",j);
     }
     */
     
     if (powf(threadIdx.x-7.5,2)+powf(threadIdx.y-7.5,2) < psf_rad2) {

        ix = (int)floorf(xpos+0.5)+threadIdx.x-8.0;
        jx = (int)floorf(ypos+0.5)+threadIdx.y-8.0;

        if (j>0) inv_var = 1.0 / ((star_sky[blockIdx.x] + fl * mpsf[id]) / gain + RON*RON);
        dr2 = (xpos-ix)*(xpos-ix) + (ypos-jx)*(ypos-jx);
        inv_var *= max(5.0 / (5.0 + 1.0 / (psf_rad2/dr2 - 1.0) ),0.0);

        //inv_var = 1.0/(1.0/tex2DLayered(tex,ix,jx,1) + fl*mpsf[id]/gain);

        fsum1[id] = mpsf[id]*(tex2DLayered(tex,ix,jx,0)-star_sky[blockIdx.x])*inv_var;
        fsum2[id] = mpsf[id]*mpsf[id]*inv_var;
        fsum3[id] = mpsf[id]; 

        
        if ((blockIdx.x==14)){
           printf("ix jx xpos ypos dr2 mpsf inv_var im: %03d %03d %8.3f %8.3f %6.3f %6.5f %g %g\\n",ix,jx,xpos,ypos,dr2, mpsf[id],inv_var, tex2DLayered(tex,ix,jx,0));
        }
        


     }

     __syncthreads();
   
   
     // Parallel sum
     i = 128;
     while (i != 0) {
       if (id < i) {
         fsum1[id] += fsum1[id + i];
         fsum2[id] += fsum2[id + i];
         fsum3[id] += fsum3[id + i];
       }
       __syncthreads();
       i /= 2;
     }

     fl = fsum1[0]/fsum2[0];

   }
     
   if (id == 0) {
     flux[blockIdx.x] = fl;
     dflux[blockIdx.x] = sqrt(fsum3[0]*fsum3[0]/fsum2[0]);
   }

/* Uncomment for debug info */
/*
   __syncthreads();
   i = 128;
   while (i != 0) {
     if (id < i) {
       mpsf[id] += mpsf[id + i];
     }
     __syncthreads();
     i /= 2;
   }
   __syncthreads();

   if (id == 0) {
     if (blockIdx.x == 120) {
       printf("result: %f %f %f %f %f %f %f %f %f %f %f %f\\n",fsum1[0],fsum2[0],fsum3[0],mpsf[0],psf_norm,psf_sum[0],bgnd,flux[blockIdx.x],flux[blockIdx.x]*fsum3[0],flux[blockIdx.x]*mpsf[0],fsum4[0],dflux[blockIdx.x]);
     }
   }
*/

   __syncthreads();

   return;

}


__global__ void cu_compute_model(int dp, int ds, int db, int *kxindex,
                int *kyindex, int* ext_basis, int nkernel, float *coefficient,
                float *M) {

   int  np, ns, nb, hs, idx, ki, a, b, d1, sidx, l, m, l1, m1, i;
   float x, y, Bi;

   __shared__ float count[THREADS_PER_BLOCK];

   // Calculate number of terms in subvectors
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;
   nb = (db+1)*(db+2)/2;
   hs = (nkernel-1)*ns+np+nb;

   x = (blockIdx.x - 0.5*(gridDim.x-1))/(gridDim.x-1);
   y = (blockIdx.y - 0.5*(gridDim.y-1))/(gridDim.y-1);

   count[threadIdx.x] = 0.0;

   for (idx = threadIdx.x; idx < hs; idx += blockDim.x) {

     // This is the index of the subvector and its kernel offsets
     ki = idx < np ? 0 : (idx-np)/ns + 1;
     a = b = 0;
     if (ki<nkernel) {
       a = kxindex[ki];
       b = kyindex[ki];
     }
   
     // Set the polynomial degree for the subvector and the
     // index within the subvector
     if (ki == 0) {
       d1 = dp;
       sidx = idx;
     } else if (ki < nkernel) {
       d1 = ds;
       sidx = idx - np - (ki-1)*ns;
     } else {
       d1 = db;
       sidx = idx - np - (ki-1)*ns;
     }

     // Compute the (l,m) values corresponding to the index within
     // the subvector
     l1 = m1 = 0;
     if (d1 > 0) {
       i = 0;
       for (l=0; l<=d1; l++) {
         for (m=0; m<=d1-l; m++) {
           if (i == sidx) {
             l1 = l;
             m1 = m;
           }
           i++;
         }
       }
     }

     if (ki == 0) {
       Bi = tex2DLayered(tex,blockIdx.x,blockIdx.y,0);
     } else if (ki < nkernel) {
       if (ext_basis[ki]) {
         Bi = tex2DLayered(tex,blockIdx.x+a,blockIdx.y+b,1)-
                tex2DLayered(tex,blockIdx.x,blockIdx.y,0);
       } else {
         Bi =  tex2DLayered(tex,blockIdx.x+a,blockIdx.y+b,0)-
                tex2DLayered(tex,blockIdx.x,blockIdx.y,0);
       }
     } else {
       Bi = 1.0;
     }
     
     count[threadIdx.x] += coefficient[idx]*powf(x,l1)*powf(y,m1)*Bi;


   }

   __syncthreads();

   // Then parallel-sum the results
   i = blockDim.x/2;
   while (i != 0) {
     if (threadIdx.x < i) {
       count[threadIdx.x] += count[threadIdx.x + i];
     }
     __syncthreads();
     i /= 2;
   }
   if (threadIdx.x == 0) {
     M[blockIdx.x+gridDim.x*blockIdx.y] = count[0];
   }

}


__global__ void cu_compute_vector(int dp, int ds, int db, int nx,
                int ny, int *kxindex, int *kyindex, int *ext_basis, int nkernel,
                int kernelRadius,float *V) {

   int idx; 
   int np, ns, ki, a, b, d1, i, j;
   int l, m, l1, m1;
   float py, x, y, Bi;
   float temp;
   
    __shared__ float count[THREADS_PER_BLOCK];

   // Calculate number of terms in subvectors
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;

   // This is the index of the subvector and its kernel offsets
   ki = blockIdx.x < np ? 0 : (blockIdx.x-np)/ns + 1;
   a = b = 0;
   if (ki<nkernel) {
     a = kxindex[ki];
     b = kyindex[ki];
   }
   
   // Set the polynomial degrees for the submatrix and the
   // indices within the submatrix
   if (ki == 0) {
     d1 = dp;
     idx = blockIdx.x;
   } else if (ki < nkernel) {
     d1 = ds;
     idx = blockIdx.x - np - (ki-1)*ns;
   } else {
     d1 = db;
     idx = blockIdx.x - np - (ki-1)*ns;
   }

   // Compute the (l,m) values corresponding to the index within
   // the subvector
   i = 0;
   for (l=0; l<=d1; l++) {
     for (m=0; m<=d1-l; m++) {
       if (i == idx) {
         l1 = l;
         m1 = m;
       }
       i++;
     }
   }   

   // Compute the contribution to V from each image location.
   // Use individual threads to sum over columns.
   // tex[:,:,0] is the reference image,
   // tex[:,:,1] is the blurred reference image,
   // tex[:,:,2] is the target image,
   // tex[:,:,3] is the inverse variance,
   // tex[:,:,4] is the mask.
   // Bi is the basis image value.
   temp = 0.0;
   Bi = 1.0;
   __syncthreads();
   for (j=kernelRadius; j<ny-kernelRadius; j++) {
     y = (j - 0.5*(ny-1))/(ny-1);
     py = powf(y,m1);
     for (i=threadIdx.x+kernelRadius; i<nx-kernelRadius; i+=blockDim.x) {
         x = (i - 0.5*(nx-1))/(nx-1);
         if (ki == 0) {
           Bi = tex2DLayered(tex,i,j,0);
         } else if (ki < nkernel) {
           if (ext_basis[ki]) {
             Bi = tex2DLayered(tex,i+a,j+b,1)-tex2DLayered(tex,i,j,0);
           } else {
             Bi = tex2DLayered(tex,i+a,j+b,0)-tex2DLayered(tex,i,j,0);
           }
         } else {
           Bi = 1.0;
         }
         temp += powf(x,l1)*py*Bi*tex2DLayered(tex,i,j,2)*tex2DLayered(tex,i,j,3)*
                 tex2DLayered(tex,i,j,4);
     }
   }

   count[threadIdx.x] = temp;

   __syncthreads();

   // Then parallel-sum the rows
   i = blockDim.x/2;
   while (i != 0) {
     if (threadIdx.x < i) {
       count[threadIdx.x] += count[threadIdx.x + i];
     }
     __syncthreads();
     i /= 2;
   }
   if (threadIdx.x == 0) {
     V[blockIdx.x] = count[0];
   }

}



__global__ void cu_compute_vector_stamps(int dp, int ds, int db, int nx,
                int ny, int nstamps, int stamp_half_width, float *stamp_xpos, float* stamp_ypos,
                int *kxindex, int *kyindex, int *ext_basis, int nkernel,
                int kernelRadius,float *V) {

   int idx; 
   int np, ns, ki, a, b, d1, i, j, i1, i2, j1, j2;
   int l, m, l1, m1;
   float py, x, y, Bi;
   float temp;
   
    __shared__ float count[THREADS_PER_BLOCK];

   // Calculate number of terms in subvectors
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;

   // This is the index of the subvector and its kernel offsets
   ki = blockIdx.x < np ? 0 : (blockIdx.x-np)/ns + 1;
   a = b = 0;
   if (ki<nkernel) {
     a = kxindex[ki];
     b = kyindex[ki];
   }
   
   // Set the polynomial degrees for the submatrix and the
   // indices within the submatrix
   if (ki == 0) {
     d1 = dp;
     idx = blockIdx.x;
   } else if (ki < nkernel) {
     d1 = ds;
     idx = blockIdx.x - np - (ki-1)*ns;
   } else {
     d1 = db;
     idx = blockIdx.x - np - (ki-1)*ns;
   }

   // Compute the (l,m) values corresponding to the index within
   // the subvector
   i = 0;
   for (l=0; l<=d1; l++) {
     for (m=0; m<=d1-l; m++) {
       if (i == idx) {
         l1 = l;
         m1 = m;
       }
       i++;
     }
   }   

   // Compute the contribution to V from each image location.
   // Use individual threads to sum over columns.
   // tex[:,:,0] is the reference image,
   // tex[:,:,1] is the blurred reference image,
   // tex[:,:,2] is the target image,
   // tex[:,:,3] is the inverse variance,
   // tex[:,:,4] is the mask.
   // Bi is the basis image value.
   temp = 0.0;
   Bi = 1.0;
   __syncthreads();
   for (idx = threadIdx.x; idx<nstamps; idx += blockDim.x) {
     j1 = max(0,(int)stamp_ypos[idx]-stamp_half_width);
     j2 = min(ny,(int)stamp_ypos[idx]+stamp_half_width);
     for (j=j1; j<j2; j++) {
       y = (j - 0.5*(ny-1))/(ny-1);
       py = powf(y,m1);
       i1 = max(0,(int)stamp_xpos[idx]-stamp_half_width);
       i2 = min(nx,(int)stamp_xpos[idx]+stamp_half_width);
       for (i=i1; i<i2; i++) {
         x = (i - 0.5*(nx-1))/(nx-1);
         if (ki == 0) {
           Bi = tex2DLayered(tex,i,j,0);
         } else if (ki < nkernel) {
           if (ext_basis[ki]) {
             Bi = tex2DLayered(tex,i+a,j+b,1)-tex2DLayered(tex,i,j,0);
           } else {
             Bi = tex2DLayered(tex,i+a,j+b,0)-tex2DLayered(tex,i,j,0);
           }
         } else {
           Bi = 1.0;
         }
         temp += powf(x,l1)*py*Bi*tex2DLayered(tex,i,j,2)*tex2DLayered(tex,i,j,3)*
                 tex2DLayered(tex,i,j,4);
       }
     }
   }

   count[threadIdx.x] = temp;

   __syncthreads();

   // Then parallel-sum the rows
   i = blockDim.x/2;
   while (i != 0) {
     if (threadIdx.x < i) {
       count[threadIdx.x] += count[threadIdx.x + i];
     }
     __syncthreads();
     i /= 2;
   }
   if (threadIdx.x == 0) {
     V[blockIdx.x] = count[0];
   }

}




__global__ void cu_compute_matrix(int dp, int ds, int db, int nx,
                int ny, int *kxindex, int *kyindex, int *ext_basis, int nkernel,
                int kernelRadius,float *H) {

   int idx, idy, idx0, idy0, idx1, idy1; 
   int np, ns, ki, kj, a, b, c, d, d1, d2, i, j;
   int l, m, l1, m1, l2, m2;
   float py, x, y, Bi, Bj;
   float temp;
   
   __shared__ float count[THREADS_PER_BLOCK];


   // Terminate if we are not in the lower triangle
   if (blockIdx.x > blockIdx.y) {
     return;
   }

   // Calculate number of terms in submatrices
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;

   // These are indices of the submatrix and their kernel offsets
   ki = blockIdx.x < np ? 0 : (blockIdx.x-np)/ns + 1;
   kj = blockIdx.y < np ? 0 : (blockIdx.y-np)/ns + 1;


   a = b = 0;
   if (ki<nkernel) {
     a = kxindex[ki];
     b = kyindex[ki];
   }
   if (kj<nkernel) {
     c = kxindex[kj];
     d = kyindex[kj];
   }
   
   // Set the polynomial degrees for the submatrix and the
   // indices within the submatrix
   if (ki == 0) {
     d1 = dp;
     idx = blockIdx.x;
   } else if (ki < nkernel) {
     d1 = ds;
     idx = blockIdx.x - np - (ki-1)*ns;
   } else {
     d1 = db;
     idx = blockIdx.x - np - (ki-1)*ns;
   }
   if (kj == 0) {
     d2 = dp;
     idy = blockIdx.y;
   } else if (kj < nkernel) {
     d2 = ds;
     idy = blockIdx.y - np - (kj-1)*ns;
   } else {
     d2 = db;
     idy = blockIdx.y - np - (kj-1)*ns;
   }


   if ((ki>0) && (ki<nkernel) && (kj>0) && (kj<nkernel) && (idx > idy)) {
     return;
   }

   idx0 = idx;
   idy0 = idy;

   // Compute the (l,m) values corresponding to the indices within
   // the submatrix
   i = 0;
   for (l=0; l<=d1; l++) {
     for (m=0; m<=d1-l; m++) {
       if (i == idx) {
         l1 = l;
         m1 = m;
       }
       i++;
     }
   }   
   i = 0;
   for (l=0; l<=d2; l++) {
     for (m=0; m<=d2-l; m++) {
       if (i == idy) {
         l2 = l;
         m2 = m;
       }
       i++;
     }
   }

   // Compute the contribution to H from each image location.
   // Use individual threads to sum over columns.
   // tex[:,:,0] is the reference image,
   // tex[:,:,1] is the blurred reference image,
   // tex[:,:,2] is the target image,
   // tex[:,:,3] is the inverse variance,
   // tex[:,:,4] is the mask.
   // Bi and Bj are the basis image values.
   temp = 0.0;
   Bi = Bj = 1.0;
   __syncthreads();
   for (j=kernelRadius; j<ny-kernelRadius; j++) {
     y = (j - 0.5*(ny-1))/(ny-1);
     py = powf(y,m1+m2);
     for (i=threadIdx.x+kernelRadius; i<nx-kernelRadius; i+=blockDim.x) {
         x = (i - 0.5*(nx-1))/(nx-1);
         if (ki == 0) {
           Bi = tex2DLayered(tex,i,j,0);
         } else if (ki < nkernel) {
           if (ext_basis[ki]) {
             Bi = tex2DLayered(tex,i+a,j+b,1)-tex2DLayered(tex,i,j,0);
           } else {
             Bi = tex2DLayered(tex,i+a,j+b,0)-tex2DLayered(tex,i,j,0);
           }
         } else {
           Bi = 1.0;
         }
         if (kj == 0) {
           Bj = tex2DLayered(tex,i,j,0);
         } else if (kj < nkernel) {
           if (ext_basis[kj]) {
             Bj = tex2DLayered(tex,i+c,j+d,1)-tex2DLayered(tex,i,j,0);
           } else {
             Bj = tex2DLayered(tex,i+c,j+d,0)-tex2DLayered(tex,i,j,0);
           }
         } else {
           Bj = 1.0;
         }
         temp += powf(x,l1+l2)*py*Bi*Bj*tex2DLayered(tex,i,j,3)*tex2DLayered(tex,i,j,4);
     }
   }

   count[threadIdx.x] = temp;

   __syncthreads();

   // Then parallel-sum the rows
   i = blockDim.x/2;
   while (i != 0) {
     if (threadIdx.x < i) {
       count[threadIdx.x] += count[threadIdx.x + i];
     }
     __syncthreads();
     i /= 2;
   }

   if (threadIdx.x == 0) {

     H[blockIdx.x+gridDim.x*blockIdx.y] = count[0];
     H[blockIdx.y+gridDim.x*blockIdx.x] = count[0];
     if ((ki>0) && (ki<nkernel) && (kj>0) && (kj<nkernel)) {
       idx1 = np + (ki-1)*ns;
       idy1 = np + (kj-1)*ns;
       H[(idx1+idy0)+gridDim.x*(idy1+idx0)] = count[0];
       H[(idy1+idx0)+gridDim.x*(idx1+idy0)] = count[0];
     }
   }

}



__global__ void cu_compute_matrix_stamps(int dp, int ds, int db, int nx,
                int ny, int nstamps, int stamp_half_width, float *stamp_xpos, float* stamp_ypos,
                int *kxindex, int *kyindex, int *ext_basis, int nkernel,
                int kernelRadius,float *H) {

   int idx, idy, idx0, idy0, idx1, idy1; 
   int np, ns, ki, kj, a, b, c, d, d1, d2, i, j, i1, i2, j1, j2;
   int l, m, l1, m1, l2, m2;
   float px, py, x, y, Bi, Bj;
   float temp;
   
    __shared__ float count[THREADS_PER_BLOCK];

   // Terminate if we are not in the lower triangle
   if (blockIdx.x > blockIdx.y) {
     return;
   }

   // Calculate number of terms in submatrices
   np = (dp+1)*(dp+2)/2;
   ns = (ds+1)*(ds+2)/2;

   // These are indices of the submatrix and their kernel offsets
   ki = blockIdx.x < np ? 0 : (blockIdx.x-np)/ns + 1;
   kj = blockIdx.y < np ? 0 : (blockIdx.y-np)/ns + 1;

   a = b = 0;
   if (ki<nkernel) {
     a = kxindex[ki];
     b = kyindex[ki];
   }
   if (kj<nkernel) {
     c = kxindex[kj];
     d = kyindex[kj];
   }
   
   // Set the polynomial degrees for the submatrix and the
   // indices within the submatrix
   if (ki == 0) {
     d1 = dp;
     idx = blockIdx.x;
   } else if (ki < nkernel) {
     d1 = ds;
     idx = blockIdx.x - np - (ki-1)*ns;
   } else {
     d1 = db;
     idx = blockIdx.x - np - (ki-1)*ns;
   }
   if (kj == 0) {
     d2 = dp;
     idy = blockIdx.y;
   } else if (kj < nkernel) {
     d2 = ds;
     idy = blockIdx.y - np - (kj-1)*ns;
   } else {
     d2 = db;
     idy = blockIdx.y - np - (kj-1)*ns;
   }

   if ((ki>0) && (ki<nkernel) && (kj>0) && (kj<nkernel) && (idx > idy)) {
     return;
   }

   idx0 = idx;
   idy0 = idy;

   // Compute the (l,m) values corresponding to the indices within
   // the submatrix
   i = 0;
   for (l=0; l<=d1; l++) {
     for (m=0; m<=d1-l; m++) {
       if (i == idx) {
         l1 = l;
         m1 = m;
       }
       i++;
     }
   }   
   i = 0;
   for (l=0; l<=d2; l++) {
     for (m=0; m<=d2-l; m++) {
       if (i == idy) {
         l2 = l;
         m2 = m;
       }
       i++;
     }
   }

   // Compute the contribution to H from each image location.
   // Use individual threads to sum over stamps.
   // tex[:,:,0] is the reference image,
   // tex[:,:,1] is the blurred reference image,
   // tex[:,:,2] is the target image,
   // tex[:,:,3] is the inverse variance,
   // tex[:,:,4] is the mask.
   // Bi and Bj are the basis image values.
   temp = 0.0;
   Bi = Bj = 1.0;
   __syncthreads();


   for (idx = threadIdx.x; idx<nstamps; idx += blockDim.x) {
     i1 = max(0,(int)stamp_xpos[idx]-stamp_half_width);
     i2 = min(nx,(int)stamp_xpos[idx]+stamp_half_width);
     for (i=i1; i<i2; i++) {
       x = (i - 0.5*(nx-1))/(nx-1);
       px = powf(x,l1+l2);
       j1 = max(0,(int)stamp_ypos[idx]-stamp_half_width);
       j2 = min(ny,(int)stamp_ypos[idx]+stamp_half_width);
       for (j=j1; j<j2; j++) {
         y = (j - 0.5*(ny-1))/(ny-1);
         py = powf(y,m1+m2);
         if (ki == 0) {
           Bi = tex2DLayered(tex,i,j,0);
         } else if (ki < nkernel) {
           if (ext_basis[ki]) {
             Bi = tex2DLayered(tex,i+a,j+b,1)-tex2DLayered(tex,i,j,0);
           } else {
             Bi = tex2DLayered(tex,i+a,j+b,0)-tex2DLayered(tex,i,j,0);
           }
         } else {
           Bi = 1.0;
         }
         if (kj == 0) {
           Bj = tex2DLayered(tex,i,j,0);
         } else if (kj < nkernel) {
           if (ext_basis[kj]) {
             Bj = tex2DLayered(tex,i+c,j+d,1)-tex2DLayered(tex,i,j,0);
           } else {
             Bj = tex2DLayered(tex,i+c,j+d,0)-tex2DLayered(tex,i,j,0);
           }
         } else {
           Bj = 1.0;
         }
         temp += px*py*Bi*Bj*tex2DLayered(tex,i,j,3)*tex2DLayered(tex,i,j,4);
       }
     }
   }

   count[threadIdx.x] = temp;

   __syncthreads();

   // Then parallel-sum the rows
   i = blockDim.x/2;
   while (i != 0) {
     if (threadIdx.x < i) {
       count[threadIdx.x] += count[threadIdx.x + i];
     }
     __syncthreads();
     i /= 2;
   }

   if (threadIdx.x == 0) {
     H[blockIdx.x+gridDim.x*blockIdx.y] = count[0];
     H[blockIdx.y+gridDim.x*blockIdx.x] = count[0];
     if ((ki>0) && (ki<nkernel) && (kj>0) && (kj<nkernel)) {
       idx1 = np + (ki-1)*ns;
       idy1 = np + (kj-1)*ns;
       H[(idx1+idy0)+gridDim.x*(idy1+idx0)] = count[0];
       H[(idy1+idx0)+gridDim.x*(idx1+idy0)] = count[0];
     }

   }

}


""")
