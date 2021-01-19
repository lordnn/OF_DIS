#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "opticalflow_aux.h"

#if defined(USE_SSE)
#include <xmmintrin.h>
#elif defined(USE_NEON)
#include <arm_neon.h>
#define __m128 float32x4_t
#define _mm_load_ps vld1q_f32
#define _mm_store_ps vst1q_f32
#define _mm_add_ps vaddq_f32
#define _mm_sub_ps vsubq_f32
#define _mm_mul_ps vmulq_f32
#define _mm_div_ps vdivq_f32
#define _mm_set1_ps vdupq_n_f32
#define _mm_sqrt_ps vsqrtq_f32
#endif

#define datanorm 0.1f*0.1f//0.01f // square of the normalization factor
#define epsilon_color (0.001f*0.001f)//0.000001f
#define epsilon_grad (0.001f*0.001f)//0.000001f
#define epsilon_desc (0.001f*0.001f)//0.000001f
#define epsilon_smooth (0.001f*0.001f)//0.000001f

/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image_delete
void image_warp(image_t *dst, image_t *mask, const image_t *src, const image_t *wx, const image_t *wy)
#else
void image_warp(color_image_t *dst, image_t *mask, const color_image_t *src, const image_t *wx, const image_t *wy)
#endif
{
    int i, j, offset, incr_line = mask->stride-mask->width, x, y, x1, x2, y1, y2;
    float xx, yy, dx, dy;
    for(j=0,offset=0 ; j<src->height ; j++)
    {
        for(i=0 ; i<src->width ; i++,offset++)
        {
            xx = i+wx->c1[offset];
            yy = j+wy->c1[offset];
            x = (int)floor(xx);
            y = (int)floor(yy);
            dx = xx-x;
            dy = yy-y;
            mask->c1[offset] = (xx>=0 && xx<=src->width-1 && yy>=0 && yy<=src->height-1);
            x1 = MINMAX_TA(x,src->width);
            x2 = MINMAX_TA(x+1,src->width);
            y1 = MINMAX_TA(y,src->height);
            y2 = MINMAX_TA(y+1,src->height);
            dst->c1[offset] = 
                src->c1[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
                src->c1[y1*src->stride+x2]*dx*(1.0f-dy) +
                src->c1[y2*src->stride+x1]*(1.0f-dx)*dy +
                src->c1[y2*src->stride+x2]*dx*dy;
          #if (SELECTCHANNEL==3)
            dst->c2[offset] = 
                src->c2[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
                src->c2[y1*src->stride+x2]*dx*(1.0f-dy) +
                src->c2[y2*src->stride+x1]*(1.0f-dx)*dy +
                src->c2[y2*src->stride+x2]*dx*dy;
            dst->c3[offset] = 
                src->c3[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
                src->c3[y1*src->stride+x2]*dx*(1.0f-dy) +
                src->c3[y2*src->stride+x1]*(1.0f-dx)*dy +
                src->c3[y2*src->stride+x2]*dx*dy;
          #endif
        }
        offset += incr_line;
    }
}

/* compute image first and second order spatio-temporal derivatives of a color image */
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image_delete
void get_derivatives(const image_t *im1, const image_t *im2, const convolution_t *deriv,
         image_t *dx, image_t *dy, image_t *dt, 
         image_t *dxx, image_t *dxy, image_t *dyy, image_t *dxt, image_t *dyt)
#else
void get_derivatives(const color_image_t *im1, const color_image_t *im2, const convolution_t *deriv,
             color_image_t *dx, color_image_t *dy, color_image_t *dt, 
             color_image_t *dxx, color_image_t *dxy, color_image_t *dyy, color_image_t *dxt, color_image_t *dyt)
#endif
{
    // derivatives are computed on the mean of the first image and the warped second image
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
    image_t *tmp_im2 = image_new(im2->width,im2->height);    
    float *tmp_im2p = tmp_im2->c1, *dtp = dt->c1, *im1p = im1->c1, *im2p = im2->c1;
    const __m128 half = _mm_set1_ps(0.5f);
    int i=0;
    for(i=0 ; i<im1->height*im1->stride/4 ; i++){
        const __m128 mim2p = _mm_load_ps(im2p), mim1p = _mm_load_ps(im1p);
        _mm_store_ps(tmp_im2p, _mm_mul_ps(half, _mm_add_ps(mim2p, mim1p)));
        _mm_store_ps(dtp, _mm_sub_ps(mim2p, mim1p));
        dtp+=4; im1p+=4; im2p+=4; tmp_im2p+=4;
    }   
    // compute all other derivatives
    image_convolve_hv(dx, tmp_im2, deriv, NULL);
    image_convolve_hv(dy, tmp_im2, NULL, deriv);
    image_convolve_hv(dxx, dx, deriv, NULL);
    image_convolve_hv(dxy, dx, NULL, deriv);
    image_convolve_hv(dyy, dy, NULL, deriv);
    image_convolve_hv(dxt, dt, deriv, NULL);
    image_convolve_hv(dyt, dt, NULL, deriv);
    // free memory
    image_delete(tmp_im2);
#else
    color_image_t *tmp_im2 = color_image_new(im2->width,im2->height);    
    float *tmp_im2p = tmp_im2->c1, *dtp = dt->c1, *im1p = im1->c1, *im2p = im2->c1;
    const __m128 half = _mm_set1_ps(0.5f);
    int i=0;
    for(i=0 ; i<3*im1->height*im1->stride/4 ; i++){
        const __m128 mim2p = _mm_load_ps(im2p), mim1p = _mm_load_ps(im1p);
        _mm_store_ps(tmp_im2p, _mm_mul_ps(half, _mm_add_ps(mim2p, mim1p)));
        _mm_store_ps(dtp, _mm_sub_ps(mim2p, mim1p));
        dtp+=4; im1p+=4; im2p+=4; tmp_im2p+=4;
    }   
    // compute all other derivatives
    color_image_convolve_hv(dx, tmp_im2, deriv, NULL);
    color_image_convolve_hv(dy, tmp_im2, NULL, deriv);
    color_image_convolve_hv(dxx, dx, deriv, NULL);
    color_image_convolve_hv(dxy, dx, NULL, deriv);
    color_image_convolve_hv(dyy, dy, NULL, deriv);
    color_image_convolve_hv(dxt, dt, deriv, NULL);
    color_image_convolve_hv(dyt, dt, NULL, deriv);
    // free memory
    color_image_delete(tmp_im2);
#endif
}

/* compute the smoothness term */
/* It is represented as two images, the first one for horizontal smoothness, the second for vertical
   in dst_horiz, the pixel i,j represents the smoothness weight between pixel i,j and i,j+1
   in dst_vert, the pixel i,j represents the smoothness weight between pixel i,j and i+1,j */
void compute_smoothness(image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, const convolution_t *deriv_flow, const float quarter_alpha){
#define STEP 4
    const int width = uu->width, height = vv->height, stride = uu->stride;
    int j;
    image_t *ux = image_new(width,height), *vx = image_new(width,height), *uy = image_new(width,height), *vy = image_new(width,height), *smoothness = image_new(width,height);
    // compute derivatives [-0.5 0 0.5]
    convolve_horiz(ux, uu, deriv_flow);
    convolve_horiz(vx, vv, deriv_flow);
    convolve_vert(uy, uu, deriv_flow);
    convolve_vert(vy, vv, deriv_flow);
    // compute smoothness
    float *uxp = ux->c1, *vxp = vx->c1, *uyp = uy->c1, *vyp = vy->c1, *sp = smoothness->c1;
    const __m128 qa = _mm_set1_ps(quarter_alpha), epsmooth = _mm_set1_ps(epsilon_smooth);
    for(j=0 ; j< height*stride/4 ; j++){
        __m128 muxp = _mm_load_ps(uxp);
        const __m128 muyp = _mm_load_ps(uyp);
        const __m128 mvxp = _mm_load_ps(vxp);
        const __m128 mvyp = _mm_load_ps(vyp);
        muxp = _mm_add_ps(epsmooth, _mm_add_ps(_mm_add_ps(_mm_mul_ps(muxp, muxp), _mm_mul_ps(muyp, muyp)), _mm_add_ps(_mm_mul_ps(mvxp, mvxp), _mm_mul_ps(mvyp, mvyp))));
        _mm_store_ps(sp, _mm_div_ps(qa, _mm_sqrt_ps(muxp)));
        sp+=STEP;uxp+=STEP; uyp+=STEP; vxp+=STEP; vyp+=STEP;
    }
    image_delete(ux); image_delete(uy); image_delete(vx); image_delete(vy); 
    // compute dst_horiz
    float *dsthp = dst_horiz->c1; sp = smoothness->c1;
    image_t *sp_shift = image_new(width, 1); // aligned shifted copy of the current line
    for(j=0;j<height;j++){
        // create an aligned copy
        float *spf = sp;
        float *sps = sp_shift->c1;
        memcpy(sps, spf+1, sizeof(float)*(stride-1));
        int i;
        for(i=0;i<stride/4;i++){
            _mm_store_ps(dsthp, _mm_add_ps(_mm_load_ps(sp), _mm_load_ps(sps)));
            dsthp+=STEP; sp+=STEP; sps+=STEP;
        }
        memset( &dst_horiz->c1[j*stride+width-1], 0, sizeof(float)*(stride-width+1));
    }
    image_delete(sp_shift);
    // compute dst_vert
    sp = smoothness->c1;
    float *dstvp = dst_vert->c1, *sp_bottom = sp+stride;
    for(j=0 ; j<(height-1)*stride/4 ; j++){
        _mm_store_ps(dstvp, _mm_add_ps(_mm_load_ps(sp), _mm_load_ps(sp_bottom)));
        dstvp+=STEP; sp+=STEP; sp_bottom+=STEP;
    }
    memset( &dst_vert->c1[(height-1)*stride], 0, sizeof(float)*stride);
    image_delete(smoothness);
#undef STEP
}





/* sub the laplacian (smoothness term) to the right-hand term */
void sub_laplacian(image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert){
#define STEP 4
    int j;
    const int offsetline = src->stride-src->width;
    float *src_ptr = src->c1, *dst_ptr = dst->c1, *weight_horiz_ptr = weight_horiz->c1;
    // horizontal filtering
    for(j=src->height+1;--j;){ // faster than for(j=0;j<src->height;j++)
        int i;
        for(i=src->width;--i;){
            const float tmp = (*weight_horiz_ptr)*((*(src_ptr+1))-(*src_ptr));
            *dst_ptr += tmp;
            *(dst_ptr+1) -= tmp;
            dst_ptr++;
            src_ptr++;
            weight_horiz_ptr++;
        }
        dst_ptr += offsetline+1;
        src_ptr += offsetline+1;
        weight_horiz_ptr += offsetline+1;
    }
  
    float *wvp = weight_vert->c1, *srcp = src->c1, *srcp_s = srcp+src->stride, *dstp = dst->c1, *dstp_s = dstp+src->stride;
    for(j=1+(src->height-1)*src->stride/4 ; --j ;){
        const __m128 tmp = _mm_mul_ps(_mm_load_ps(wvp), _mm_sub_ps(_mm_load_ps(srcp_s), _mm_load_ps(srcp)));
        _mm_store_ps(dstp, _mm_add_ps(_mm_load_ps(dstp), tmp));
        _mm_store_ps(dstp_s, _mm_sub_ps(_mm_load_ps(dstp_s), tmp));
        wvp+=STEP; srcp+=STEP; srcp_s+=STEP; dstp+=STEP; dstp_s+=STEP;
    }
#undef STEP
}

/* compute the dataterm and the matching term
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
void compute_data_and_match(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, image_t *desc_weight, image_t *desc_flow_x, image_t *desc_flow_y, const float half_delta_over3, const float half_beta, const float half_gamma_over3) {
#define STEP 4
    const __m128 dnorm = _mm_set1_ps(datanorm);
    const __m128 hdover3 = _mm_set1_ps(half_delta_over3);
    const __m128 epscolor = _mm_set1_ps(epsilon_color);
    const __m128 hgover3 = _mm_set1_ps(half_gamma_over3);
    const __m128 epsgrad = _mm_set1_ps(epsilon_grad);
    const __m128 hbeta = _mm_set1_ps(half_beta);
    const __m128 epsdesc = _mm_set1_ps(epsilon_desc);

    float *dup = du->c1, *dvp = dv->c1,
        *maskp = mask->c1,
        *a11p = a11->c1, *a12p = a12->c1, *a22p = a22->c1, 
        *b1p = b1->c1, *b2p = b2->c1, 
        *ix1p=Ix->c1, *iy1p=Iy->c1, *iz1p=Iz->c1, *ixx1p=Ixx->c1, *ixy1p=Ixy->c1, *iyy1p=Iyy->c1, *ixz1p=Ixz->c1, *iyz1p=Iyz->c1, 
        *ix2p=Ix->c2, *iy2p=Iy->c2, *iz2p=Iz->c2, *ixx2p=Ixx->c2, *ixy2p=Ixy->c2, *iyy2p=Iyy->c2, *ixz2p=Ixz->c2, *iyz2p=Iyz->c2, 
        *ix3p=Ix->c3, *iy3p=Iy->c3, *iz3p=Iz->c3, *ixx3p=Ixx->c3, *ixy3p=Ixy->c3, *iyy3p=Iyy->c3, *ixz3p=Ixz->c3, *iyz3p=Iyz->c3, 
        *uup = uu->c1, *vvp = vv->c1, *wxp = wx->c1, *wyp = wy->c1,
        *descflowxp = desc_flow_x->c1, *descflowyp = desc_flow_y->c1, *descweightp = desc_weight->c1;

    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a12->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a22->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);
    memset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride);

    int i;
    for(i = 0 ; i<uu->height*uu->stride/4 ; i++){
        __m128 tmp, tmp2, tmp3, tmpx, tmpy, tmpxy, n1, n2, n3, n4, n5, n6;
        __m128 ma11p = _mm_load_ps(a11p);
        __m128 ma12p = _mm_load_ps(a12p);
        __m128 ma22p = _mm_load_ps(a22p);
        __m128 mb1p = _mm_load_ps(b1p);
        __m128 mb2p = _mm_load_ps(b2p);
        __m128 mdup = _mm_load_ps(dup);
        __m128 mdvp = _mm_load_ps(dvp);
        // dpsi color
        if(half_delta_over3){
            tmpx = _mm_load_ps(ix1p);
            tmpy = _mm_load_ps(iy1p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz1p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpy, mdvp)));
            n1 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n1);
            tmpx = _mm_load_ps(ix2p);
            tmpy = _mm_load_ps(iy2p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz2p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpy, mdvp)));
            n2 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n1));
            tmpx = _mm_load_ps(ix3p);
            tmpy = _mm_load_ps(iy3p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz3p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpy, mdvp)));
            n3 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n3));
            tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hdover3), _mm_sqrt_ps(_mm_add_ps(tmp, epscolor)));

            tmp3 = _mm_div_ps(tmp, n3); // tmp = _mm_div_ps(tmp, n1);
            tmpxy = _mm_load_ps(iz3p);
            tmp2 = _mm_mul_ps(tmp3, tmpx);
            tmp3 = _mm_mul_ps(tmp3, tmpy);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            ma12p = _mm_add_ps(ma12p, _mm_mul_ps(tmp2, tmpy));
            ma22p = _mm_add_ps(ma22p, _mm_mul_ps(tmp3, tmpy));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, tmpxy));
            mb2p = _mm_sub_ps(mb2p, _mm_mul_ps(tmp3, tmpxy));
            tmp3 = _mm_div_ps(tmp, n2);
            tmpx = _mm_load_ps(ix2p);
            tmpy = _mm_load_ps(iy2p);
            tmpxy = _mm_load_ps(iz2p);
            tmp2 = _mm_mul_ps(tmp3, tmpx);
            tmp3 = _mm_mul_ps(tmp3, tmpy);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            ma12p = _mm_add_ps(ma12p, _mm_mul_ps(tmp2, tmpy));
            ma22p = _mm_add_ps(ma22p, _mm_mul_ps(tmp3, tmpy));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, tmpxy));
            mb2p = _mm_sub_ps(mb2p, _mm_mul_ps(tmp3, tmpxy));
            tmp3 = _mm_div_ps(tmp, n1);
            tmpx = _mm_load_ps(ix1p);
            tmpy = _mm_load_ps(iy1p);
            tmpxy = _mm_load_ps(iz1p);
            tmp2 = _mm_mul_ps(tmp3, tmpx);
            tmp3 = _mm_mul_ps(tmp3, tmpy);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            ma12p = _mm_add_ps(ma12p, _mm_mul_ps(tmp2, tmpy));
            ma22p = _mm_add_ps(ma22p, _mm_mul_ps(tmp3, tmpy));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, tmpxy));
            mb2p = _mm_sub_ps(mb2p, _mm_mul_ps(tmp3, tmpxy));
        }
        // dpsi gradient
        tmpx = _mm_load_ps(ixx1p);
        tmpy = _mm_load_ps(iyy1p);
        tmpxy = _mm_load_ps(ixy1p);
        tmp3 = _mm_add_ps(_mm_load_ps(ixz1p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpxy, mdvp)));
        tmp2 = _mm_add_ps(_mm_load_ps(iyz1p), _mm_add_ps(_mm_mul_ps(tmpxy, mdup), _mm_mul_ps(tmpy, mdvp)));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        n1 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(tmpxy, dnorm));
        n2 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy ), _mm_add_ps(tmpxy, dnorm));
        tmp = _mm_div_ps(_mm_mul_ps(tmp3, tmp3), n1);
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n2));
        tmpx = _mm_load_ps(ixx2p);
        tmpy = _mm_load_ps(iyy2p);
        tmpxy = _mm_load_ps(ixy2p);
        tmp3 = _mm_add_ps(_mm_load_ps(ixz2p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpxy, mdvp)));
        tmp2 = _mm_add_ps(_mm_load_ps(iyz2p), _mm_add_ps(_mm_mul_ps(tmpxy, mdup), _mm_mul_ps(tmpy, mdvp)));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        n3 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(tmpxy, dnorm));
        n4 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), _mm_add_ps(tmpxy, dnorm));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp3, tmp3), n3));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n4));
        tmpx = _mm_load_ps(ixx3p);
        tmpy = _mm_load_ps(iyy3p);
        tmpxy = _mm_load_ps(ixy3p);
        tmp3 = _mm_add_ps(_mm_load_ps(ixz3p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpxy, mdvp)));
        tmp2 = _mm_add_ps(_mm_load_ps(iyz3p), _mm_add_ps(_mm_mul_ps(tmpxy, mdup), _mm_mul_ps(tmpy, mdvp)));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        n5 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(tmpxy, dnorm));
        n6 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), _mm_add_ps(tmpxy, dnorm));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp3, tmp3), n5));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n6));
        tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hgover3), _mm_sqrt_ps(_mm_add_ps(tmp, epsgrad)));

        tmp2 = _mm_div_ps(tmp2, n6); tmp3 = _mm_div_ps(tmp3, n5);
        mdup = _mm_load_ps(ixz3p);
        mdvp = _mm_load_ps(iyz3p);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpx)), _mm_mul_ps(tmp2, tmpxy)));
        ma22p = _mm_add_ps(ma22p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpy)), _mm_mul_ps(tmp3, tmpxy)));
        tmpxy = _mm_load_ps(ixy3p);
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, mdup)), _mm_mul_ps(tmp2, _mm_mul_ps(tmpxy, mdvp))));
        mb2p = _mm_sub_ps(mb2p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, mdvp)), _mm_mul_ps(tmp3, _mm_mul_ps(tmpxy, mdup))));
        ma12p = _mm_add_ps(ma12p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpxy)), _mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpxy))));
        tmp2 = _mm_div_ps(tmp, n4); tmp3 = _mm_div_ps(tmp, n3);
        tmpx = _mm_load_ps(ixx2p);
        tmpy = _mm_load_ps(iyy2p);
        tmpxy = _mm_load_ps(ixy2p);
        mdup = _mm_load_ps(ixz2p);
        mdvp = _mm_load_ps(iyz2p);
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, mdup)), _mm_mul_ps(tmp2, _mm_mul_ps(tmpxy, mdvp))));
        mb2p = _mm_sub_ps(mb2p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, mdvp)), _mm_mul_ps(tmp3, _mm_mul_ps(tmpxy, mdup))));
        ma12p = _mm_add_ps(ma12p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpxy)), _mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpxy))));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpx)), _mm_mul_ps(tmp2, tmpxy)));
        ma22p = _mm_add_ps(ma22p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpy)), _mm_mul_ps(tmp3, tmpxy)));
        tmp2 = _mm_div_ps(tmp, n2); tmp3 = _mm_div_ps(tmp, n1);
        tmpx = _mm_load_ps(ixx1p);
        tmpy = _mm_load_ps(iyy1p);
        tmpxy = _mm_load_ps(ixy1p);
        mdup = _mm_load_ps(ixz1p);
        mdvp = _mm_load_ps(iyz1p);
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, mdup)), _mm_mul_ps(tmp2, _mm_mul_ps(tmpxy, mdvp))));
        mb2p = _mm_sub_ps(mb2p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, mdvp)), _mm_mul_ps(tmp3, _mm_mul_ps(tmpxy, mdup))));
        ma12p = _mm_add_ps(ma12p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpxy)), _mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpxy))));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpx)), _mm_mul_ps(tmp2, tmpxy)));
        ma22p = _mm_add_ps(ma22p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpy)), _mm_mul_ps(tmp3, tmpxy)));
        if (half_beta){ // dpsi_match
            tmp  = _mm_sub_ps(_mm_load_ps(uup), _mm_load_ps(descflowxp));
            tmp2 = _mm_sub_ps(_mm_load_ps(vvp), _mm_load_ps(descflowyp));
            tmp = _mm_div_ps(_mm_mul_ps(hbeta, _mm_load_ps(descweightp)), _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(tmp, tmp), _mm_add_ps(_mm_mul_ps(tmp2, tmp2), epsdesc))));
            ma11p = _mm_add_ps(ma11p, tmp);
            ma22p = _mm_add_ps(ma22p, tmp);
            mb1p = _mm_sub_ps(mb1p, _mm_sub_ps(_mm_mul_ps(tmp, _mm_load_ps(wxp)), _mm_load_ps(descflowxp)));
            mb2p = _mm_sub_ps(mb2p, _mm_sub_ps(_mm_mul_ps(tmp, _mm_load_ps(wyp)), _mm_load_ps(descflowyp)));
        }
        _mm_store_ps(a11p, ma11p);
        _mm_store_ps(a12p, ma12p);
        _mm_store_ps(a22p, ma22p);
        _mm_store_ps(b1p, mb1p);
        _mm_store_ps(b2p, mb2p);

        dup+=STEP; dvp+=STEP; maskp+=STEP; a11p+=STEP; a12p+=STEP; a22p+=STEP; b1p+=STEP; b2p+=STEP;
        ix1p+=STEP; iy1p+=STEP; iz1p+=STEP; ixx1p+=STEP; ixy1p+=STEP; iyy1p+=STEP; ixz1p+=STEP; iyz1p+=STEP;
        ix2p+=STEP; iy2p+=STEP; iz2p+=STEP; ixx2p+=STEP; ixy2p+=STEP; iyy2p+=STEP; ixz2p+=STEP; iyz2p+=STEP;
        ix3p+=STEP; iy3p+=STEP; iz3p+=STEP; ixx3p+=STEP; ixy3p+=STEP; iyy3p+=STEP; ixz3p+=STEP; iyz3p+=STEP;
        uup+=STEP;vvp+=STEP;wxp+=STEP; wyp+=STEP;descflowxp+=STEP;descflowyp+=STEP;descweightp+=STEP;
    }
#undef STEP
}

/* compute the dataterm // REMOVED MATCHING TERM
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image_delete
void compute_data(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, image_t *Ix, image_t *Iy, image_t *Iz, image_t *Ixx, image_t *Ixy, image_t *Iyy, image_t *Ixz, image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3)
#else
void compute_data(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3)
#endif
{
#define STEP 4
    const __m128 dnorm = _mm_set1_ps(datanorm);
    const __m128 hdover3 = _mm_set1_ps(half_delta_over3);
    const __m128 epscolor = _mm_set1_ps(epsilon_color);
    const __m128 hgover3 = _mm_set1_ps(half_gamma_over3);
    const __m128 epsgrad = _mm_set1_ps(epsilon_grad);
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
    const __m128 three = _mm_set1_ps(3.f);
#endif
    //const v4sf hbeta = {half_beta,half_beta,half_beta,half_beta};
    //const v4sf epsdesc = {epsilon_desc,epsilon_desc,epsilon_desc,epsilon_desc};
    
    float *dup = du->c1, *dvp = dv->c1,
        *maskp = mask->c1,
        *a11p = a11->c1, *a12p = a12->c1, *a22p = a22->c1, 
        *b1p = b1->c1, *b2p = b2->c1, 
        *ix1p=Ix->c1, *iy1p=Iy->c1, *iz1p=Iz->c1, *ixx1p=Ixx->c1, *ixy1p=Ixy->c1, *iyy1p=Iyy->c1, *ixz1p=Ixz->c1, *iyz1p=Iyz->c1, 
        #if (SELECTCHANNEL==3)
        *ix2p=Ix->c2, *iy2p=Iy->c2, *iz2p=Iz->c2, *ixx2p=Ixx->c2, *ixy2p=Ixy->c2, *iyy2p=Iyy->c2, *ixz2p=Ixz->c2, *iyz2p=Iyz->c2, 
        *ix3p=Ix->c3, *iy3p=Iy->c3, *iz3p=Iz->c3, *ixx3p=Ixx->c3, *ixy3p=Ixy->c3, *iyy3p=Iyy->c3, *ixz3p=Ixz->c3, *iyz3p=Iyz->c3, 
        #endif
        *uup = uu->c1, *vvp = vv->c1, *wxp = wx->c1, *wyp = wy->c1;


    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a12->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a22->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);
    memset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride);

    int i;
    for(i = 0 ; i<uu->height*uu->stride/4 ; i++){
        __m128 tmp, tmp2, tmp3, tmpx, tmpy, tmpxy, n1, n2, ma11p, ma12p, ma22p, mb1p, mb2p;
    #if (SELECTCHANNEL==3)
        __m128 n3, n4, n5, n6;
    #endif
        ma11p = _mm_load_ps(a11p);
        ma12p = _mm_load_ps(a12p);
        ma22p = _mm_load_ps(a22p);
        mb1p = _mm_load_ps(b1p);
        mb2p = _mm_load_ps(b2p);
        const __m128 mdup = _mm_load_ps(dup);
        const __m128 mdvp = _mm_load_ps(dvp);
        // dpsi color
        if (half_delta_over3) {
            tmpx = _mm_load_ps(ix1p);
            tmpy = _mm_load_ps(iy1p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz1p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpy, mdvp)));
            n1 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n1);
            #if (SELECTCHANNEL==3)
            tmpx = _mm_load_ps(ix2p);
            tmpy = _mm_load_ps(iy2p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz2p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpy, mdvp)));
            n2 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n2));
            tmpx = _mm_load_ps(ix3p);
            tmpy = _mm_load_ps(iy3p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz3p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpy, mdvp)));
            n3 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n3));
            tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hdover3), _mm_sqrt_ps(_mm_add_ps(tmp, epscolor)));

            tmp3 = _mm_div_ps(tmp, n3);
            tmp2 = _mm_mul_ps(tmp3, tmpx);
            tmp3 = _mm_mul_ps(tmp3, tmpy);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            ma12p = _mm_add_ps(ma12p, _mm_mul_ps(tmp2, tmpy));
            ma22p = _mm_add_ps(ma22p, _mm_mul_ps(tmp3, tmpy));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, _mm_load_ps(iz3p)));
            mb2p = _mm_sub_ps(mb2p, _mm_mul_ps(tmp3, _mm_load_ps(iz3p)));
            tmpx = _mm_load_ps(ix2p);
            tmpy = _mm_load_ps(iy2p);
            tmp3 = _mm_div_ps(tmp, n2);
            tmp2 = _mm_mul_ps(tmp3, tmpx);
            tmp3 = _mm_mul_ps(tmp3, tmpy);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            ma12p = _mm_add_ps(ma12p, _mm_mul_ps(tmp2, tmpy));
            ma22p = _mm_add_ps(ma22p, _mm_mul_ps(tmp3, tmpy));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, _mm_load_ps(iz2p)));
            mb2p = _mm_sub_ps(mb2p, _mm_mul_ps(tmp3, _mm_load_ps(iz2p)));
            tmpx = _mm_load_ps(ix1p);
            tmpy = _mm_load_ps(iy2p);
            #else
            tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hdover3), _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(three, tmp), epscolor)));
            #endif
            tmp3  = _mm_div_ps(tmp, n1);
            tmp2 = _mm_mul_ps(tmp3, tmpx);
            tmp3 = _mm_mul_ps(tmp3, tmpy);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            ma12p = _mm_add_ps(ma12p, _mm_mul_ps(tmp2, tmpy));
            ma22p = _mm_add_ps(ma22p, _mm_mul_ps(tmp3, tmpy));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, _mm_load_ps(iz1p)));
            mb2p = _mm_sub_ps(mb2p, _mm_mul_ps(tmp3, _mm_load_ps(iz1p)));
        }

        // dpsi gradient
        tmpx = _mm_load_ps(ixx1p);
        tmpy = _mm_load_ps(iyy1p);
        tmpxy = _mm_load_ps(ixy1p);
        tmp2 = _mm_add_ps(_mm_load_ps(ixz1p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpxy, mdvp)));
        tmp3 = _mm_add_ps(_mm_load_ps(iyz1p), _mm_add_ps(_mm_mul_ps(tmpxy, mdup), _mm_mul_ps(tmpy, mdvp)));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        n1 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(tmpxy, dnorm));
        n2 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), _mm_add_ps(tmpxy, dnorm));
        tmp = _mm_add_ps(_mm_div_ps(_mm_mul_ps(tmp2, tmp2), n1), _mm_div_ps(_mm_mul_ps(tmp3, tmp3), n2));
        #if (SELECTCHANNEL==3)
        tmpx = _mm_load_ps(ixx2p);
        tmpy = _mm_load_ps(iyy2p);
        tmpxy = _mm_load_ps(ixy2p);
        tmp2 = _mm_add_ps(_mm_load_ps(ixz2p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpxy, mdvp)));
        tmp3 = _mm_add_ps(_mm_load_ps(iyz2p), _mm_add_ps(_mm_mul_ps(tmpxy, mdup), _mm_mul_ps(tmpy, mdvp)));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        n3 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(tmpxy, dnorm));
        n4 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), _mm_add_ps(tmpxy, dnorm));
        tmp = _mm_add_ps(_mm_div_ps(_mm_mul_ps(tmp2, tmp2), n3), _mm_div_ps(_mm_mul_ps(tmp3, tmp3), n4));
        tmpx = _mm_load_ps(ixx3p);
        tmpy = _mm_load_ps(iyy3p);
        tmpxy = _mm_load_ps(ixy3p);
        tmp2 = _mm_add_ps(_mm_load_ps(ixz3p), _mm_add_ps(_mm_mul_ps(tmpx, mdup), _mm_mul_ps(tmpxy, mdvp)));
        tmp3 = _mm_add_ps(_mm_load_ps(iyz3p), _mm_add_ps(_mm_mul_ps(tmpxy, mdup), _mm_mul_ps(tmpy, mdvp)));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        n5 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(tmpxy, dnorm));
        n6 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), _mm_add_ps(tmpxy, dnorm));
        tmp = _mm_add_ps(_mm_div_ps(_mm_mul_ps(tmp2, tmp2), n5), _mm_div_ps(_mm_mul_ps(tmp3, tmp3), n6));
        tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hgover3), _mm_sqrt_ps(_mm_add_ps(tmp, epsgrad)));

        tmp2 = _mm_div_ps(tmp, n6); tmp3 = _mm_div_ps(tmp, n5);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpx)), _mm_mul_ps(tmp2, tmpxy)));
        ma22p = _mm_add_ps(ma22p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpy)), _mm_mul_ps(tmp3, tmpxy)));
        tmpxy = _mm_load_ps(ixy3p);
        ma12p = _mm_add_ps(ma12p, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(tmp3, tmpx), _mm_mul_ps(tmp2, tmpy)), tmpxy));
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(tmp3, tmpx), _mm_load_ps(ixz3p)), _mm_mul_ps(_mm_mul_ps(tmp2, tmpxy), _mm_load_ps(iyz3p))));
        mb2p = _mm_sub_ps(mb2p, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(tmp2, tmpy), _mm_load_ps(iyz3p)), _mm_mul_ps(_mm_mul_ps(tmp3, tmpxy), _mm_load_ps(ixz3p))));
        tmp2 = _mm_div_ps(tmp, n4); tmp3 = _mm_div_ps(tmp, n3);
        tmpx = _mm_load_ps(ixx2p);
        tmpy = _mm_load_ps(iyy2p);
        tmpxy = _mm_load_ps(ixy2p);
        ma12p = _mm_add_ps(ma12p, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(tmp3, tmpx), _mm_mul_ps(tmp2, tmpy)), tmpxy));
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(tmp3, tmpx), _mm_load_ps(ixz2p)), _mm_mul_ps(_mm_mul_ps(tmp2, tmpxy), _mm_load_ps(iyz2p))));
        mb2p = _mm_sub_ps(mb2p, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(tmp2, tmpy), _mm_load_ps(iyz2p)), _mm_mul_ps(_mm_mul_ps(tmp3, tmpxy), _mm_load_ps(ixz2p))));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpx)), _mm_mul_ps(tmp2, tmpxy)));
        ma22p = _mm_add_ps(ma22p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpy)), _mm_mul_ps(tmp3, tmpxy)));
        tmpx = _mm_load_ps(ixx1p);
        tmpy = _mm_load_ps(iyy1p);
        #else
        tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hgover3), _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(three, tmp), epsgrad)));
        #endif
        tmp2 = _mm_div_ps(tmp, n2); tmp3 = _mm_div_ps(tmp, n1);
        tmpxy = _mm_load_ps(ixy1p);
        ma12p = _mm_add_ps(ma12p, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(tmp3, tmpx), _mm_mul_ps(tmp2, tmpy)), tmpxy));
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(tmp3, tmpx), _mm_load_ps(ixz1p)), _mm_mul_ps(_mm_mul_ps(tmp2, tmpxy), _mm_load_ps(iyz1p))));
        mb2p = _mm_sub_ps(mb2p, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(tmp2, tmpy), _mm_load_ps(iyz1p)), _mm_mul_ps(_mm_mul_ps(tmp3, tmpxy), _mm_load_ps(ixz1p))));
        tmpxy = _mm_mul_ps(tmpxy, tmpxy);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_mul_ps(tmpx, tmpx)), _mm_mul_ps(tmp2, tmpxy)));
        ma22p = _mm_add_ps(ma22p, _mm_add_ps(_mm_mul_ps(tmp2, _mm_mul_ps(tmpy, tmpy)), _mm_mul_ps(tmp3, tmpxy)));

        #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // multiply system to make smoothing parameters same for RGB and single-channel image
        ma11p = _mm_mul_ps(ma11p, three);
        ma12p = _mm_mul_ps(ma12p, three);
        ma22p = _mm_mul_ps(ma22p, three);
        mb1p = _mm_mul_ps(mb1p, three);
        mb2p = _mm_mul_ps(mb2p, three);
        #endif
        _mm_store_ps(a11p, ma11p);
        _mm_store_ps(a12p, ma12p);
        _mm_store_ps(a22p, ma22p);
        _mm_store_ps(b1p, mb1p);
        _mm_store_ps(b2p, mb2p);

        dup+=STEP; dvp+=STEP; maskp+=STEP; a11p+=STEP; a12p+=STEP; a22p+=STEP; b1p+=STEP; b2p+=STEP; 
        ix1p+=STEP; iy1p+=STEP; iz1p+=STEP; ixx1p+=STEP; ixy1p+=STEP; iyy1p+=STEP; ixz1p+=STEP; iyz1p+=STEP;
        #if (SELECTCHANNEL==3)
        ix2p+=STEP; iy2p+=STEP; iz2p+=STEP; ixx2p+=STEP; ixy2p+=STEP; iyy2p+=STEP; ixz2p+=STEP; iyz2p+=STEP;
        ix3p+=STEP; iy3p+=STEP; iz3p+=STEP; ixx3p+=STEP; ixy3p+=STEP; iyy3p+=STEP; ixz3p+=STEP; iyz3p+=STEP;
        #endif
        uup+=STEP;vvp+=STEP;wxp+=STEP; wyp+=STEP;
    }
#undef STEP
}



/* compute the dataterm // REMOVED MATCHING TERM
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image_delete
void compute_data_DE(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, image_t *Ix, image_t *Iy, image_t *Iz, image_t *Ixx, image_t *Ixy, image_t *Iyy, image_t *Ixz, image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3)
#else
void compute_data_DE(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3)
#endif
{
#define STEP 4
    const __m128 dnorm = _mm_set1_ps(datanorm);
    const __m128 hdover3 = _mm_set1_ps(half_delta_over3);
    const __m128 epscolor = _mm_set1_ps(epsilon_color);
    const __m128 hgover3 = _mm_set1_ps(half_gamma_over3);
    const __m128 epsgrad = _mm_set1_ps(epsilon_grad);
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
    const __m128 three = _mm_set1_ps(3.f);
#endif
    //const v4sf hbeta = {half_beta,half_beta,half_beta,half_beta};
    //const v4sf epsdesc = {epsilon_desc,epsilon_desc,epsilon_desc,epsilon_desc};

    float *dup = du->c1,
        *maskp = mask->c1,
        *a11p = a11->c1,  
        *b1p = b1->c1, 
        *ix1p=Ix->c1, *iy1p=Iy->c1, *iz1p=Iz->c1, *ixx1p=Ixx->c1, *ixy1p=Ixy->c1, *iyy1p=Iyy->c1, *ixz1p=Ixz->c1, *iyz1p=Iyz->c1,
        #if (SELECTCHANNEL==3)
        *ix2p=Ix->c2, *iy2p=Iy->c2, *iz2p=Iz->c2, *ixx2p=Ixx->c2, *ixy2p=Ixy->c2, *iyy2p=Iyy->c2, *ixz2p=Ixz->c2, *iyz2p=Iyz->c2, 
        *ix3p=Ix->c3, *iy3p=Iy->c3, *iz3p=Iz->c3, *ixx3p=Ixx->c3, *ixy3p=Ixy->c3, *iyy3p=Iyy->c3, *ixz3p=Ixz->c3, *iyz3p=Iyz->c3, 
        #endif
        *uup = uu->c1, *wxp = wx->c1;


    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);

    int i;
    for(i = 0 ; i<uu->height*uu->stride/4 ; i++){
        __m128 tmp, tmp2, tmp3, n1, n2, tmpx, tmpy, tmpxy;
    #if (SELECTCHANNEL==3)
        __m128 n3, n4, n5, n6;
    #endif
        __m128 ma11p = _mm_load_ps(a11p), mb1p = _mm_load_ps(b1p);
        const __m128 mdup = _mm_load_ps(dup);
        // dpsi color
        if(half_delta_over3){
            tmpx = _mm_load_ps(ix1p);
            tmpy = _mm_load_ps(iy1p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz1p), _mm_mul_ps(tmpx, mdup));
            n1 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n1);
            #if (SELECTCHANNEL==3)
            tmpx = _mm_load_ps(ix2p);
            tmpy = _mm_load_ps(iy2p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz2p), _mm_mul_ps(tmpx, mdup));
            n2 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n2));
            tmpx = _mm_load_ps(ix3p);
            tmpy = _mm_load_ps(iy3p);
            tmp2 = _mm_add_ps(_mm_load_ps(iz3p), _mm_mul_ps(tmpx, mdup));
            n3 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), _mm_add_ps(_mm_mul_ps(tmpy, tmpy), dnorm));
            tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n3));
            tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hdover3), _mm_sqrt_ps(_mm_add_ps(tmp, epscolor)));

            tmp2 = _mm_mul_ps(_mm_div_ps(tmp, n3), tmpx);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, _mm_load_ps(iz3p)));
            tmpx = _mm_load_ps(ix2p);
            tmp2 = _mm_mul_ps(_mm_div_ps(tmp, n2), tmpx);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, _mm_load_ps(iz2p)));
            tmpx = _mm_load_ps(ix1p);
            #else
            tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hdover3), _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(three, tmp), epscolor)));
            #endif
            tmp2 = _mm_mul_ps(_mm_div_ps(tmp, n1), tmpx);
            ma11p = _mm_add_ps(ma11p, _mm_mul_ps(tmp2, tmpx));
            mb1p = _mm_sub_ps(mb1p, _mm_mul_ps(tmp2, _mm_load_ps(iz1p)));
        }
        // dpsi gradient
        tmpx = _mm_load_ps(ixx1p);
        tmpy = _mm_load_ps(iyy1p);
        tmpxy = _mm_load_ps(ixy1p);
        tmp2 = _mm_add_ps(_mm_load_ps(iyz1p), _mm_mul_ps(tmpxy, mdup));
        tmpxy = _mm_add_ps(_mm_mul_ps(tmpxy, tmpxy), dnorm);
        n1 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), tmpxy);
        n2 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), tmpxy);
        tmp = _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n2);
        tmp2 = _mm_add_ps(_mm_load_ps(ixz1p), _mm_mul_ps(tmpx, mdup));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n1));
        #if (SELECTCHANNEL==3)
        tmpx = _mm_load_ps(ixx2p);
        tmpy = _mm_load_ps(iyy2p);
        tmpxy = _mm_load_ps(ixy2p);
        tmp2 = _mm_add_ps(_mm_load_ps(iyz2p), _mm_mul_ps(tmpxy, mdup));
        tmpxy = _mm_add_ps(_mm_mul_ps(tmpxy, tmpxy), dnorm);
        n3 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), tmpxy);
        n4 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), tmpxy);
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n4));
        tmp2 = _mm_add_ps(_mm_load_ps(ixz2p), _mm_mul_ps(tmpx, mdup));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n3));
        tmpx = _mm_load_ps(ixx3p);
        tmpy = _mm_load_ps(iyy3p);
        tmpxy = _mm_load_ps(ixy3p);
        tmp2 = _mm_add_ps(_mm_load_ps(iyz3p), _mm_mul_ps(tmpxy, mdup));
        tmpxy = _mm_add_ps(_mm_mul_ps(tmpxy, tmpxy), dnorm);
        n5 = _mm_add_ps(_mm_mul_ps(tmpx, tmpx), tmpxy);
        n6 = _mm_add_ps(_mm_mul_ps(tmpy, tmpy), tmpxy);
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n6));
        tmp2 = _mm_add_ps(_mm_load_ps(ixz3p), _mm_mul_ps(tmpx, mdup));
        tmp = _mm_add_ps(tmp, _mm_div_ps(_mm_mul_ps(tmp2, tmp2), n5));
        tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hgover3), _mm_sqrt_ps(_mm_add_ps(tmp, epsgrad)));

        tmpxy = _mm_load_ps(ixy3p);
        tmp2 = _mm_mul_ps(_mm_div_ps(tmp, n6), tmpxy); tmp3 = _mm_mul_ps(_mm_div_ps(tmp, n5), tmpx);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, tmpx), _mm_mul_ps(tmp2, tmpxy)));
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_load_ps(ixz3p)), _mm_mul_ps(tmp2, _mm_load_ps(iyz3p))));
        tmpx = _mm_load_ps(ixx2p);
        tmpy = _mm_load_ps(iyy2p);
        tmpxy = _mm_load_ps(ixy2p);
        tmp2 = _mm_mul_ps(_mm_div_ps(tmp, n4), tmpxy); tmp3 = _mm_mul_ps(_mm_div_ps(tmp, n3), tmpx);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, tmpx), _mm_mul_ps(tmp2, tmpxy)));
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_load_ps(ixz2p)), _mm_mul_ps(tmp2, _mm_load_ps(iyz2p))));
        tmpx = _mm_load_ps(ixx1p);
        tmpy = _mm_load_ps(iyy1p);
        #else
        tmp = _mm_div_ps(_mm_mul_ps(_mm_load_ps(maskp), hgover3), _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(three, tmp), epsgrad)));
        #endif
        tmpxy = _mm_load_ps(ixy1p);
        tmp2 = _mm_mul_ps(_mm_div_ps(tmp, n2), tmpxy); tmp3 = _mm_mul_ps(_mm_div_ps(tmp, n1), tmpx);
        ma11p = _mm_add_ps(ma11p, _mm_add_ps(_mm_mul_ps(tmp3, tmpx), _mm_mul_ps(tmp2, tmpxy)));
        mb1p = _mm_sub_ps(mb1p, _mm_add_ps(_mm_mul_ps(tmp3, _mm_load_ps(ixz1p)), _mm_mul_ps(tmp2, _mm_load_ps(iyz1p))));

        #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // multiply system to make smoothing parameters same for RGB and single-channel image
        ma11p = _mm_mul_ps(ma11p, three);
        mb1p = _mm_mul_ps(mb1p, three);
        #endif
        _mm_store_ps(a11p, ma11p);
        _mm_store_ps(b1p, mb1p);

        dup+=STEP; maskp+=STEP; a11p+=STEP; b1p+=STEP; 
        ix1p+=STEP; iy1p+=STEP; iz1p+=STEP; ixx1p+=STEP; ixy1p+=STEP; iyy1p+=STEP; ixz1p+=STEP; iyz1p+=STEP;
        #if (SELECTCHANNEL==3)
        ix2p+=STEP; iy2p+=STEP; iz2p+=STEP; ixx2p+=STEP; ixy2p+=STEP; iyy2p+=STEP; ixz2p+=STEP; iyz2p+=STEP;
        ix3p+=STEP; iy3p+=STEP; iz3p+=STEP; ixx3p+=STEP; ixy3p+=STEP; iyy3p+=STEP; ixz3p+=STEP; iyz3p+=STEP;
        #endif
        uup+=STEP; wxp+=STEP;
    }
#undef STEP
}
