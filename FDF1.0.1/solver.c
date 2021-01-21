#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <omp.h>

#include "image.h"
#include "solver.h"

#if defined(USE_SSE)
#include <xmmintrin.h>
#define _vmla_ps(a, b, c) _mm_add_ps((a), _mm_mul_ps((b), (c)))
#define _vmls_ps(a, b, c) _mm_sub_ps((a), _mm_mul_ps((b), (c)))
#elif defined(USE_NEON)
#include <arm_neon.h>
#define __m128 float32x4_t
#define _mm_load_ps vld1q_f32
#define _mm_store_ps vst1q_f32
#define _mm_storeu_ps vst1q_f32
#define _mm_add_ps vaddq_f32
#define _mm_sub_ps vsubq_f32
#define _mm_mul_ps vmulq_f32
#define _mm_div_ps vdivq_f32
#define _mm_set1_ps vdupq_n_f32
#define _vmla_ps(a, b, c) vmlaq_f32((a), (b), (c))
#define _vmls_ps(a, b, c) vmlsq_f32((a), (b), (c))
#endif

//THIS IS A SLOW VERSION BUT READABLE
//Perform n iterations of the sor_coupled algorithm
//du and dv are used as initial guesses
//The system form is the same as in opticalflow.c
void sor_coupled_slow_but_readable(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega)
{
    int i,j,iter;
    for (iter = 0; iter < iterations; iter++) {
    #pragma omp parallel for
    for (j = 0; j < du->height; j++) {
      float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2;//,det;
      for (i = 0; i < du->width; i++) {
          sigma_u = 0.0f;
          sigma_v = 0.0f;
          sum_dpsis = 0.0f;
          if (j > 0) {
            sigma_u -= dpsis_vert->c1[(j-1)*du->stride+i]*du->c1[(j-1)*du->stride+i];
            sigma_v -= dpsis_vert->c1[(j-1)*du->stride+i]*dv->c1[(j-1)*du->stride+i];
            sum_dpsis += dpsis_vert->c1[(j-1)*du->stride+i];
          }
          if (i > 0) {
            sigma_u -= dpsis_horiz->c1[j*du->stride+i-1]*du->c1[j*du->stride+i-1];
            sigma_v -= dpsis_horiz->c1[j*du->stride+i-1]*dv->c1[j*du->stride+i-1];
            sum_dpsis += dpsis_horiz->c1[j*du->stride+i-1];
          }
          if (j < du->height - 1) {
            sigma_u -= dpsis_vert->c1[j*du->stride+i]*du->c1[(j+1)*du->stride+i];
            sigma_v -= dpsis_vert->c1[j*du->stride+i]*dv->c1[(j+1)*du->stride+i];
            sum_dpsis += dpsis_vert->c1[j*du->stride+i];
          }
          if (i < du->width - 1) {
            sigma_u -= dpsis_horiz->c1[j*du->stride+i]*du->c1[j*du->stride+i+1];
            sigma_v -= dpsis_horiz->c1[j*du->stride+i]*dv->c1[j*du->stride+i+1];
            sum_dpsis += dpsis_horiz->c1[j*du->stride+i];
          }
          A11 = a11->c1[j*du->stride+i]+sum_dpsis;
          A12 = a12->c1[j*du->stride+i];
          A22 = a22->c1[j*du->stride+i]+sum_dpsis;
          //det = A11*A22-A12*A12;
          B1 = b1->c1[j*du->stride+i]-sigma_u;
          B2 = b2->c1[j*du->stride+i]-sigma_v;
//           du->c1[j*du->stride+i] = (1.0f-omega)*du->c1[j*du->stride+i] +omega*( A22*B1-A12*B2)/det;
//           dv->c1[j*du->stride+i] = (1.0f-omega)*dv->c1[j*du->stride+i] +omega*(-A12*B1+A11*B2)/det;
          du->c1[j*du->stride+i] = (1.0f-omega)*du->c1[j*du->stride+i] + omega/A11 *(B1 - A12* dv->c1[j*du->stride+i] );
          dv->c1[j*du->stride+i] = (1.0f-omega)*dv->c1[j*du->stride+i] + omega/A22 *(B2 - A12* du->c1[j*du->stride+i] );
      }
    }
  }
}

 // THIS IS A FASTER VERSION BUT UNREADABLE, ONLY OPTICAL FLOW WITHOUT OPENMP PARALLELIZATION
 // the first iteration is separated from the other to compute the inverse of the 2x2 block diagonal
 // each iteration is split in two first line / middle lines / last line, and the left block is computed separately on each line
void sor_coupled(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega){
    //sor_coupled_slow(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations,omega); return; printf("test\n");

    if (du->width<2 || du->height<2 || iterations < 1) {
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations,omega);
        return;
    }

    const int stride = du->stride, width = du->width;
    const int iterheight = du->height-1, iterline = (stride)/4, width_minus_1_sizeoffloat = sizeof(float)*(width-1);
    int j,iter,i,k;
    image_t *floatarray = image_new(width, 3);
    if (floatarray==NULL) {
        fprintf(stderr, "error in sor_coupled(): not enough memory\n");
        exit(1);
    }
    float *f1 = floatarray->c1;
    float *f2 = f1 + floatarray->stride;
    float *f3 = f2 + floatarray->stride;
    f1[0] = 0.0f;
    memset(&f1[width], 0, sizeof(float)*(stride-width));
    memset(&f2[width-1], 0, sizeof(float)*(stride-width+1));
    memset(&f3[width-1], 0, sizeof(float)*(stride-width+1));
    const __m128 mzero = _mm_set1_ps(0.f);

    { // first iteration
        float *a11p = a11->c1, *a12p = a12->c1, *a22p = a22->c1, *b1p = b1->c1, *b2p = b2->c1, *hp = dpsis_horiz->c1, *vp = dpsis_vert->c1;
        float *du_ptr = du->c1, *dv_ptr = dv->c1;
        float *dub = du_ptr+stride, *dvb = dv_ptr+stride;

        { // first iteration - first line

            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            float *hpl = f1, *dur = f2, *dvr = f3;

            { // left block
                // reverse 2x2 diagonal block
                const __m128 dpsis = _mm_add_ps(_mm_load_ps(hpl), _mm_add_ps(_mm_load_ps(hp), _mm_load_ps(vp)));
                const __m128 A11 = _mm_add_ps(_mm_load_ps(a22p), dpsis), A22 = _mm_add_ps(_mm_load_ps(a11p), dpsis);
                const __m128 ma12p = _mm_load_ps(a12p);
                const __m128 det = _vmls_ps(_mm_mul_ps(A11, A22), ma12p, ma12p);
                _mm_store_ps(a11p, _mm_div_ps(A11, det));
                _mm_store_ps(a22p, _mm_div_ps(A22, det));
                _mm_store_ps(a12p,  _mm_div_ps(ma12p, _mm_sub_ps(mzero, det)));
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vp), _mm_load_ps(dub)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vp), _mm_load_ps(dvb)));

                du_ptr[0] += omega*( a11p[0]*s1[0] + a12p[0]*s2[0] - du_ptr[0] );
                dv_ptr[0] += omega*( a12p[0]*s1[0] + a22p[0]*s2[0] - dv_ptr[0] );
                for (k = 1; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;        
            }
            for (i=iterline;--i;) {
                // reverse 2x2 diagonal block
                const __m128 dpsis = _mm_add_ps(_mm_load_ps(hpl), _mm_add_ps(_mm_load_ps(hp), _mm_load_ps(vp)));
                const __m128 A11 = _mm_add_ps(_mm_load_ps(a22p), dpsis), A22 = _mm_add_ps(_mm_load_ps(a11p), dpsis);
                const __m128 ma12p = _mm_load_ps(a12p);
                const __m128 det = _vmls_ps(_mm_mul_ps(A11, A22), ma12p, ma12p);
                _mm_store_ps(a11p, _mm_div_ps(A11, det));
                _mm_store_ps(a22p, _mm_div_ps(A22, det));
                _mm_store_ps(a12p, _mm_div_ps(ma12p, _mm_sub_ps(mzero, det)));
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vp), _mm_load_ps(dub)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vp), _mm_load_ps(dvb)));
                for (k = 0; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
        }

        float *vpt = dpsis_vert->c1;
        float *dut = du->c1, *dvt = dv->c1;

        for(j=iterheight;--j;){ // first iteration - middle lines
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            float *hpl = f1, *dur = f2, *dvr = f3;

            { // left block
                // reverse 2x2 diagonal block
                const __m128 dpsis = _mm_add_ps(_mm_add_ps(_mm_load_ps(hpl), _mm_load_ps(hp)), _mm_add_ps(_mm_load_ps(vpt), _mm_load_ps(vp)));
                const __m128 A11 = _mm_add_ps(_mm_load_ps(a22p), dpsis), A22 = _mm_add_ps(_mm_load_ps(a11p), dpsis);
                const __m128 ma12p = _mm_load_ps(a12p);
                const __m128 det = _vmls_ps(_mm_mul_ps(A11, A22), ma12p, ma12p);
                _mm_store_ps(a11p, _mm_div_ps(A11, det));
                _mm_store_ps(a22p, _mm_div_ps(A22, det));
                _mm_store_ps(a12p, _mm_div_ps(ma12p, _mm_sub_ps(mzero, det)));
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)), _vmla_ps(_mm_load_ps(b1p), _mm_load_ps(vp), _mm_load_ps(dub))));
                _mm_storeu_ps(s2, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)), _vmla_ps(_mm_load_ps(b2p), _mm_load_ps(vp), _mm_load_ps(dvb))));
                
                du_ptr[0] += omega*( a11p[0]*s1[0] + a12p[0]*s2[0] - du_ptr[0] );
                dv_ptr[0] += omega*( a12p[0]*s1[0] + a22p[0]*s2[0] - dv_ptr[0] );
                for (k = 1; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
            for (i=iterline;--i;) {
                // reverse 2x2 diagonal block
                const __m128 dpsis = _mm_add_ps(_mm_add_ps(_mm_load_ps(hpl), _mm_load_ps(hp)), _mm_add_ps(_mm_load_ps(vpt), _mm_load_ps(vp)));
                const __m128 A11 = _mm_add_ps(_mm_load_ps(a22p), dpsis), A22 = _mm_add_ps(_mm_load_ps(a11p), dpsis);
                const __m128 ma12p = _mm_load_ps(a12p);
                const __m128 det = _vmls_ps(_mm_mul_ps(A11, A22), ma12p, ma12p);
                _mm_store_ps(a11p, _mm_div_ps(A11, det));
                _mm_store_ps(a22p, _mm_div_ps(A22, det));
                _mm_store_ps(a12p, _mm_div_ps(ma12p, _mm_sub_ps(mzero, det)));
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)), _vmla_ps(_mm_load_ps(b1p), _mm_load_ps(vp), _mm_load_ps(dub))));
                _mm_storeu_ps(s2, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)), _vmla_ps(_mm_load_ps(b2p), _mm_load_ps(vp), _mm_load_ps(dvb))));
                
                for (k = 0; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
        }

        { // first iteration - last line
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            float *hpl = f1, *dur = f2, *dvr = f3;

            { // left block
                // reverse 2x2 diagonal block
                const __m128 dpsis = _mm_add_ps(_mm_load_ps(hpl), _mm_add_ps(_mm_load_ps(hp), _mm_load_ps(vpt)));
                const __m128 A11 = _mm_add_ps(_mm_load_ps(a22p), dpsis), A22 = _mm_add_ps(_mm_load_ps(a11p), dpsis);
                const __m128 ma12p = _mm_load_ps(a12p);
                const __m128 det = _vmls_ps(_mm_mul_ps(A11, A22), ma12p, ma12p);
                _mm_store_ps(a11p, _mm_div_ps(A11, det));
                _mm_store_ps(a22p, _mm_div_ps(A22, det));
                _mm_store_ps(a12p, _mm_div_ps(ma12p, _mm_sub_ps(mzero, det)));
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)));
                du_ptr[0] += omega*( a11p[0]*s1[0] + a12p[0]*s2[0] - du_ptr[0] );
                dv_ptr[0] += omega*( a12p[0]*s1[0] + a22p[0]*s2[0] - dv_ptr[0] );
                for (k = 1; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
            for (i=iterline;--i;) {
                // reverse 2x2 diagonal block
                const __m128 dpsis = _mm_add_ps(_mm_load_ps(hpl), _mm_add_ps(_mm_load_ps(hp), _mm_load_ps(vpt)));
                const __m128 A11 = _mm_add_ps(_mm_load_ps(a22p), dpsis), A22 = _mm_add_ps(_mm_load_ps(a11p), dpsis);
                const __m128 ma12p = _mm_load_ps(a12p);
                const __m128 det = _vmls_ps(_mm_mul_ps(A11, A22), ma12p, ma12p);
                _mm_store_ps(a11p, _mm_div_ps(A11, det));
                _mm_store_ps(a22p, _mm_div_ps(A22, det));
                _mm_store_ps(a12p, _mm_div_ps(ma12p, _mm_sub_ps(mzero, det)));
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)));
                for (k = 0; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
        }
    }

   for(iter=iterations;--iter;)   // other iterations
   {
        float *a11p = a11->c1, *a12p = a12->c1, *a22p = a22->c1, *b1p = b1->c1, *b2p = b2->c1, *hp = dpsis_horiz->c1, *vp = dpsis_vert->c1;
        float *du_ptr = du->c1, *dv_ptr = dv->c1;
        float *dub = du_ptr+stride, *dvb = dv_ptr+stride;

        { // other iteration - first line

            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            float *hpl = f1, *dur = f2, *dvr = f3;

            { // left block
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vp), _mm_load_ps(dub)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vp), _mm_load_ps(dvb)));
                du_ptr[0] += omega*( a11p[0]*s1[0] + a12p[0]*s2[0] - du_ptr[0] );
                dv_ptr[0] += omega*( a12p[0]*s1[0] + a22p[0]*s2[0] - dv_ptr[0] );
                for (k = 1; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
            for (i=iterline;--i;) {
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vp), _mm_load_ps(dub)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vp), _mm_load_ps(dvb)));
                for (k = 0; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
        }

        float *vpt = dpsis_vert->c1;
        float *dut = du->c1, *dvt = dv->c1;

        for(j=iterheight;--j;)  // other iteration - middle lines
        {
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            float *hpl = f1, *dur = f2, *dvr = f3;

            { // left block
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)), _vmla_ps(_mm_load_ps(b1p), _mm_load_ps(vp), _mm_load_ps(dub))));
                _mm_storeu_ps(s2, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)), _vmla_ps(_mm_load_ps(b2p), _mm_load_ps(vp), _mm_load_ps(dvb))));
                du_ptr[0] += omega*( a11p[0]*s1[0] + a12p[0]*s2[0] - du_ptr[0] );
                dv_ptr[0] += omega*( a12p[0]*s1[0] + a22p[0]*s2[0] - dv_ptr[0] );
                for (k = 1; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
            for (i=iterline; --i;) {
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)), _vmla_ps(_mm_load_ps(b1p), _mm_load_ps(vp), _mm_load_ps(dub))));
                _mm_storeu_ps(s2, _mm_add_ps(_vmla_ps(_mm_mul_ps(_mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)), _vmla_ps(_mm_load_ps(b2p), _mm_load_ps(vp), _mm_load_ps(dvb))));
                for (k = 0; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; vp+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; dub+=4; dvb +=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
        }

        { // other iteration - last line
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            float *hpl = f1, *dur = f2, *dvr = f3;

            { // left block
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)));
                du_ptr[0] += omega*( a11p[0]*s1[0] + a12p[0]*s2[0] - du_ptr[0] );
                dv_ptr[0] += omega*( a12p[0]*s1[0] + a22p[0]*s2[0] - dv_ptr[0] );
                for (k = 1; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
            for (i=iterline;--i;) {
                // do one iteration
                float s1[4], s2[4];
                _mm_storeu_ps(s1, _vmla_ps(_vmla_ps(_mm_load_ps(b1p), _mm_load_ps(hp), _mm_load_ps(dur)), _mm_load_ps(vpt), _mm_load_ps(dut)));
                _mm_storeu_ps(s2, _vmla_ps(_vmla_ps(_mm_load_ps(b2p), _mm_load_ps(hp), _mm_load_ps(dvr)), _mm_load_ps(vpt), _mm_load_ps(dvt)));
                for (k = 0; k < 4; k++) {
                    const float B1 = hpl[k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[k]*B1 + a12p[k]*B2 - du_ptr[k] );
                    dv_ptr[k] += omega*( a12p[k]*B1 + a22p[k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=4; hp+=4; vpt+=4; a11p+=4; a12p+=4; a22p+=4;
                dur+=4; dvr+=4; dut+=4; dvt+=4; b1p+=4; b2p+=4;
                du_ptr += 4; dv_ptr += 4;
            }
        }
    }
    image_delete(floatarray);
}

//THIS IS A SLOW VERSION BUT READABLE
//Perform n iterations of the sor_coupled algorithm
//du is used as initial guesses
//The system form is the same as in opticalflow.c
void sor_coupled_slow_but_readable_DE(image_t *du, const image_t *a11, const image_t *b1, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega)
{
    int i,j,iter;
    for (iter = 0; iter < iterations; ++iter) {
    #pragma omp parallel for
        for (j = 0; j < du->height; ++j) {
            float sigma_u,sum_dpsis,A11,B1;
            for (i = 0; i < du->width; ++i) {
                sigma_u = 0.0f;
                sum_dpsis = 0.0f;
                if (j > 0) {
                  sigma_u -= dpsis_vert->c1[(j-1)*du->stride+i]*du->c1[(j-1)*du->stride+i];
                  sum_dpsis += dpsis_vert->c1[(j-1)*du->stride+i];
                }
                if (i > 0) {
                  sigma_u -= dpsis_horiz->c1[j*du->stride+i-1]*du->c1[j*du->stride+i-1];
                  sum_dpsis += dpsis_horiz->c1[j*du->stride+i-1];
                }
                if (j < du->height - 1) {
                  sigma_u -= dpsis_vert->c1[j*du->stride+i]*du->c1[(j+1)*du->stride+i];
                  sum_dpsis += dpsis_vert->c1[j*du->stride+i];
                }
                if (i < du->width - 1) {
                  sigma_u -= dpsis_horiz->c1[j*du->stride+i]*du->c1[j*du->stride+i+1];
                  sum_dpsis += dpsis_horiz->c1[j*du->stride+i];
                }
                A11 = a11->c1[j*du->stride+i]+sum_dpsis;
                B1 = b1->c1[j*du->stride+i]-sigma_u;
                du->c1[j*du->stride+i] = (1.0f-omega)*du->c1[j*du->stride+i] +omega*( B1/A11 );
            }
        }
    }
}
