#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "image.h"

#include <smmintrin.h>
#include <xmmintrin.h>

/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(const int width, const int height){
    image_t *image = (image_t*) malloc(sizeof(image_t));
    if(image == NULL){
        fprintf(stderr, "Error: image_new() - not enough memory !\n");
        exit(1);
    }
    image->width = width;
    image->height = height;  
    image->stride = ( (width+3) / 4 ) * 4;
    image->c1 = (float*) _aligned_malloc(image->stride*height*sizeof(float), 16);
    if(image->c1 == NULL){
        fprintf(stderr, "Error: image_new() - not enough memory !\n");
        exit(1);
    }
    return image;
}

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src){
    image_t *dst = image_new(src->width, src->height);
    memcpy(dst->c1, src->c1, src->stride*src->height*sizeof(float));
    return dst;
}

/* set all pixels values to zeros */
void image_erase(image_t *image){
    memset(image->c1, 0, image->stride*image->height*sizeof(float));
}


/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, const float scalar){
    int i;
    float *imp = image->c1;
    const __m128 scalarp = _mm_set1_ps(scalar);
    for( i=0 ; i<image->stride/4*image->height ; i++){
        _mm_store_ps(imp, _mm_mul_ps(_mm_load_ps(imp), scalarp));
        imp+=4;
    }
}

/* free memory of an image */
void image_delete(image_t *image){
    if(image == NULL){
        //fprintf(stderr, "Warning: Delete image --> Ignore action (image not allocated)\n");
    }else{
    _aligned_free(image->c1);
    free(image);
    }
}


/* allocate a new color image of size width x height */
color_image_t *color_image_new(const int width, const int height){
    color_image_t *image = (color_image_t*) malloc(sizeof(color_image_t));
    if(image == NULL){
        fprintf(stderr, "Error: color_image_new() - not enough memory !\n");
        exit(1);
    }
    image->width = width;
    image->height = height;  
    image->stride = ( (width+3) / 4 ) * 4;
    image->c1 = (float*) _aligned_malloc(3*image->stride*height*sizeof(float), 16);
    if(image->c1 == NULL){
        fprintf(stderr, "Error: color_image_new() - not enough memory !\n");
        exit(1);
    }
    image->c2 =  image->c1+image->stride*height;
    image->c3 =  image->c2+image->stride*height;
    return image;
}

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src){
    color_image_t *dst = color_image_new(src->width, src->height);
    memcpy(dst->c1, src->c1, 3*src->stride*src->height*sizeof(float));
    return dst;
}

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image){
    memset(image->c1, 0, 3*image->stride*image->height*sizeof(float));
}

/* free memory of a color image */
void color_image_delete(color_image_t *image){
    if(image){
        _aligned_free(image->c1); // c2 and c3 was allocated at the same moment
        free(image);
    }
}

/* reallocate the memory of an image to fit the new width height */
void resize_if_needed_newsize(image_t *im, const int w, const int h){
    if(im->width != w || im->height != h){
        im->width = w;
        im->height = h;
        im->stride = ((w+3)/4)*4;
        float *data = (float *) _aligned_malloc(im->stride*h*sizeof(float), 16);
        if(data == NULL){
            fprintf(stderr, "Error: resize_if_needed_newsize() - not enough memory !\n");
            exit(1);
        }
        _aligned_free(im->c1);
        im->c1 = data;
    }
}


/************ Resizing *********/

/* resize an image to a new size (assumes a difference only in width) */
static void image_resize_horiz(image_t *dst, const image_t *src){
    const float real_scale = ((float) src->width-1) / ((float) dst->width-1);
    int i;
    for(i = 0; i < dst->height; i++){
        int j;
        for(j = 0; j < dst->width; j++){
            const int x = floor((float) j * real_scale);
            const float dx = j * real_scale - x; 
            if(x >= (src->width - 1)){
                dst->c1[i * dst->stride + j] = src->c1[i * src->stride + src->width - 1]; 
            }else{
                dst->c1[i * dst->stride + j] = 
                    (1.0f - dx) * src->c1[i * src->stride + x    ] + 
                    (       dx) * src->c1[i * src->stride + x + 1];
            }
        }
    }
}

/* resize a color image to a new size (assumes a difference only in width) */
static void color_image_resize_horiz(color_image_t *dst, const color_image_t *src){
    const float real_scale = ((float) src->width-1) / ((float) dst->width-1);
    int i;
    for(i = 0; i < dst->height; i++){
        int j;
        for(j = 0; j < dst->width; j++){
            const int x = floor((float) j * real_scale);
            const float dx = j * real_scale - x; 
            if(x >= (src->width - 1)){
                dst->c1[i * dst->stride + j] = src->c1[i * src->stride + src->width - 1]; 
                dst->c2[i * dst->stride + j] = src->c2[i * src->stride + src->width - 1]; 
                dst->c3[i * dst->stride + j] = src->c3[i * src->stride + src->width - 1]; 
            }else{
                dst->c1[i * dst->stride + j] = 
                    (1.0f - dx) * src->c1[i * src->stride + x    ] + 
                    (       dx) * src->c1[i * src->stride + x + 1];
                dst->c2[i * dst->stride + j] = 
                    (1.0f - dx) * src->c2[i * src->stride + x    ] + 
                    (       dx) * src->c2[i * src->stride + x + 1];
                dst->c3[i * dst->stride + j] = 
                    (1.0f - dx) * src->c3[i * src->stride + x    ] + 
                    (       dx) * src->c3[i * src->stride + x + 1];
            }
        }
    }
}

/* resize an image to a new size (assumes a difference only in height) */
static void image_resize_vert(image_t *dst, const image_t *src){
    const float real_scale = ((float) src->height-1) / ((float) dst->height-1);
    int i;
    for(i = 0; i < dst->width; i++){
        int j;
        for(j = 0; j < dst->height; j++){
        const int y = floor((float) j * real_scale);
        const float dy = j * real_scale - y;
        if(y >= (src->height - 1)){
                dst->c1[j * dst->stride + i] = src->c1[i + (src->height - 1) * src->stride]; 
            }else{
                dst->c1[j * dst->stride + i] =
                    (1.0f - dy) * src->c1[i + (y    ) * src->stride] + 
                    (       dy) * src->c1[i + (y + 1) * src->stride];
            }
        }
    }
}

/* resize a color image to a new size (assumes a difference only in height) */
static void color_image_resize_vert(color_image_t *dst, const color_image_t *src){
    const float real_scale = ((float) src->height) / ((float) dst->height);
    int i;
    for(i = 0; i < dst->width; i++){
        int j;
        for(j = 0; j < dst->height; j++){
        const int y = floor((float) j * real_scale);
        const float dy = j * real_scale - y;
        if(y >= (src->height - 1)){
            dst->c1[j * dst->stride + i] = src->c1[i + (src->height - 1) * src->stride]; 
            dst->c2[j * dst->stride + i] = src->c2[i + (src->height - 1) * src->stride]; 
            dst->c3[j * dst->stride + i] = src->c3[i + (src->height - 1) * src->stride]; 
        }else{
            dst->c1[j * dst->stride + i] =
                (1.0f - dy) * src->c1[i +  y      * src->stride] + 
                (       dy) * src->c1[i + (y + 1) * src->stride];
            dst->c2[j * dst->stride + i] =
                (1.0f - dy) * src->c2[i +  y      * src->stride] + 
                (       dy) * src->c2[i + (y + 1) * src->stride];
            dst->c3[j * dst->stride + i] =
                (1.0f - dy) * src->c3[i +  y      * src->stride] + 
                (       dy) * src->c3[i + (y + 1) * src->stride];
            }
        }
    }
}

/* return a resize version of the image with bilinear interpolation */
image_t *image_resize_bilinear(const image_t *src, const float scale){
    const int width = src->width, height = src->height;
    const int newwidth = (int) (1.5f + (width-1) / scale); // 0.5f for rounding instead of flooring, and the remaining comes from scale = (dst-1)/(src-1)
    const int newheight = (int) (1.5f + (height-1) / scale);
    image_t *dst = image_new(newwidth,newheight);
    if(height*newwidth < width*newheight){
        image_t *tmp = image_new(newwidth,height);
        image_resize_horiz(tmp,src);
        image_resize_vert(dst,tmp);
        image_delete(tmp);
    }else{
        image_t *tmp = image_new(width,newheight);
        image_resize_vert(tmp,src);
        image_resize_horiz(dst,tmp);
        image_delete(tmp);
    }
    return dst;
}

/* resize an image with bilinear interpolation to fit the new weidht, height ; reallocation is done if necessary */
void image_resize_bilinear_newsize(image_t *dst, const image_t *src, const int new_width, const int new_height){
    resize_if_needed_newsize(dst,new_width,new_height);
    if(new_width < new_height){
        image_t *tmp = image_new(new_width,src->height);
        image_resize_horiz(tmp,src);
        image_resize_vert(dst,tmp);
        image_delete(tmp);
    }else{
        image_t *tmp = image_new(src->width,new_height);
        image_resize_vert(tmp,src);
        image_resize_horiz(dst,tmp); 
        image_delete(tmp);
    }
}

/* resize a color image  with bilinear interpolation */
color_image_t *color_image_resize_bilinear(const color_image_t *src, const float scale){
    const int width = src->width, height = src->height;
    const int newwidth = (int) (1.5f + (width-1) / scale); // 0.5f for rounding instead of flooring, and the remaining comes from scale = (dst-1)/(src-1)
    const int newheight = (int) (1.5f + (height-1) / scale);
    color_image_t *dst = color_image_new(newwidth,newheight);
    if(height*newwidth < width*newheight){
        color_image_t *tmp = color_image_new(newwidth,height);
        color_image_resize_horiz(tmp,src);
        color_image_resize_vert(dst,tmp);
        color_image_delete(tmp);
    }else{
        color_image_t *tmp = color_image_new(width,newheight);
        color_image_resize_vert(tmp,src);
        color_image_resize_horiz(dst,tmp);
        color_image_delete(tmp);
    }
    return dst;
}

/************ Convolution ******/

/* return half coefficient of a gaussian filter
Details:
- return a float* containing the coefficient from middle to border of the filter, so starting by 0,
- it so contains half of the coefficient.
- sigma is the standard deviation.
- filter_order is an output where the size of the output array is stored */
float *gaussian_filter(const float sigma, int *filter_order){
    if(sigma == 0.0f){
        fprintf(stderr, "gaussian_filter() error: sigma is zeros\n");
        exit(1);
    }
    if(!filter_order){
        fprintf(stderr, "gaussian_filter() error: filter_order is null\n");
        exit(1);
    }
    // computer the filter order as 1 + 2* floor(3*sigma)
    *filter_order = floor(3*sigma); 
    if ( *filter_order == 0 )
        *filter_order = 1; 
    // compute coefficients
    float *data = (float*) malloc(sizeof(float) * (2*(*filter_order)+1));
    if(data == NULL ){
        fprintf(stderr, "gaussian_filter() error: not enough memory\n");
        exit(1);
    }
    const float alpha = 1.0f/(2.0f*sigma*sigma);
    float sum = 0.0f;
    int i;
    for(i=-(*filter_order) ; i<=*filter_order ; i++){
        data[i+(*filter_order)] = exp(-i*i*alpha);
        sum += data[i+(*filter_order)];
    }
    for(i=-(*filter_order) ; i<=*filter_order ; i++){
        data[i+(*filter_order)] /= sum;
    }
    // fill the output
    float *data2 = (float*) malloc(sizeof(float)*(*filter_order+1));
    if(data2 == NULL ){
        fprintf(stderr, "gaussian_filter() error: not enough memory\n");
        exit(1);
    }
    memcpy(data2, &data[*filter_order], sizeof(float)*(*filter_order)+sizeof(float));
    free(data);
    return data2;
}

/* given half of the coef, compute the full coefficients and the accumulated coefficients */
static void convolve_extract_coeffs(const int order, const float *half_coeffs, float *coeffs, float *coeffs_accu, const int even){
    int i;
    float accu = 0.0;
    if(even){
        for(i = 0 ; i <= order; i++){
            coeffs[order - i] = coeffs[order + i] = half_coeffs[i];
        }
        for(i = 0 ; i <= order; i++){
            accu += coeffs[i];
            coeffs_accu[2 * order - i] = coeffs_accu[i] = accu;
        }
    }else{
        for(i = 0; i <= order; i++){
            coeffs[order - i] = +half_coeffs[i];
            coeffs[order + i] = -half_coeffs[i];
        }
        for(i = 0 ; i <= order; i++){
            accu += coeffs[i];
            coeffs_accu[i] = accu;
            coeffs_accu[2 * order - i]= -accu;
        }
    }
}

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(const int order, const float *half_coeffs, const int even){
    convolution_t *conv = (convolution_t *) malloc(sizeof(convolution_t));
    if(conv == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        exit(1);
    }
    conv->order = order;
    conv->coeffs = (float *) malloc((2 * order + 1) * sizeof(float));
    if(conv->coeffs == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        free(conv);
        exit(1);
    }
    conv->coeffs_accu = (float *) malloc((2 * order + 1) * sizeof(float));
    if(conv->coeffs_accu == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        free(conv->coeffs);
        free(conv);
        exit(1);
    }
    convolve_extract_coeffs(order, half_coeffs, conv->coeffs,conv->coeffs_accu, even);
    return conv;
}

static void convolve_vert_fast_3(image_t *dst, const image_t *src, const convolution_t *conv) {
    const int iterline = src->stride;
    const float *coeff = conv->coeffs;
    const __m128 mc0 = _mm_set1_ps(coeff[0]), mc1 = _mm_set1_ps(coeff[1]), mc2 = _mm_set1_ps(coeff[2]);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int x{}; x < iterline; x += 4) {
        __m128 tmpSrc[3];
        std::array<int, 3> idx = {{0, 1, 2}};
        const float *pSrc = src->c1 + x;
        float *pDst = dst->c1 + x;
        tmpSrc[idx[0]] = tmpSrc[idx[1]] = _mm_load_ps(pSrc);
        for (int y{src->height - 1}; y--;) {
            pSrc += src->stride;
            tmpSrc[idx[2]] = _mm_load_ps(pSrc);
            _mm_store_ps(pDst, _mm_add_ps(_mm_mul_ps(mc0, tmpSrc[idx[0]]), _mm_add_ps(_mm_mul_ps(mc1, tmpSrc[idx[1]]), _mm_mul_ps(mc2, tmpSrc[idx[2]]))));
            pDst += dst->stride;
            // rotate
            std::rotate(std::begin(idx), std::next(std::begin(idx), 1), std::end(idx));
        }
        tmpSrc[idx[2]] = tmpSrc[idx[1]];
        _mm_store_ps(pDst, _mm_add_ps(_mm_mul_ps(mc0, tmpSrc[idx[0]]), _mm_add_ps(_mm_mul_ps(mc1, tmpSrc[idx[1]]), _mm_mul_ps(mc2, tmpSrc[idx[2]]))));
    }
}

static void convolve_vert_fast_5(image_t *dst, const image_t *src, const convolution_t *conv) {
    const int iterline = src->stride;
    const float *coeff = conv->coeffs;
    const __m128 mc0 = _mm_set1_ps(coeff[0]), mc1 = _mm_set1_ps(coeff[1]), mc2 = _mm_set1_ps(coeff[2]), mc3 = _mm_set1_ps(coeff[3]), mc4 = _mm_set1_ps(coeff[4]);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int x{}; x < iterline; x += 4) {
        __m128 tmpSrc[5];
        std::array<int, 5> idx = {{0, 1, 2, 3, 4}};
        const float *pSrc = src->c1 + x;
        float *pDst = dst->c1 + x;
        tmpSrc[idx[0]] = tmpSrc[idx[1]] = tmpSrc[idx[2]] = _mm_load_ps(pSrc);
        pSrc += src->stride;
        tmpSrc[idx[3]] = _mm_load_ps(pSrc);
        for (int y{src->height - 2}; y--;) {
            pSrc += src->stride;
            tmpSrc[idx[4]] = _mm_load_ps(pSrc);
            _mm_store_ps(pDst, _mm_add_ps(_mm_mul_ps(mc0, tmpSrc[idx[0]]), _mm_add_ps(_mm_add_ps(_mm_mul_ps(mc1, tmpSrc[idx[1]]), _mm_mul_ps(mc2, tmpSrc[idx[2]])), _mm_add_ps(_mm_mul_ps(mc3, tmpSrc[idx[3]]), _mm_mul_ps(mc4, tmpSrc[idx[4]])))));
            pDst += dst->stride;
            // rotate
            std::rotate(std::begin(idx), std::next(std::begin(idx), 1), std::end(idx));
        }
        tmpSrc[idx[4]] = tmpSrc[idx[3]];
        _mm_store_ps(pDst, _mm_add_ps(_mm_mul_ps(mc0, tmpSrc[idx[0]]), _mm_add_ps(_mm_add_ps(_mm_mul_ps(mc1, tmpSrc[idx[1]]), _mm_mul_ps(mc2, tmpSrc[idx[2]])), _mm_add_ps(_mm_mul_ps(mc3, tmpSrc[idx[3]]), _mm_mul_ps(mc4, tmpSrc[idx[4]])))));
        pDst += dst->stride;
        std::rotate(std::begin(idx), std::next(std::begin(idx), 1), std::end(idx));
        tmpSrc[idx[4]] = tmpSrc[idx[3]];
        _mm_store_ps(pDst, _mm_add_ps(_mm_mul_ps(mc0, tmpSrc[idx[0]]), _mm_add_ps(_mm_add_ps(_mm_mul_ps(mc1, tmpSrc[idx[1]]), _mm_mul_ps(mc2, tmpSrc[idx[2]])), _mm_add_ps(_mm_mul_ps(mc3, tmpSrc[idx[3]]), _mm_mul_ps(mc4, tmpSrc[idx[4]])))));
    }
}

static void convolve_horiz_fast_3(image_t *dst, const image_t *src, const convolution_t *conv){
#define STEP 4
    const int iterline = src->stride >> 2;
    const float *coeff = conv->coeffs;
    float *dstp = dst->c1;
    // create shifted version of src
    image_t *tmp = image_new(src->stride + 4, 1);
    const __m128 mc0 = _mm_set1_ps(coeff[0]), mc1 = _mm_set1_ps(coeff[1]), mc2 = _mm_set1_ps(coeff[2]);
    for (int j{}; j < src->height; ++j) {
        float *ptr_tmp = tmp->c1;
        const float *srcptr = src->c1 + j * src->stride;
        ptr_tmp[0] = srcptr[0];
        memcpy(ptr_tmp + 1, srcptr , sizeof(float) * src->width);
        std::fill_n(std::next(ptr_tmp, src->width + 1), tmp->width - src->width - 1, srcptr[src->width - 1]);

        __m128 s0 = _mm_load_ps(ptr_tmp);
        for (int i{}; i < iterline; ++i) {
            const __m128 s4 = _mm_load_ps(ptr_tmp + 4);
            __m128 s2 = _mm_shuffle_ps(s4, s0, 0xE4);
            s2 = _mm_shuffle_ps(s2, s2, 0x4E);
            __m128 s1 = _mm_blend_ps(s4, s0, 0x0E);
            s1 = _mm_shuffle_ps(s1, s1, 0x39);
            _mm_store_ps(dstp ,_mm_add_ps(_mm_mul_ps(mc0, s0), _mm_add_ps(_mm_mul_ps(mc1, s1), _mm_mul_ps(mc2, s2))));
            s0 = s4;
            dstp += STEP; ptr_tmp += STEP;
        }
    }
    image_delete(tmp);
#undef STEP
}

static void convolve_horiz_fast_5(image_t *dst, const image_t *src, const convolution_t *conv){
#define STEP 4
    const int iterline = src->stride >> 2;
    const float *coeff = conv->coeffs;
    float *dstp = dst->c1;
    image_t *tmp = image_new(src->stride + 4, 1);
    const __m128 mc0 = _mm_set1_ps(coeff[0]), mc1 = _mm_set1_ps(coeff[1]), mc2 = _mm_set1_ps(coeff[2]), mc3 = _mm_set1_ps(coeff[3]), mc4 = _mm_set1_ps(coeff[4]);
    for (int j{}; j < src->height; ++j) {
        float *ptr_tmp = tmp->c1;
        const float *srcptr = src->c1 + j * src->stride;
        ptr_tmp[0] = ptr_tmp[1] = srcptr[0];
        memcpy(ptr_tmp + 2, srcptr , sizeof(float) * src->width);
        std::fill_n(std::next(ptr_tmp, src->width + 2), tmp->width - src->width - 2, srcptr[src->width - 1]);

        __m128 s0 = _mm_load_ps(ptr_tmp);
        for (int i{}; i < iterline; ++i) {
            const __m128 s4 = _mm_load_ps(ptr_tmp + 4);
            __m128 s2 = _mm_shuffle_ps(s4, s0, 0xE4);
            s2 = _mm_shuffle_ps(s2, s2, 0x4E);
            __m128 s1 = _mm_blend_ps(s4, s0, 0x0E);
            s1 = _mm_shuffle_ps(s1, s1, 0x39);
            __m128 s3 = _mm_blend_ps(s4, s0, 0x08);
            s3 = _mm_shuffle_ps(s3, s3, 0x93);
            _mm_store_ps(dstp, _mm_add_ps(_mm_mul_ps(mc0, s0), _mm_add_ps(_mm_add_ps(_mm_mul_ps(mc1, s1), _mm_mul_ps(mc2, s2)), _mm_add_ps(_mm_mul_ps(mc3, s3), _mm_mul_ps(mc4, s4)))));
            s0 = s4;
            dstp += STEP; ptr_tmp += STEP;
        }
    }
    image_delete(tmp);
#undef STEP
}

/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_t *src, const convolution_t *conv) {
    if (conv->order==1) {
        convolve_horiz_fast_3(dest,src,conv);
        return;
    } else if (conv->order==2) {
        convolve_horiz_fast_5(dest,src,conv);
        return;
    }
    float *in = src->c1;
    float * out = dest->c1;
    int i, j, ii;
    float *o = out;
    int i0 = -conv->order;
    int i1 = +conv->order;
    float *coeff = conv->coeffs + conv->order;
    float *coeff_accu = conv->coeffs_accu + conv->order;
    std::vector<float> tmp(conv->order * 2 + 1);
    for (j = 0; j < src->height; ++j) {
        const float *al = in + j * src->stride;
        const float *f0 = coeff + i0;
        float sum;
        std::copy_n(al, tmp.size() - 1, tmp.data());
        al = std::next(al, tmp.size() - 1);
        for (i = 0; i < -i0; ++i) {
            sum=coeff_accu[-i - 1] * tmp[0];
            for (ii = i1 + i; ii >= 0; --ii) {
                sum += coeff[ii - i] * tmp[ii];
            }
            *o++ = sum;
        }
        for (; i < src->width - i1; ++i) {
            tmp.back() = al[0];
            sum = 0;
            for (ii = i1 - i0; ii >= 0; --ii) {
                sum += f0[ii] * tmp[ii];
            }
            al++;
            *o++ = sum;
            std::rotate(std::begin(tmp), std::next(std::begin(tmp), 1), std::end(tmp));
        }
        for (; i < src->width; ++i) {
            sum = coeff_accu[src->width - i] * tmp[src->width - i0 - 1 - i];
            for (ii = src->width - i0 - 1 - i; ii >= 0; --ii) {
                sum += f0[ii] * tmp[ii];
            }
            *o++ = sum;
            std::rotate(std::begin(tmp), std::next(std::begin(tmp), 1), std::end(tmp));
        }
        for (; i < src->stride; ++i) {
            o++;
        }
    }
}

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_t *src, const convolution_t *conv) {
    if (conv->order==1) {
        convolve_vert_fast_3(dest,src,conv);
        return;
    } else if (conv->order==2) {
        convolve_vert_fast_5(dest,src,conv);
        return;
    }
    const float *coeff = conv->coeffs;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int x = 0; x < src->width; ++x) {
        const auto *pSrc = src->c1;
        auto *pDst = dest->c1;
        std::vector<float> tmp(conv->order * 2 + 1);
        std::fill_n(std::begin(tmp), conv->order + 1, pSrc[x]);
        for (int y = 1; y < conv->order; ++y) {
            pSrc += src->stride;
            tmp[conv->order + y] = pSrc[x];
        }
        for (int y = conv->order; y < src->height; ++y) {
            pSrc += src->stride;
            tmp.back() = pSrc[x];
            float sum{};
            for (int i = conv->order * 2 + 1; i--;) {
                sum += tmp[i] * coeff[i];
            }
            pDst[x] = sum;
            pDst += dest->stride;
            std::rotate(std::begin(tmp), std::next(std::begin(tmp), 1), std::end(tmp));
        }
        for (int y = 0; y <conv->order; ++y) {
            tmp.back() = pSrc[x];
            float sum{};
            for (int i = conv->order * 2 + 1; i--;) {
                sum += tmp[i] * coeff[i];
            }
            pDst[x] = sum;
            pDst += dest->stride;
            std::rotate(std::begin(tmp), std::next(std::begin(tmp), 1), std::end(tmp));
        }
    }
}

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv){
    if(conv) {
        free(conv->coeffs);
        free(conv->coeffs_accu);
        free(conv);
    }
}

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv){
    const int width = src->width, height = src->height, stride = src->stride;
    // separate channels of images
    image_t src_red = {width, height, stride, src->c1}, src_green = {width, height, stride, src->c2}, src_blue = {width, height, stride, src->c3},
            dst_red = {width, height, stride, dst->c1}, dst_green = {width, height, stride, dst->c2}, dst_blue = {width, height, stride, dst->c3};
    // horizontal and vertical
    if (horiz_conv != NULL && vert_conv != NULL) {
        // perform convolution for each channel
        convolve_horiz(&dst_red, &src_red, horiz_conv); 
        convolve_vert(&dst_red, &dst_red, vert_conv); 
        convolve_horiz(&dst_green, &src_green, horiz_conv);
        convolve_vert(&dst_green, &dst_green, vert_conv); 
        convolve_horiz(&dst_blue, &src_blue, horiz_conv); 
        convolve_vert(&dst_blue, &dst_blue, vert_conv);
    } else if (horiz_conv != NULL && vert_conv == NULL) { // only horizontal
        convolve_horiz(&dst_red, &src_red, horiz_conv);
        convolve_horiz(&dst_green, &src_green, horiz_conv);
        convolve_horiz(&dst_blue, &src_blue, horiz_conv);
    } else if (vert_conv != NULL && horiz_conv == NULL) { // only vertical
        convolve_vert(&dst_red, &src_red, vert_conv);
        convolve_vert(&dst_green, &src_green, vert_conv);
        convolve_vert(&dst_blue, &src_blue, vert_conv);
    }
}

/* perform horizontal and/or vertical convolution to a single band image*/
void image_convolve_hv(image_t *dst, const image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv)
{
    const int width = src->width, height = src->height, stride = src->stride;
    // separate channels of images
    image_t src_red = {width, height, stride, src->c1},
            dst_red = {width, height, stride, dst->c1};
    // horizontal and vertical
    if (horiz_conv != NULL && vert_conv != NULL) {
        // perform convolution for each channel
        convolve_horiz(&dst_red, &src_red, horiz_conv);
        convolve_vert(&dst_red, &dst_red, vert_conv);
    } else if (horiz_conv != NULL && vert_conv == NULL) { // only horizontal
        convolve_horiz(&dst_red, &src_red, horiz_conv);
    } else if (vert_conv != NULL && horiz_conv == NULL) { // only vertical
        convolve_vert(&dst_red, &src_red, vert_conv);
    }
}

/************ Pyramid **********/

/* create new color image pyramid structures */
static color_image_pyramid_t* color_image_pyramid_new(){
    color_image_pyramid_t* pyr = (color_image_pyramid_t*) malloc(sizeof(color_image_pyramid_t));
    if(pyr == NULL){
        fprintf(stderr,"Error in color_image_pyramid_new(): not enough memory\n");
        exit(1);
    }
    pyr->min_size = -1;
    pyr->scale_factor = -1.0f;
    pyr->size = -1;
    pyr->images = NULL;
    return pyr;
}

/* set the size of the color image pyramid structures (reallocate the array of pointers to images) */
static void color_image_pyramid_set_size(color_image_pyramid_t* pyr, const int size){
    if(size<0){
        fprintf(stderr,"Error in color_image_pyramid_set_size(): size is negative\n");
        exit(1);
    }
    if(pyr->images == NULL){
        pyr->images = (color_image_t**) malloc(sizeof(color_image_t*)*size);
    }else{
        pyr->images = (color_image_t**) realloc(pyr->images,sizeof(color_image_t*)*size);
    }
    if(pyr->images == NULL){
        fprintf(stderr,"Error in color_image_pyramid_set_size(): not enough memory\n");
        exit(1);      
    }
    pyr->size = size;
}

/* create a pyramid of color images using a given scale factor, stopping when one dimension reach min_size and with applying a gaussian smoothing of standard deviation spyr (no smoothing if 0) */
color_image_pyramid_t *color_image_pyramid_create(const color_image_t *src, const float scale_factor, const int min_size, const float spyr){
    const int nb_max_scale = 1000;
    // allocate structure
    color_image_pyramid_t *pyramid = color_image_pyramid_new();
    pyramid->min_size = min_size;
    pyramid->scale_factor = scale_factor;
    convolution_t *conv = NULL;
    if(spyr>0.0f){
        int fsize;
        float *filter_coef = gaussian_filter(spyr, &fsize);
        conv = convolution_new(fsize, filter_coef, 1);
        free(filter_coef);
    }
    color_image_pyramid_set_size(pyramid, nb_max_scale);
    pyramid->images[0] = color_image_cpy(src);
    int i;
    for( i=1 ; i<nb_max_scale ; i++){
        const int oldwidth = pyramid->images[i-1]->width, oldheight = pyramid->images[i-1]->height;
        const int newwidth = (int) (1.5f + (oldwidth-1) / scale_factor);
        const int newheight = (int) (1.5f + (oldheight-1) / scale_factor);
        if( newwidth <= min_size || newheight <= min_size){
            color_image_pyramid_set_size(pyramid, i);
            break;
        }
        if(spyr>0.0f){
            color_image_t* tmp = color_image_new(oldwidth, oldheight);
            color_image_convolve_hv(tmp,pyramid->images[i-1], conv, conv);
            pyramid->images[i]= color_image_resize_bilinear(tmp, scale_factor);
            color_image_delete(tmp);
        }else{
            pyramid->images[i] = color_image_resize_bilinear(pyramid->images[i-1], scale_factor);
        }
    }
    if(spyr>0.0f){
        convolution_delete(conv);
    }
    return pyramid;
}

/* delete the structure of a pyramid of color images and all the color images in it*/
void color_image_pyramid_delete(color_image_pyramid_t *pyr){
    if(pyr==NULL){
        return;
    }
    int i;
    for(i=0 ; i<pyr->size ; i++){
        color_image_delete(pyr->images[i]);
    }
    free(pyr->images);
    free(pyr);
}
