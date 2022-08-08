// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "deformableconv2d_x86.h"
#include <iostream>
#if __SSE2__
#include <emmintrin.h>
#if __SSE4_1__
#include <smmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE4_1__
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {


static void ppppp222(const ncnn::Mat& m, const char* name)
{
    int dims = m.dims;
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    size_t elemsize = m.elemsize;
    int elempack = m.elempack;
    size_t cstep = m.cstep;
    printf("%s shape dims=%d, C=%d, D=%d, H=%d, W=%d, elemsize=%d, elempack=%d, cstep=%d\n", name, dims, C, D, H, W, (int)elemsize, elempack, (int)cstep);
}


#include "deformableconv2d_sgemm.h"

#if __SSE2__
#include "deformableconv2d_pack4.h"
#include "deformableconv2d_pack1to4.h"
#include "deformableconv2d_pack4to1.h"

#include "deformableconv2d_sgemm_pack4.h"
#include "deformableconv2d_sgemm_pack1to4.h"
#include "deformableconv2d_sgemm_pack4to1.h"

#if __AVX__
#include "deformableconv2d_pack8.h"
#include "deformableconv2d_pack4to8.h"
#include "deformableconv2d_pack1to8.h"
#include "deformableconv2d_pack8to4.h"
#include "deformableconv2d_pack8to1.h"

#include "deformableconv2d_sgemm_pack8.h"
#include "deformableconv2d_sgemm_pack4to8.h"
#include "deformableconv2d_sgemm_pack1to8.h"
#include "deformableconv2d_sgemm_pack8to4.h"
#include "deformableconv2d_sgemm_pack8to1.h"

#if __AVX512F__
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

DeformableConv2D_x86::DeformableConv2D_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

    activation = 0;
}

static void deformableconv2d_transform_kernel_packed_sse(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }
}

int DeformableConv2D_x86::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        elempack = num_input % 16 == 0 ? 16 : num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

#if __SSE2__
#if __AVX__
#if __AVX512F__
#endif // __AVX512F__

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack8.h
            convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack4to8.h
            convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack1to8.h
            convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack8to4.h
            convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack8to1.h
            convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif // __AVX__

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                            || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                            || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack4.h
            convolution_im2col_sgemm_transform_kernel_pack4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                            || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                            || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack1to4.h
            convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                            || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                            || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack4to1.h
            convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            deformableconv2d_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif // __SSE2__

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm.h
            convolution_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            weight_data_tm = weight_data;
        }
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int DeformableConv2D_x86::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int DeformableConv2D_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    Mat& top_blob = top_blobs[0];
    printf("-------------- DeformableConv2D_x86 --------------\n");
    std::cout << this->name << std::endl;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    printf("elempack=%d\n", elempack);
    ppppp222(bottom_blob, "bottom_blob");

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int out_w = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
    const int out_h = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

    int out_elempack = 1;
#if __SSE2__
    printf("-------------- __SSE2__ --------------\n");
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        printf("-------------- __AVX512F__ --------------\n");
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
        printf("out_elempack=%d\n", out_elempack);
#elif __AVX__
        printf("-------------- __AVX__ --------------\n");
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
        printf("out_elempack=%d\n", out_elempack);
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
        printf("out_elempack=%d\n", out_elempack);
#endif
    }
#endif // __SSE2__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(out_w, out_h, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
#endif // __AVX512F__

    if (elempack == 8 && out_elempack == 8)
    {
        printf("-------------- 8to8 --------------\n");
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack8.h
            deformableconv2d_im2col_sgemm_pack8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack8.h
            deformableconv2d_pack8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        printf("-------------- 1to8 --------------\n");
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack1to8.h
            deformableconv2d_im2col_sgemm_pack1to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack1to8.h
            deformableconv2d_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack4to8.h
            deformableconv2d_im2col_sgemm_pack4to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack4to8.h
            deformableconv2d_pack4to8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack8to1.h
            deformableconv2d_im2col_sgemm_pack8to1_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack8to1.h
            deformableconv2d_pack8to1_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack8to4.h
            deformableconv2d_im2col_sgemm_pack8to4_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack8to4.h
            deformableconv2d_pack8to4_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __AVX__

    if (elempack == 4 && out_elempack == 4)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                            || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                            || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack4.h
            deformableconv2d_im2col_sgemm_pack4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack4.h
            deformableconv2d_pack4_sse(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                            || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                            || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack1to4.h
            deformableconv2d_im2col_sgemm_pack1to4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack1to4.h
            deformableconv2d_pack1to4_sse(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                            || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                            || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            // from src/layer/x86/deformableconv2d_sgemm_pack4to1.h
            deformableconv2d_im2col_sgemm_pack4to1_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // from src/layer/x86/deformableconv2d_pack4to1.h
            deformableconv2d_pack4to1_sse(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_sgemm_convolution)
        {
            // from src/layer/x86/deformableconv2d_sgemm.h
            deformableconv2d_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            const int maxk = kernel_w * kernel_h;

            // kernel offsets
            std::vector<int> _space_ofs(maxk);
            int* space_ofs = &_space_ofs[0];
            {
                int p1 = 0;
                int p2 = 0;
                int gap = w * dilation_h - kernel_w * dilation_w;
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        space_ofs[p1] = p2;
                        p1++;
                        p2 += dilation_w;
                    }
                    p2 += gap;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < out_h; i++)
                {
                    for (int j = 0; j < out_w; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_tm + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float wt = kptr[k];
                                sum += val * wt;
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += out_w;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
