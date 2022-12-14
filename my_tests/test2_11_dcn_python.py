
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np


def DmcnIm2colBilinear(bottom_data, data_im_ptr, data_width, height, width, h, w):
    h_low = np.floor(h)
    w_low = np.floor(w)
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0.
    if h_low >= 0 and w_low >= 0:
        v1 = bottom_data[int(data_im_ptr + h_low * data_width + w_low)]
    v2 = 0.
    if h_low >= 0 and w_high <= width - 1:
        v2 = bottom_data[int(data_im_ptr + h_low * data_width + w_high)]
    v3 = 0.
    if h_high <= height - 1 and w_low >= 0:
        v3 = bottom_data[int(data_im_ptr + h_high * data_width + w_low)]
    v4 = 0.
    if h_high <= height - 1 and w_high <= width - 1:
        v4 = bottom_data[int(data_im_ptr + h_high * data_width + w_high)]

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4



def DmcnIm2colBilinear2(input, data_im_ptr, W, H, W2, h_im, w_im):
    h_low = np.floor(h_im)
    w_low = np.floor(w_im)
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h_im - h_low
    lw = w_im - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0.
    if h_low >= 0 and w_low >= 0:
        v1 = input[int(data_im_ptr + h_low * W + w_low)]
    v2 = 0.
    if h_low >= 0 and w_high <= W - 1:
        v2 = input[int(data_im_ptr + h_low * W + w_high)]
    v3 = 0.
    if h_high <= H - 1 and w_low >= 0:
        v3 = input[int(data_im_ptr + h_high * W + w_low)]
    v4 = 0.
    if h_high <= H - 1 and w_high <= W - 1:
        v4 = input[int(data_im_ptr + h_high * W + w_high)]

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


class DeformableConvCPUKernel(object):
    def __init__(self):
        super(DeformableConvCPUKernel, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
??????DCN???????????????
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # im2col?????????output = weight * im2col
        # weight??????reshape???[out_C, in_C * kH * kW]???im2col?????????[in_C * kH * kW, out_H * out_W]
        # ?????????????????????????????????[out_C, out_H * out_W]????????????????????????????????????

        # im2col??????????????????[in_C * kH * kW, out_H * out_W]?????????, ????????????C++???????????????
        im2col = np.zeros((in_C * kH * kW * out_H * out_W, ), dtype=np.float32)
        im2col_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # im2col_ncnn??????????????????[out_H * out_W, in_C * kH * kW]????????????
        # ncnn???????????????A*B=C??????A?????????[m, n]???B?????????[k, n]???C?????????[m, k]??????B???????????????????????????????????????
        im2col_ncnn = np.zeros((out_H * out_W * in_C * kH * kW, ), dtype=np.float32)
        im2col_ncnn_ptr = 0

        # im2col
        # ????????????????????????????????????in_C????????????
        for c_im in range(in_C):
            for h_col in range(out_H):
                for w_col in range(out_W):
                    c_col = c_im * kH * kW   # ????????????[in_C*kH*kW, ...]????????? ??????c_im==???????????? ???0?????????????????????miemie2013: ????????????????????????????????????

                    h_in = h_col * stride_h - pad_top    # ?????????(?????????)??????????????????????????????y??????
                    w_in = w_col * stride_w - pad_left   # ?????????(?????????)??????????????????????????????x??????

                    '''
            // ??????1?????????data_col_ptr???????????????im2col????????????+????????????????????????3D???????????????1???1D????????????
            // ???????????????[in_C*kH*kW, out_H, out_W]????????? reshape ???[in_C*kH*kW * out_H * out_W, ]?????????????????????????????????
            // miemie2013: ??????????????????????????????????????????0????????????c_col????????????0????????????in_C*kH*kW
            // ??????[c_col, h_col, w_col] ?????? xxx
                    '''
                    data_col_ptr = im2col_ptr + (c_col * out_H + h_col) * out_W + w_col
                    data_col_ptr2 = im2col_ncnn_ptr + (h_col * out_W + w_col) * in_C * kH * kW + c_col

                    data_im_ptr = data_im + c_im * H * W
                    data_offset_ptr = data_offset
                    data_mask_ptr = data_mask

                    # ?????????????????????????????????
                    for i in range(kH):
                        for j in range(kW):
                            '''
                // data_offset_ptr?????????[kH*kW*2, out_H, out_W], ???0??????yxyxyx...???????????????
                // reshape data_offset_ptr?????????[kH, kW, 2, out_H, out_W]
                // ???????????? 5D??????[i, j, 0, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????y??????
                            '''
                            data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                            '''
                // ???????????? 5D??????[i, j, 1, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????x??????
                            '''
                            data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                            '''
                // data_mask_ptr?????????[kH*kW, out_H, out_W],
                // reshape data_mask_ptr?????????[kH, kW, out_H, out_W]
                // ???????????? 4D??????[i, j, h_col, w_col] ?????? 1D??????data_mask_hw_ptr ??????????????????mask????????????
                            '''
                            data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                            offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                            offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                            mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                            val = 0.
                            h_im = h_in + i * dilation_h + offset_h
                            w_im = w_in + j * dilation_w + offset_w
                            if h_im > -1 and w_im > -1 and h_im < H and w_im < W:
                                val = DmcnIm2colBilinear(input, data_im_ptr, W, H, W, h_im, w_im)
                            im2col[data_col_ptr] = val * mask_
                            im2col_ncnn[data_col_ptr2] = val * mask_
                            data_col_ptr += out_H * out_W
                            data_col_ptr2 += 1
        # ???im2col???????????????
        im2col = np.reshape(im2col, (in_C * kH * kW, out_H * out_W))   # [in_C * kH * kW, out_H * out_W]
        im2col_ncnn = np.reshape(im2col_ncnn, (out_H * out_W, in_C * kH * kW))
        aaaaaaaaa1 = im2col.T
        aaaaaaaaa2 = im2col_ncnn
        ddd1 = np.sum((aaaaaaaaa1 - aaaaaaaaa2) ** 2)
        print('ddd=%.9f' % ddd1)



        # ncnn_output = '../build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, aaaaaaaaa2.shape)
        #
        # yy1 = y
        # yy2 = aaaaaaaaa2
        # ddd = np.sum((yy1 - yy2) ** 2)
        # print('ddd=%.9f' % ddd)


        weight = np.reshape(weight, (out_C, in_C * kH * kW))   # [out_C, in_C * kH * kW]
        output = np.matmul(weight, im2col)   # [out_C, out_H * out_W]
        if bias_term:
            bias = np.reshape(bias, (out_C, 1))   # [out_C, 1]
            output += bias
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output

class DeformableConvCPUKernelv2(object):
    def __init__(self):
        super(DeformableConvCPUKernelv2, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
??????DCN???????????????
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # im2col?????????output = im2col * weight_t
        # im2col?????????[out_H * out_W, in_C * kH * kW]
        # weight??????reshape???[out_C, in_C * kH * kW]???????????????weight_t.shape=[in_C * kH * kW, out_C]
        # ?????????????????????????????????[out_H * out_W, out_C]????????????????????????????????????

        # im2col??????????????????[out_H * out_W, in_C * kH * kW]?????????, ????????????C++???????????????
        im2col = np.zeros((out_H * out_W * in_C * kH * kW, ), dtype=np.float32)
        im2col_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # im2col
        # ????????????????????????????????????in_C????????????
        for c_im in range(in_C):
            for h_col in range(out_H):
                for w_col in range(out_W):
                    c_col = c_im * kH * kW   # ????????????[in_C*kH*kW, ...]????????? ??????c_im==???????????? ???0?????????????????????miemie2013: ????????????????????????????????????

                    h_in = h_col * stride_h - pad_top    # ?????????(?????????)??????????????????????????????y??????
                    w_in = w_col * stride_w - pad_left   # ?????????(?????????)??????????????????????????????x??????

                    '''
            // ??????1?????????data_col_ptr???????????????im2col????????????+????????????????????????3D???????????????1???1D????????????
            // ???????????????[in_C*kH*kW, out_H, out_W]????????? reshape ???[in_C*kH*kW * out_H * out_W, ]?????????????????????????????????
            // miemie2013: ??????????????????????????????????????????0????????????c_col????????????0????????????in_C*kH*kW
            // ??????[c_col, h_col, w_col] ?????? xxx
                    '''
                    data_col_ptr = im2col_ptr + (h_col * out_W + w_col) * in_C * kH * kW + c_col

                    data_im_ptr = data_im + c_im * H * W
                    data_offset_ptr = data_offset
                    data_mask_ptr = data_mask

                    # ?????????????????????????????????
                    for i in range(kH):
                        for j in range(kW):
                            '''
                // data_offset_ptr?????????[kH*kW*2, out_H, out_W], ???0??????yxyxyx...???????????????
                // reshape data_offset_ptr?????????[kH, kW, 2, out_H, out_W]
                // ???????????? 5D??????[i, j, 0, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????y??????
                            '''
                            data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                            '''
                // ???????????? 5D??????[i, j, 1, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????x??????
                            '''
                            data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                            '''
                // data_mask_ptr?????????[kH*kW, out_H, out_W],
                // reshape data_mask_ptr?????????[kH, kW, out_H, out_W]
                // ???????????? 4D??????[i, j, h_col, w_col] ?????? 1D??????data_mask_hw_ptr ??????????????????mask????????????
                            '''
                            data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                            offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                            offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                            mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                            val = 0.
                            h_im = h_in + i * dilation_h + offset_h
                            w_im = w_in + j * dilation_w + offset_w
                            if h_im > -1 and w_im > -1 and h_im < H and w_im < W:
                                val = DmcnIm2colBilinear(input, data_im_ptr, W, H, W, h_im, w_im)
                            im2col[data_col_ptr] = val * mask_
                            data_col_ptr += 1
        # ???im2col???????????????
        im2col = np.reshape(im2col, (out_H * out_W, in_C * kH * kW))   # [out_H * out_W, in_C * kH * kW]



        # ncnn_output = '../build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, aaaaaaaaa2.shape)
        #
        # yy1 = y
        # yy2 = aaaaaaaaa2
        # ddd = np.sum((yy1 - yy2) ** 2)
        # print('ddd=%.9f' % ddd)


        weight = np.reshape(weight, (out_C, in_C * kH * kW))   # [out_C, in_C * kH * kW]
        weight_t = weight.T     # [in_C * kH * kW, out_C]
        output = np.matmul(im2col, weight_t)   # [out_H * out_W, out_C]
        if bias_term:
            bias = np.reshape(bias, (1, out_C))   # [1, out_C]
            output += bias
        output = output.T   # [out_C, out_H * out_W]
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output

class DeformableConvCPUKernelv3(object):
    def __init__(self):
        super(DeformableConvCPUKernelv3, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
??????DCN???????????????
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # im2col?????????output = im2col * weight2
        # im2col?????????[out_H * out_W, kH * kW * in_C]
        # weight??????reshape???[out_C, in_C, kH * kW]???????????????[out_C, kH * kW, in_C]??????reshape???[out_C, kH * kW * in_C]???????????????[kH * kW * in_C, out_C]
        # ?????????????????????????????????[out_H * out_W, out_C]????????????????????????????????????

        # im2col??????????????????[out_H * out_W, in_C * kH * kW]?????????, ????????????C++???????????????
        im2col = np.zeros((out_H * out_W * kH * kW * in_C, ), dtype=np.float32)
        im2col_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # im2col
        # ????????????????????????????????????in_C????????????
        for h_col in range(out_H):
            for w_col in range(out_W):
                h_in = h_col * stride_h - pad_top    # ?????????(?????????)??????????????????????????????y??????
                w_in = w_col * stride_w - pad_left   # ?????????(?????????)??????????????????????????????x??????

                # im2col????????????[out_H, out_W, kH, kW, in_C]??????????????????[h_col, w_col, 0, 0, 0]?????????1D??????
                data_col_ptr = im2col_ptr + (h_col * out_W + w_col) * kH * kW * in_C

                data_offset_ptr = data_offset
                data_mask_ptr = data_mask

                # ?????????????????????????????????
                for i in range(kH):
                    for j in range(kW):
                        '''
            // data_offset_ptr?????????[kH*kW*2, out_H, out_W], ???0??????yxyxyx...???????????????
            // reshape data_offset_ptr?????????[kH, kW, 2, out_H, out_W]
            // ???????????? 5D??????[i, j, 0, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????y??????
                        '''
                        data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                        '''
            // ???????????? 5D??????[i, j, 1, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????x??????
                        '''
                        data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                        '''
            // data_mask_ptr?????????[kH*kW, out_H, out_W],
            // reshape data_mask_ptr?????????[kH, kW, out_H, out_W]
            // ???????????? 4D??????[i, j, h_col, w_col] ?????? 1D??????data_mask_hw_ptr ??????????????????mask????????????
                        '''
                        data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                        offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                        offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                        mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                        h_im = h_in + i * dilation_h + offset_h
                        w_im = w_in + j * dilation_w + offset_w
                        cond = h_im > -1 and w_im > -1 and h_im < H and w_im < W
                        if cond:
                            # Bilinear
                            h_low = np.floor(h_im)
                            w_low = np.floor(w_im)
                            h_high = h_low + 1
                            w_high = w_low + 1

                            lh = h_im - h_low
                            lw = w_im - w_low
                            hh = 1 - lh
                            hw = 1 - lw

                            v1_cond = h_low >= 0 and w_low >= 0
                            v2_cond = h_low >= 0 and w_high <= W - 1
                            v3_cond = h_high <= H - 1 and w_low >= 0
                            v4_cond = h_high <= H - 1 and w_high <= W - 1

                            if v1_cond:
                                v1_pos = h_low * W + w_low
                            if v2_cond:
                                v2_pos = h_low * W + w_high
                            if v3_cond:
                                v3_pos = h_high * W + w_low
                            if v4_cond:
                                v4_pos = h_high * W + w_high

                            w1 = hh * hw
                            w2 = hh * lw
                            w3 = lh * hw
                            w4 = lh * lw

                        # ???????????????????????????[in_C, H, W]
                        data_im_ptr = data_im
                        for c_im in range(in_C):
                            val = 0.
                            if cond:
                                # Bilinear
                                v1 = 0.
                                if v1_cond:
                                    v1 = input[int(data_im_ptr + v1_pos)]
                                v2 = 0.
                                if v2_cond:
                                    v2 = input[int(data_im_ptr + v2_pos)]
                                v3 = 0.
                                if v3_cond:
                                    v3 = input[int(data_im_ptr + v3_pos)]
                                v4 = 0.
                                if v4_cond:
                                    v4 = input[int(data_im_ptr + v4_pos)]
                                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                            im2col[data_col_ptr] = val * mask_
                            data_col_ptr += 1
                            data_im_ptr += H*W
        # ???im2col???????????????
        im2col = np.reshape(im2col, (out_H * out_W, kH * kW * in_C))   # [out_H * out_W, kH * kW * in_C]



        # ncnn_output = '../build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, aaaaaaaaa2.shape)
        #
        # yy1 = y
        # yy2 = aaaaaaaaa2
        # ddd = np.sum((yy1 - yy2) ** 2)
        # print('ddd=%.9f' % ddd)


        # weight??????reshape???[out_C, in_C, kH * kW]???????????????[out_C, kH * kW, in_C]??????reshape???[out_C, kH * kW * in_C]???????????????[kH * kW * in_C, out_C]
        weight2 = np.reshape(weight, (out_C, in_C, kH * kW))
        weight2 = weight2.transpose(0, 2, 1)
        weight2 = np.reshape(weight2, (out_C, kH * kW * in_C))   # [out_C, kH * kW * in_C]
        weight2 = weight2.T     # [kH * kW * in_C, out_C]
        output = np.matmul(im2col, weight2)   # [out_H * out_W, out_C]
        if bias_term:
            bias = np.reshape(bias, (1, out_C))   # [1, out_C]
            output += bias
        output = output.T   # [out_C, out_H * out_W]
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output

class DeformableConvCPUKernel_naive(object):
    def __init__(self):
        super(DeformableConvCPUKernel_naive, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
??????DCN???????????????
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # output.shape is [num_output, out_h, out_w] (in python).
        output = np.zeros((out_W * out_H * num_output, ), dtype=np.float32)
        output_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # deformable conv
        # ????????????????????????????????????in_C????????????
        for h_col in range(out_H):
            for w_col in range(out_W):
                h_in = h_col * stride_h - pad_top    # ?????????(?????????)??????????????????????????????y??????
                w_in = w_col * stride_w - pad_left   # ?????????(?????????)??????????????????????????????x??????

                output_hw_ptr = output_ptr + (h_col * out_W + w_col)
                for oc in range(out_C):
                    sum = 0
                    if bias_term:
                        sum = bias[oc]
                    data_offset_ptr = data_offset
                    data_mask_ptr = data_mask

                    # ?????????????????????????????????
                    for i in range(kH):
                        for j in range(kW):
                            '''
                // data_offset_ptr?????????[kH*kW*2, out_H, out_W], ???0??????yxyxyx...???????????????
                // reshape data_offset_ptr?????????[kH, kW, 2, out_H, out_W]
                // ???????????? 5D??????[i, j, 0, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????y??????
                            '''
                            data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                            '''
                // ???????????? 5D??????[i, j, 1, h_col, w_col] ?????? 1D??????data_offset_h_ptr ??????????????????x??????
                            '''
                            data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                            '''
                // data_mask_ptr?????????[kH*kW, out_H, out_W],
                // reshape data_mask_ptr?????????[kH, kW, out_H, out_W]
                // ???????????? 4D??????[i, j, h_col, w_col] ?????? 1D??????data_mask_hw_ptr ??????????????????mask????????????
                            '''
                            data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                            offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                            offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                            mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                            h_im = h_in + i * dilation_h + offset_h
                            w_im = w_in + j * dilation_w + offset_w
                            cond = h_im > -1 and w_im > -1 and h_im < H and w_im < W
                            if cond:
                                # Bilinear
                                h_low = np.floor(h_im)
                                w_low = np.floor(w_im)
                                h_high = h_low + 1
                                w_high = w_low + 1

                                lh = h_im - h_low
                                lw = w_im - w_low
                                hh = 1 - lh
                                hw = 1 - lw

                                v1_cond = h_low >= 0 and w_low >= 0
                                v2_cond = h_low >= 0 and w_high <= W - 1
                                v3_cond = h_high <= H - 1 and w_low >= 0
                                v4_cond = h_high <= H - 1 and w_high <= W - 1

                                if v1_cond:
                                    v1_pos = h_low * W + w_low
                                if v2_cond:
                                    v2_pos = h_low * W + w_high
                                if v3_cond:
                                    v3_pos = h_high * W + w_low
                                if v4_cond:
                                    v4_pos = h_high * W + w_high

                                w1 = hh * hw
                                w2 = hh * lw
                                w3 = lh * hw
                                w4 = lh * lw

                            # ???????????????????????????[in_C, H, W]
                            data_im_ptr = data_im
                            for c_im in range(in_C):
                                val = 0.
                                if cond:
                                    # Bilinear
                                    v1 = 0.
                                    if v1_cond:
                                        v1 = input[int(data_im_ptr + v1_pos)]
                                    v2 = 0.
                                    if v2_cond:
                                        v2 = input[int(data_im_ptr + v2_pos)]
                                    v3 = 0.
                                    if v3_cond:
                                        v3 = input[int(data_im_ptr + v3_pos)]
                                    v4 = 0.
                                    if v4_cond:
                                        v4 = input[int(data_im_ptr + v4_pos)]
                                    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                                    print(v1)
                                    print(v2)
                                    print(v3)
                                    print(v4)
                                    print(w1)
                                    print(w2)
                                    print(w3)
                                    print(w4)
                                sum += val * mask_ * weight[oc, c_im, i, j]
                                print(val)
                                print(mask_)
                                print(weight[oc, c_im, i, j])
                                print()
                                data_im_ptr += H*W
                    output[output_hw_ptr] = sum
                    output_hw_ptr += out_H * out_W
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output


class ConvNormLayer2(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer2, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False)
            self.conv_w_lr = lr
            # ???????????????
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)
        else:
            self.offset_channel = 2 * filter_size ** 2
            self.mask_channel = filter_size ** 2

            self.conv_offset = nn.Conv2d(
                in_channels=ch_in,
                out_channels=3 * filter_size ** 2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                bias=True)
            # ???????????????
            torch.nn.init.constant_(self.conv_offset.weight, 0.0)
            torch.nn.init.constant_(self.conv_offset.bias, 0.0)

            # ????????????DCNv2
            # self.conv = MyDCNv2(
            #     in_channels=ch_in,
            #     out_channels=ch_out,
            #     kernel_size=filter_size,
            #     stride=stride,
            #     padding=(filter_size - 1) // 2,
            #     dilation=1,
            #     groups=groups,
            #     bias=True)
            # ??????DCN
            self.conv = torchvision.ops.DeformConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                bias=True)

            self.dcn_w_lr = lr
            # ???????????????
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)

        self.freeze_norm = freeze_norm
        norm_lr = 0. if freeze_norm else lr
        self.norm_lr = norm_lr
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)
        return offset, mask, out


ch_in = 2
ch_out = 2

filter_size = 1
stride = 1

# filter_size = 1
# stride = 2
#
# filter_size = 2
# stride = 1
#
# filter_size = 2
# stride = 2
#
# filter_size = 3
# stride = 1

# filter_size = 3
# stride = 2
#
# filter_size = 4
# stride = 1
#
# filter_size = 4
# stride = 2
#
# filter_size = 5
# stride = 1
#
# filter_size = 5
# stride = 2


model = ConvNormLayer2(ch_in, ch_out, filter_size=filter_size, stride=stride, dcn_v2=True)
torch.nn.init.normal_(model.conv_offset.weight)
torch.nn.init.normal_(model.conv_offset.bias)
torch.nn.init.normal_(model.conv.weight)
torch.nn.init.normal_(model.conv.bias)
# torch.nn.init.normal_(model.norm.weight)
# torch.nn.init.normal_(model.norm.bias)
# torch.nn.init.normal_(model.norm.running_mean)
# torch.nn.init.constant_(model.norm.running_var, 2.3)
model.eval()
# state_dict = torch.load('11.pth', map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)


# aaaaaaaaa = cv2.imread('my_test32.jpg')
aaaaaaaaa = cv2.imread('my_test2_1.jpg')
# aaaaaaaaa = cv2.imread('my_test2.jpg')
aaaaaaaaa = aaaaaaaaa.astype(np.float32)

mean = [117.3, 126.5, 130.2]
std = [108.4, 117.3, 127.6]
mean = np.array(mean)[np.newaxis, np.newaxis, :]
std = np.array(std)[np.newaxis, np.newaxis, :]
aaaaaaaaa -= mean
aaaaaaaaa /= std


x = torch.from_numpy(aaaaaaaaa)
x = x.to(torch.float32)
x = x.permute((2, 0, 1))
x = torch.unsqueeze(x, 0)
x.requires_grad_(False)
x = x[:, :2, :, :]

offset, mask, y = model(x)

x = x.cpu().detach().numpy()
offset = offset.cpu().detach().numpy()
mask = mask.cpu().detach().numpy()

dcn_w = model.conv.weight.cpu().detach().numpy()
dcn_b = model.conv.bias.cpu().detach().numpy()
# deformableConvCPUKernel = DeformableConvCPUKernel()
# deformableConvCPUKernel = DeformableConvCPUKernelv2()
deformableConvCPUKernel = DeformableConvCPUKernelv3()
deformableConvCPUKernel222 = DeformableConvCPUKernel_naive()


ncnn_output = '../build/tests/input.txt'
with open(ncnn_output, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
line = line[:-1]
ss = line.split(',')
aaa = []
for s in ss:
    aaa.append(float(s))
aaa = np.array(aaa).astype(np.float32)
xx = np.reshape(aaa, x[0].shape)


ncnn_output = '../build/tests/mask.txt'
with open(ncnn_output, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
line = line[:-1]
ss = line.split(',')
aaa = []
for s in ss:
    aaa.append(float(s))
aaa = np.array(aaa).astype(np.float32)
maskk = np.reshape(aaa, mask[0].shape)


ncnn_output = '../build/tests/offset.txt'
with open(ncnn_output, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
line = line[:-1]
ss = line.split(',')
aaa = []
for s in ss:
    aaa.append(float(s))
aaa = np.array(aaa).astype(np.float32)
offsett = np.reshape(aaa, offset[0].shape)


ncnn_output = '../build/tests/weight.txt'
with open(ncnn_output, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
line = line[:-1]
ss = line.split(',')
aaa = []
for s in ss:
    aaa.append(float(s))
aaa = np.array(aaa).astype(np.float32)
dcn_ww = np.reshape(aaa, dcn_w.shape)


dcn_b *= 0

# y2 = deformableConvCPUKernel(x[0], offset[0], mask[0], dcn_w, dcn_b, stride=stride)
y2 = deformableConvCPUKernel(xx, offsett, maskk, dcn_ww, dcn_b, stride=stride)
aaaaaaaaaaaaaaaaaaaaaaa = deformableConvCPUKernel222(xx, offsett, maskk, dcn_ww, dcn_b, stride=stride)


# yy1 = y.cpu().detach().numpy()
yy1 = aaaaaaaaaaaaaaaaaaaaaaa
yy2 = y2
ddd = np.sum((yy1 - yy2) ** 2)
print('ddd=%.9f' % ddd)


print()
