/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef BLUESTEIN_H
#define BLUESTEIN_H

#include "common.h"
#include "rocfft_hip.h"

template <typename T>
__global__ void chirp_device(
    const size_t N, const size_t M, T* output, T* twiddles_large, const int twl, const int dir)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    T val = lib_make_vector2<T>(0, 0);

    if(twl == 1)
        val = TWLstep1(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 2)
        val = TWLstep2(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 3)
        val = TWLstep3(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 4)
        val = TWLstep4(twiddles_large, (tx * tx) % (2 * N));

    val.y *= (real_type_t<T>)(dir);

    if(tx == 0)
    {
        output[tx]     = val;
        output[tx + M] = val;
    }
    else if(tx < N)
    {
        output[tx]     = val;
        output[tx + M] = val;

        output[M - tx]     = val;
        output[M - tx + M] = val;
    }
    else if(tx <= (M - N))
    {
        output[tx]     = lib_make_vector2<T>(0, 0);
        output[tx + M] = lib_make_vector2<T>(0, 0);
    }
}

template <typename T>
__global__ void mul_device(const size_t  numof,
                           const size_t  totalWI,
                           const size_t  N,
                           const size_t  M,
                           const T*      input,
                           T*            output,
                           const size_t  dim,
                           const size_t* lengths,
                           const size_t* stride_in,
                           const size_t* stride_out,
                           const int     dir,
                           const int     scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx = tx % numof;
    if(scheme == 0)
    {
        output += oOffset;

        T out        = output[tx];
        output[tx].x = input[tx].x * out.x - input[tx].y * out.y;
        output[tx].y = input[tx].x * out.y + input[tx].y * out.x;
    }
    else if(scheme == 1)
    {
        T* chirp = output;

        input += iOffset;

        output += M;
        output += oOffset;

        if(tx < N)
        {
            output[tx].x = input[tx].x * chirp[tx].x + input[tx].y * chirp[tx].y;
            output[tx].y = -input[tx].x * chirp[tx].y + input[tx].y * chirp[tx].x;
        }
        else
        {
            output[tx] = lib_make_vector2<T>(0, 0);
        }
    }
    else if(scheme == 2)
    {
        const T* chirp = input;

        input += 2 * M;
        input += iOffset;

        output += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        output[tx].x      = MI * (input[tx].x * chirp[tx].x + input[tx].y * chirp[tx].y);
        output[tx].y      = MI * (-input[tx].x * chirp[tx].y + input[tx].y * chirp[tx].x);
    }
}

template <typename T>
__global__ void mul_device(const size_t  numof,
                           const size_t  totalWI,
                           const size_t  N,
                           const size_t  M,
                           const real_type_t<T>* inputRe,
                           const real_type_t<T>* inputIm,
                           T*            output,
                           const size_t  dim,
                           const size_t* lengths,
                           const size_t* stride_in,
                           const size_t* stride_out,
                           const int     dir,
                           const int     scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx = tx % numof;
    if(scheme == 0)
    {
        output += oOffset;

        T out        = output[tx];
        output[tx].x = inputRe[tx] * out.x - inputIm[tx] * out.y;
        output[tx].y = inputRe[tx] * out.y + inputIm[tx] * out.x;
    }
    else if(scheme == 1)
    {
        T* chirp = output;

        inputRe += iOffset;
        inputIm += iOffset;

        output += M;
        output += oOffset;

        if(tx < N)
        {
            output[tx].x = inputRe[tx] * chirp[tx].x + inputIm[tx] * chirp[tx].y;
            output[tx].y = -inputRe[tx] * chirp[tx].y + inputIm[tx] * chirp[tx].x;
        }
        else
        {
            output[tx] = lib_make_vector2<T>(0, 0);
        }
    }
    else if(scheme == 2)
    {
        const real_type_t<T>* chirpRe = inputRe;
        const real_type_t<T>* chirpIm = inputIm;

        inputRe += 2 * M;
        inputRe += iOffset;

        inputIm += 2 * M;
        inputIm += iOffset;

        output += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        output[tx].x      = MI * (inputRe[tx] * chirpRe[tx] + inputIm[tx] * chirpIm[tx]);
        output[tx].y      = MI * (-inputRe[tx] * chirpIm[tx] + inputIm[tx] * chirpRe[tx]);
    }
}

template <typename T>
__global__ void mul_device(const size_t  numof,
                           const size_t  totalWI,
                           const size_t  N,
                           const size_t  M,
                           const T*      input,
                           real_type_t<T>* outputRe,
                           real_type_t<T>* outputIm,
                           const size_t  dim,
                           const size_t* lengths,
                           const size_t* stride_in,
                           const size_t* stride_out,
                           const int     dir,
                           const int     scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx = tx % numof;
    if(scheme == 0)
    {
        outputRe += oOffset;
        outputIm += oOffset;

        T out        = lib_make_vector2<T>(outputRe[tx], outputIm[tx]);
        outputRe[tx] = input[tx].x * out.x - input[tx].y * out.y;
        outputIm[tx] = input[tx].x * out.y + input[tx].y * out.x;
    }
    else if(scheme == 1)
    {
        real_type_t<T>* chirpRe = outputRe;
        real_type_t<T>* chirpIm = outputIm;

        input += iOffset;

        outputRe += M;
        outputRe += oOffset;

        outputIm += M;
        outputIm += oOffset;

        if(tx < N)
        {
            outputRe[tx] = input[tx].x * chirpRe[tx] + input[tx].y * chirpIm[tx];
            outputIm[tx] = -input[tx].x * chirpIm[tx] + input[tx].y * chirpRe[tx];
        }
        else
        {
            outputRe[tx] = 0;
            outputIm[tx] = 0;
        }
    }
    else if(scheme == 2)
    {
        const T* chirp = input;

        input += 2 * M;
        input += iOffset;

        outputRe += oOffset;
        outputIm += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        outputRe[tx]      = MI * (input[tx].x * chirp[tx].x + input[tx].y * chirp[tx].y);
        outputIm[tx]      = MI * (-input[tx].x * chirp[tx].y + input[tx].y * chirp[tx].x);
    }
}

template <typename T>
__global__ void mul_device(const size_t  numof,
                           const size_t  totalWI,
                           const size_t  N,
                           const size_t  M,
                           const real_type_t<T>* inputRe,
                           const real_type_t<T>* inputIm,
                           real_type_t<T>* outputRe,
                           real_type_t<T>* outputIm,
                           const size_t  dim,
                           const size_t* lengths,
                           const size_t* stride_in,
                           const size_t* stride_out,
                           const int     dir,
                           const int     scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx = tx % numof;
    if(scheme == 0)
    {
        outputRe += oOffset;
        outputIm += oOffset;

        T out        = lib_make_vector2<T>(outputRe[tx], outputIm[tx]);
        outputRe[tx] = inputRe[tx] * out.x - inputIm[tx] * out.y;
        outputIm[tx] = inputRe[tx] * out.y + inputIm[tx] * out.x;
    }
    else if(scheme == 1)
    {
        real_type_t<T>* chirpRe = outputRe;
        real_type_t<T>* chirpIm = outputIm;

        inputRe += iOffset;
        inputIm += iOffset;

        outputRe += M;
        outputRe += oOffset;

        outputIm += M;
        outputIm += oOffset;

        if(tx < N)
        {
            outputRe[tx] = inputRe[tx] * chirpRe[tx] + inputIm[tx] * chirpIm[tx];
            outputIm[tx] = -inputRe[tx] * chirpIm[tx] + inputIm[tx] * chirpRe[tx];
        }
        else
        {
            outputRe[tx] = 0;
            outputIm[tx] = 0;
        }
    }
    else if(scheme == 2)
    {
        const real_type_t<T>* chirpRe = inputRe;
        const real_type_t<T>* chirpIm = inputIm;

        inputRe += 2 * M;
        inputRe += iOffset;

        inputIm += 2 * M;
        inputIm += iOffset;

        outputRe += oOffset;
        outputIm += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        outputRe[tx]      = MI * (inputRe[tx] * chirpRe[tx] + inputIm[tx] * chirpIm[tx]);
        outputIm[tx]      = MI * (-inputRe[tx] * chirpIm[tx] + inputIm[tx] * chirpRe[tx]);
    }
}

#endif // BLUESTEIN_H
