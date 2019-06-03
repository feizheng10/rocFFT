/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "real2complex.h"
#include "./kernels/common.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include <iostream>

template <typename T>
__global__ void real2complex_kernel(size_t          input_size,
                                    size_t          input_stride,
                                    size_t          output_stride,
                                    real_type_t<T>* input,
                                    size_t          input_distance,
                                    T*              output,
                                    size_t          output_distance)
{
    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch offset

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < input_size)
    {
        output[tid].y = 0.0;
        output[tid].x = input[tid];
    }
}

/*! \brief auxiliary function

    convert a real vector into a complex one by padding the imaginary part with
   0.

    @param[in]
    input_size
           size of input buffer

    @param[in]
    input_buffer
          data type : float or double

    @param[in]
    input_distance
          distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type : complex type (float2 or double2)

    @param[in]
    output_distance
           distance between consecutive batch members for output buffer

    @param[in]
    batch
           number of transforms

    @param[in]
    precision
          data type of input buffer. rocfft_precision_single or
   rocfft_precsion_double

    ********************************************************************/

void real2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;

    size_t blocks = (input_size - 1) / 512 + 1;

    if(high_dimension > 65535 || batch > 65535)
        printf("2D and 3D or batch is too big; not implemented\n");
    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    if(precision == rocfft_precision_single)
        hipLaunchKernelGGL(real2complex_kernel<float2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (float*)input_buffer,
                           input_distance,
                           (float2*)output_buffer,
                           output_distance);
    else
        hipLaunchKernelGGL(real2complex_kernel<double2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (double*)input_buffer,
                           input_distance,
                           (double2*)output_buffer,
                           output_distance);

    /*float2* tmp; tmp = (float2*)malloc(sizeof(float2)*output_distance*batch);
  hipMemcpy(tmp, output_buffer, sizeof(float2)*output_distance*batch,
  hipMemcpyDeviceToHost);

  for(size_t j=0;j<data->node->length[1]; j++)
  {
      for(size_t i=0; i<data->node->length[0]; i++)
      {
          printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
  tmp[j*data->node->outStride[1] + i].x, tmp[j*data->node->outStride[1] + i].y);
      }
  }*/

    return;
}

/*============================================================================================*/

template <typename T>
__global__ void complex2hermitian_kernel(size_t input_size,
                                         size_t input_stride,
                                         size_t output_stride,
                                         T*     input,
                                         size_t input_distance,
                                         T*     output,
                                         size_t output_distance)
{

    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < (1 + input_size / 2)) // only read and write the first
    // [input_size/2+1] elements due to conjugate
    // redundancy
    {
        output[tid] = input[tid];
    }
}

/*! \brief auxiliary function

    read from input_buffer and store the first  [1 + input_size/2] elements to
   the output_buffer

    @param[in]
    input_size
           size of input buffer

    @param[in]
    input_buffer
          data type dictated by precision parameter but complex type (float2 or
   double2)

    @param[in]
    input_distance
           distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type dictated by precision parameter but complex type (float2 or
   double2)
          but only store first [1 + input_size/2] elements according to
   conjugate symmetry

    @param[in]
    output_distance
           distance between consecutive batch members for output buffer

    @param[in]
    batch
           number of transforms

    @param[in]
    precision
           data type of input and output buffer. rocfft_precision_single or
   rocfft_precsion_double

    ********************************************************************/

void complex2hermitian(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;

    size_t blocks = (input_size - 1) / 512 + 1;

    if(high_dimension > 65535 || batch > 65535)
        printf("2D and 3D or batch is too big; not implemented\n");
    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    /*float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_distance*batch);
  hipMemcpy(tmp, input_buffer, sizeof(float2)*input_distance*batch,
  hipMemcpyDeviceToHost);

  for(size_t j=0;j<data->node->length[1]; j++)
  {
      for(size_t i=0; i<data->node->length[0]; i++)
      {
          printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
  tmp[j*data->node->outStride[1] + i].x, tmp[j*data->node->outStride[1] + i].y);
      }
  }*/

    if(precision == rocfft_precision_single)
        hipLaunchKernelGGL(complex2hermitian_kernel<float2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (float2*)input_buffer,
                           input_distance,
                           (float2*)output_buffer,
                           output_distance);
    else
        hipLaunchKernelGGL(complex2hermitian_kernel<double2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (double2*)input_buffer,
                           input_distance,
                           (double2*)output_buffer,
                           output_distance);

    return;
}

/*============================================================================================*/

// GPU kernel for 1d r2c post-process and c2r pre-process
// T is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
template <typename T, bool IN_PLACE, bool R2C>
__global__ void real_1d_pre_post_process_kernel(size_t   input_size,
                                                size_t   input_stride,
                                                size_t   output_stride,
                                                T*       input,
                                                size_t   input_distance,
                                                T*       output,
                                                size_t   output_distance,
                                                T const* twiddles)
{
    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch offset

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t idx_q = (input_size >> 1) - idx_p;

    T p = input[idx_p];
    T q = input[idx_q];

    if(IN_PLACE)
    {
        __syncthreads(); //it reqires only for in-place
    }

    if(idx_p == 0)
    {
        if(R2C)
        {
            output[idx_p].x = p.x + p.y;
            output[idx_p].y = 0;
            output[idx_q].x = p.x - p.y;
            output[idx_q].y = 0;
        }
        else
        {
            output[idx_p].x = p.x + q.x;
            output[idx_p].y = p.x - q.x;
        }
    }
    else if(idx_p <= input_size >> 2)
    {
        T u(p.x + q.x, p.y - q.y); // p + conj(q)
        T v(p.x - q.x, p.y + q.y); // p - conj(q)

        T twd_p = twiddles[idx_p];
        T twd_q = twiddles[idx_q];

        if(R2C)
        {
            u *= 0.5;
            v *= 0.5;
        }
        else
        {
            twd_p.x = -twd_p.x;
            twd_q.x = -twd_q.x;
        }

        output[idx_p].x = u.x + v.x * twd_p.y + v.y * twd_p.x;
        output[idx_p].y = u.y + v.y * twd_p.y - v.x * twd_p.x;

        output[idx_q].x = u.x - v.x * twd_q.y + v.y * twd_q.x;
        output[idx_q].y = -u.y + v.y * twd_q.y + v.x * twd_q.x;
    }
}

// GPU intermediate host code
template <typename T, bool R2C>
void real_1d_pre_post_process(size_t const N,
                              size_t       batch,
                              T*           d_input,
                              T*           d_output,
                              T*           d_twiddles,
                              size_t       high_dimension,
                              size_t       input_stride,
                              size_t       output_stride,
                              size_t       input_distance,
                              size_t       output_distance,
                              hipStream_t  rocfft_stream)
{
    const size_t block_size = 512;
    size_t       blocks     = (N / 4 - 1) / block_size + 1;

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    if(d_input == d_output)
    {
        hipLaunchKernelGGL(real_1d_pre_post_process_kernel<T, true, R2C>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           N,
                           input_stride,
                           output_stride,
                           d_input,
                           input_distance,
                           d_output,
                           output_distance,
                           d_twiddles);
    }
    else
    {
        hipLaunchKernelGGL(real_1d_pre_post_process_kernel<T, false, R2C>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           N,
                           input_stride,
                           output_stride,
                           d_input,
                           input_distance,
                           d_output,
                           output_distance,
                           d_twiddles);
    }
}

template <bool R2C>
void real_1d_pre_post(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    // input_size is the innermost dimension
    // the upper level provides always N/2, that is regular complex fft size
    size_t input_size = data->node->length[0] * 2;

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    //Fixme!!!
    //std::cout << "data->node->length.size() " << data->node->length.size() << std::endl;
    size_t input_stride = 1;
    //= (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride = 1;
    //= (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }

    // std::cout << "real_1d_pre_post debug,  input_size " << input_size
    //         << ", high_dimension " << high_dimension
    //         << ", input_stride " << input_stride
    //         << ", output_stride " << output_stride
    //         << ", input_distance " << input_distance
    //         << ", output_distance " << output_distance
    //         << ", input_buffer " << input_buffer
    //         << ", output_buffer " << output_buffer
    //         << std::endl;

    if(data->node->precision == rocfft_precision_single)
    {
        real_1d_pre_post_process<float2, R2C>(input_size,
                                              batch,
                                              (float2*)input_buffer,
                                              (float2*)output_buffer,
                                              (float2*)(data->node->twiddles),
                                              high_dimension,
                                              input_stride,
                                              output_stride,
                                              input_distance,
                                              output_distance,
                                              data->rocfft_stream);
    }
    else
    {
        real_1d_pre_post_process<double2, R2C>(input_size,
                                               batch,
                                               (double2*)input_buffer,
                                               (double2*)output_buffer,
                                               (double2*)(data->node->twiddles),
                                               high_dimension,
                                               input_stride,
                                               output_stride,
                                               input_distance,
                                               output_distance,
                                               data->rocfft_stream);
    }
}

void r2c_1d_post(const void* data_p, void* back_p)
{
    real_1d_pre_post<true>(data_p, back_p);
}

void c2r_1d_pre(const void* data_p, void* back_p)
{
    real_1d_pre_post<false>(data_p, back_p);
}

void real2complex_pre_process(const void* data, void* back)
{
    DeviceCallIn* data_p = (DeviceCallIn*)data;
    if(data_p->node->IsRCsimple())
    {
        real2complex(data, back);
    }
    else
    {
        // nope
    }
}

void real2complex_post_process(const void* data, void* back)
{
    DeviceCallIn* data_p = (DeviceCallIn*)data;
    if(data_p->node->IsRCsimple())
    {
        complex2hermitian(data, back);
    }
    else
    {
        r2c_1d_post(data, back);
    }
}

void complex2real_pre_process(const void* data, void* back)
{
    DeviceCallIn* data_p = (DeviceCallIn*)data;
    if(data_p->node->IsRCsimple())
    {
        hermitian2complex(data, back);
    }
    else
    {
        c2r_1d_pre(data, back);
    }
}

void complex2real_post_process(const void* data, void* back)
{
    DeviceCallIn* data_p = (DeviceCallIn*)data;
    if(data_p->node->IsRCsimple())
    {
        complex2real(data, back);
    }
    else
    {
        // nope
    }
}
