// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "plan.h"
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "private.h"
#include "radix_table.h"
#include "repo.h"
#include "rocfft-version.h"
#include "rocfft.h"
#include "rocfft_ostream.hpp"

#include <algorithm>
#include <assert.h>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)

// clang-format off
#define ROCFFT_VERSION_STRING (TO_STR(rocfft_version_major) "." \
                               TO_STR(rocfft_version_minor) "." \
                               TO_STR(rocfft_version_patch) "." \
                               TO_STR(rocfft_version_tweak) )
// clang-format on

std::string PrintScheme(ComputeScheme cs)
{
    const std::map<ComputeScheme, const char*> ComputeSchemetoString
        = {{ENUMSTR(CS_NONE)},
           {ENUMSTR(CS_KERNEL_STOCKHAM)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_RC)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_CR)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_USING_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_HERM)},
           {ENUMSTR(CS_KERNEL_COPY_HERM_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_R)},

           {ENUMSTR(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_EVEN)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_CMPLX_TO_R)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_CMPLX_TO_R)},
           {ENUMSTR(CS_REAL_2D_EVEN)},
           {ENUMSTR(CS_REAL_3D_EVEN)},
           {ENUMSTR(CS_KERNEL_APPLY_CALLBACK)},

           {ENUMSTR(CS_BLUESTEIN)},
           {ENUMSTR(CS_KERNEL_CHIRP)},
           {ENUMSTR(CS_KERNEL_PAD_MUL)},
           {ENUMSTR(CS_KERNEL_FFT_MUL)},
           {ENUMSTR(CS_KERNEL_RES_MUL)},

           {ENUMSTR(CS_L1D_TRTRT)},
           {ENUMSTR(CS_L1D_CC)},
           {ENUMSTR(CS_L1D_CRT)},

           {ENUMSTR(CS_2D_STRAIGHT)},
           {ENUMSTR(CS_2D_RTRT)},
           {ENUMSTR(CS_2D_RC)},
           {ENUMSTR(CS_KERNEL_2D_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_2D_SINGLE)},

           {ENUMSTR(CS_3D_STRAIGHT)},
           {ENUMSTR(CS_3D_TRTRTR)},
           {ENUMSTR(CS_3D_RTRT)},
           {ENUMSTR(CS_3D_BLOCK_RC)},
           {ENUMSTR(CS_3D_RC)},
           {ENUMSTR(CS_KERNEL_3D_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_3D_SINGLE)}};

    return ComputeSchemetoString.at(cs);
}

std::string PrintOperatingBuffer(const OperatingBuffer ob)
{
    const std::map<OperatingBuffer, const char*> BuffertoString
        = {{ENUMSTR(OB_UNINIT)},
           {ENUMSTR(OB_USER_IN)},
           {ENUMSTR(OB_USER_OUT)},
           {ENUMSTR(OB_TEMP)},
           {ENUMSTR(OB_TEMP_CMPLX_FOR_REAL)},
           {ENUMSTR(OB_TEMP_BLUESTEIN)}};
    return BuffertoString.at(ob);
}

std::string PrintOperatingBufferCode(const OperatingBuffer ob)
{
    const std::map<OperatingBuffer, const char*> BuffertoString = {{OB_UNINIT, "ERR"},
                                                                   {OB_USER_IN, "A"},
                                                                   {OB_USER_OUT, "B"},
                                                                   {OB_TEMP, "T"},
                                                                   {OB_TEMP_CMPLX_FOR_REAL, "C"},
                                                                   {OB_TEMP_BLUESTEIN, "S"}};
    return BuffertoString.at(ob);
}

rocfft_status rocfft_plan_description_set_scale_float(rocfft_plan_description description,
                                                      const float             scale)
{
    description->scale = scale;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_scale_double(rocfft_plan_description description,
                                                       const double            scale)
{
    description->scale = scale;
    return rocfft_status_success;
}

static size_t offset_count(rocfft_array_type type)
{
    // planar data has 2 sets of offsets, otherwise we have one
    return type == rocfft_array_type_complex_planar || type == rocfft_array_type_hermitian_planar
               ? 2
               : 1;
}

rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      const rocfft_array_type in_array_type,
                                                      const rocfft_array_type out_array_type,
                                                      const size_t*           in_offsets,
                                                      const size_t*           out_offsets,
                                                      const size_t            in_strides_size,
                                                      const size_t*           in_strides,
                                                      const size_t            in_distance,
                                                      const size_t            out_strides_size,
                                                      const size_t*           out_strides,
                                                      const size_t            out_distance)
{
    log_trace(__func__,
              "description",
              description,
              "in_array_type",
              in_array_type,
              "out_array_type",
              out_array_type,
              "in_offsets",
              std::make_pair(in_offsets, offset_count(in_array_type)),
              "out_offsets",
              std::make_pair(out_offsets, offset_count(out_array_type)),
              "in_strides",
              std::make_pair(in_strides, in_strides_size),
              "in_distance",
              in_distance,
              "out_strides",
              std::make_pair(out_strides, out_strides_size),
              "out_distance",
              out_distance);

    description->inArrayType  = in_array_type;
    description->outArrayType = out_array_type;

    if(in_offsets != nullptr)
    {
        description->inOffset[0] = in_offsets[0];
        if((in_array_type == rocfft_array_type_complex_planar)
           || (in_array_type == rocfft_array_type_hermitian_planar))
            description->inOffset[1] = in_offsets[1];
    }

    if(out_offsets != nullptr)
    {
        description->outOffset[0] = out_offsets[0];
        if((out_array_type == rocfft_array_type_complex_planar)
           || (out_array_type == rocfft_array_type_hermitian_planar))
            description->outOffset[1] = out_offsets[1];
    }

    if(in_strides != nullptr)
    {
        for(size_t i = 0; i < std::min((size_t)3, in_strides_size); i++)
            description->inStrides[i] = in_strides[i];
    }

    if(in_distance != 0)
        description->inDist = in_distance;

    if(out_strides != nullptr)
    {
        for(size_t i = 0; i < std::min((size_t)3, out_strides_size); i++)
            description->outStrides[i] = out_strides[i];
    }

    if(out_distance != 0)
        description->outDist = out_distance;

    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_create(rocfft_plan_description* description)
{
    rocfft_plan_description desc = new rocfft_plan_description_t;
    *description                 = desc;
    log_trace(__func__, "description", *description);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description)
{
    log_trace(__func__, "description", description);
    if(description != nullptr)
        delete description;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_create_internal(rocfft_plan                   plan,
                                          const rocfft_result_placement placement,
                                          const rocfft_transform_type   transform_type,
                                          const rocfft_precision        precision,
                                          const size_t                  dimensions,
                                          const size_t*                 lengths,
                                          const size_t                  number_of_transforms,
                                          const rocfft_plan_description description)
{
    // Check plan validity
    if(description != nullptr)
    {
        switch(transform_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            // We need complex input data
            if(!((description->inArrayType == rocfft_array_type_complex_interleaved)
                 || (description->inArrayType == rocfft_array_type_complex_planar)))
                return rocfft_status_invalid_array_type;
            // We need complex output data
            if(!((description->outArrayType == rocfft_array_type_complex_interleaved)
                 || (description->outArrayType == rocfft_array_type_complex_planar)))
                return rocfft_status_invalid_array_type;
            // In-place transform requires that the input and output
            // format be identical
            if(placement == rocfft_placement_inplace)
            {
                if(description->inArrayType != description->outArrayType)
                    return rocfft_status_invalid_array_type;
            }
            break;
        case rocfft_transform_type_real_forward:
            // Input must be real
            if(description->inArrayType != rocfft_array_type_real)
                return rocfft_status_invalid_array_type;
            // Output must be Hermitian
            if(!((description->outArrayType == rocfft_array_type_hermitian_interleaved)
                 || (description->outArrayType == rocfft_array_type_hermitian_planar)))
                return rocfft_status_invalid_array_type;
            // In-place transform must output to interleaved format
            if((placement == rocfft_placement_inplace)
               && (description->outArrayType != rocfft_array_type_hermitian_interleaved))
                return rocfft_status_invalid_array_type;
            break;
        case rocfft_transform_type_real_inverse:
            // Output must be real
            if(description->outArrayType != rocfft_array_type_real)
                return rocfft_status_invalid_array_type;
            // Intput must be Hermitian
            if(!((description->inArrayType == rocfft_array_type_hermitian_interleaved)
                 || (description->inArrayType == rocfft_array_type_hermitian_planar)))
                return rocfft_status_invalid_array_type;
            // In-place transform must have interleaved input
            if((placement == rocfft_placement_inplace)
               && (description->inArrayType != rocfft_array_type_hermitian_interleaved))
                return rocfft_status_invalid_array_type;
            break;
        }
    }

    if(dimensions > 3)
        return rocfft_status_invalid_dimensions;

    rocfft_plan p = plan;
    p->rank       = dimensions;
    p->lengths[0] = 1;
    p->lengths[1] = 1;
    p->lengths[2] = 1;
    for(size_t ilength = 0; ilength < dimensions; ++ilength)
    {
        p->lengths[ilength] = lengths[ilength];
    }
    p->batch          = number_of_transforms;
    p->placement      = placement;
    p->precision      = precision;
    p->base_type_size = (precision == rocfft_precision_double) ? sizeof(double) : sizeof(float);
    p->transformType  = transform_type;

    if(description != nullptr)
    {
        p->desc = *description;
    }
    else
    {
        switch(transform_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            p->desc.inArrayType  = rocfft_array_type_complex_interleaved;
            p->desc.outArrayType = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            p->desc.inArrayType  = rocfft_array_type_real;
            p->desc.outArrayType = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            p->desc.inArrayType  = rocfft_array_type_hermitian_interleaved;
            p->desc.outArrayType = rocfft_array_type_real;
            break;
        }
    }

    // Set inStrides, if not specified
    if(p->desc.inStrides[0] == 0)
    {
        p->desc.inStrides[0] = 1;

        if((p->transformType == rocfft_transform_type_real_forward)
           && (p->placement == rocfft_placement_inplace))
        {
            // real-to-complex in-place
            size_t dist = 2 * (1 + (p->lengths[0]) / 2);

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.inStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.inDist == 0)
                p->desc.inDist = dist;
        }
        else if(p->transformType == rocfft_transform_type_real_inverse)
        {
            // complex-to-real
            size_t dist = 1 + (p->lengths[0]) / 2;

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.inStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.inDist == 0)
                p->desc.inDist = dist;
        }

        else
        {
            // Set the inStrides to deal with contiguous data
            for(size_t i = 1; i < (p->rank); i++)
                p->desc.inStrides[i] = p->lengths[i - 1] * p->desc.inStrides[i - 1];
        }
    }

    // Set outStrides, if not specified
    if(p->desc.outStrides[0] == 0)
    {
        p->desc.outStrides[0] = 1;

        if((p->transformType == rocfft_transform_type_real_inverse)
           && (p->placement == rocfft_placement_inplace))
        {
            // complex-to-real in-place
            size_t dist = 2 * (1 + (p->lengths[0]) / 2);

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.outStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.outDist == 0)
                p->desc.outDist = dist;
        }
        else if(p->transformType == rocfft_transform_type_real_forward)
        {
            // real-co-complex
            size_t dist = 1 + (p->lengths[0]) / 2;

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.outStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.outDist == 0)
                p->desc.outDist = dist;
        }
        else
        {
            // Set the outStrides to deal with contiguous data
            for(size_t i = 1; i < (p->rank); i++)
                p->desc.outStrides[i] = p->lengths[i - 1] * p->desc.outStrides[i - 1];
        }
    }

    // Set in and out Distances, if not specified
    if(p->desc.inDist == 0)
    {
        p->desc.inDist = p->lengths[p->rank - 1] * p->desc.inStrides[p->rank - 1];
    }
    if(p->desc.outDist == 0)
    {
        p->desc.outDist = p->lengths[p->rank - 1] * p->desc.outStrides[p->rank - 1];
    }

    // size_t prodLength = 1;
    // for(size_t i = 0; i < (p->rank); i++)
    // {
    //     prodLength *= lengths[i];
    // }
    // if(!SupportedLength(prodLength))
    // {
    //     printf("This size %zu is not supported in rocFFT, will return;\n",
    //            prodLength);
    //     return rocfft_status_invalid_dimensions;
    // }

    // add this plan into repo, incurs computation, see repo.cpp
    return Repo::GetRepo().CreatePlan(p);
}

rocfft_status rocfft_plan_allocate(rocfft_plan* plan)
{
    *plan = new rocfft_plan_t;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_create(rocfft_plan*                  plan,
                                 const rocfft_result_placement placement,
                                 const rocfft_transform_type   transform_type,
                                 const rocfft_precision        precision,
                                 const size_t                  dimensions,
                                 const size_t*                 lengths,
                                 const size_t                  number_of_transforms,
                                 const rocfft_plan_description description)
{
    rocfft_plan_allocate(plan);

    size_t log_len[3] = {1, 1, 1};
    if(dimensions > 0)
        log_len[0] = lengths[0];
    if(dimensions > 1)
        log_len[1] = lengths[1];
    if(dimensions > 2)
        log_len[2] = lengths[2];

    log_trace(__func__,
              "plan",
              *plan,
              "placement",
              placement,
              "transform_type",
              transform_type,
              "precision",
              precision,
              "dimensions",
              dimensions,
              "lengths",
              std::make_pair(lengths, dimensions),
              "number_of_transforms",
              number_of_transforms,
              "description",
              description);

    std::stringstream ss;
    ss << "./rocfft-rider"
       << " -t " << transform_type << " -x " << log_len[0] << " -y " << log_len[1] << " -z "
       << log_len[2] << " -b " << number_of_transforms;
    if(placement == rocfft_placement_notinplace)
        ss << " -o ";
    if(precision == rocfft_precision_double)
        ss << " --double ";
    if(description != NULL)
        ss << " --isX " << description->inStrides[0] << " --isY " << description->inStrides[1]
           << " --isZ " << description->inStrides[2] << " --osX " << description->outStrides[0]
           << " --osY " << description->outStrides[1] << " --osZ " << description->outStrides[2]
           << " --scale " << description->scale << " --iOff0 " << description->inOffset[0]
           << " --iOff1 " << description->inOffset[1] << " --oOff0 " << description->outOffset[0]
           << " --oOff1 " << description->outOffset[1] << " --inArrType "
           << description->inArrayType << " --outArrType " << description->outArrayType;

    log_bench(ss.str());

    return rocfft_plan_create_internal(*plan,
                                       placement,
                                       transform_type,
                                       precision,
                                       dimensions,
                                       lengths,
                                       number_of_transforms,
                                       description);
}

rocfft_status rocfft_plan_destroy(rocfft_plan plan)
{
    log_trace(__func__, "plan", plan);
    // Remove itself from Repo first, and then delete itself
    Repo& repo = Repo::GetRepo();
    repo.DeletePlan(plan);
    if(plan != nullptr)
    {
        delete plan;
        plan = nullptr;
    }
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t* size_in_bytes)
{
    Repo&     repo     = Repo::GetRepo();
    ExecPlan* execPlan = repo.GetPlan(plan);
    if(!execPlan)
        return rocfft_status_failure;

    *size_in_bytes = execPlan->WorkBufBytes(plan->base_type_size);
    log_trace(__func__, "plan", plan, "size_in_bytes ptr", size_in_bytes, "val", *size_in_bytes);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_print(const rocfft_plan plan)
{
    log_trace(__func__, "plan", plan);
    rocfft_cout << std::endl;
    rocfft_cout << "precision: "
                << ((plan->precision == rocfft_precision_single) ? "single" : "double")
                << std::endl;

    rocfft_cout << "transform type: ";
    switch(plan->transformType)
    {
    case rocfft_transform_type_complex_forward:
        rocfft_cout << "complex forward";
        break;
    case rocfft_transform_type_complex_inverse:
        rocfft_cout << "complex inverse";
        break;
    case rocfft_transform_type_real_forward:
        rocfft_cout << "real forward";
        break;
    case rocfft_transform_type_real_inverse:
        rocfft_cout << "real inverse";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "result placement: ";
    switch(plan->placement)
    {
    case rocfft_placement_inplace:
        rocfft_cout << "in-place";
        break;
    case rocfft_placement_notinplace:
        rocfft_cout << "not in-place";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input array type: ";
    switch(plan->desc.inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "complex planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "output array type: ";
    switch(plan->desc.outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "comple planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "dimensions: " << plan->rank << std::endl;

    rocfft_cout << "lengths: " << plan->lengths[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->lengths[i];
    rocfft_cout << std::endl;
    rocfft_cout << "batch size: " << plan->batch << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input offset: " << plan->desc.inOffset[0];
    if((plan->desc.inArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.inArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.inOffset[1];
    rocfft_cout << std::endl;

    rocfft_cout << "output offset: " << plan->desc.outOffset[0];
    if((plan->desc.outArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.outArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.outOffset[1];
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input strides: " << plan->desc.inStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.inStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "output strides: " << plan->desc.outStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.outStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "input distance: " << plan->desc.inDist << std::endl;
    rocfft_cout << "output distance: " << plan->desc.outDist << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "scale: " << plan->desc.scale << std::endl;
    rocfft_cout << std::endl;

    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, const size_t len)
{
    log_trace(__func__, "buf", buf, "len", len);
    static constexpr char v[] = ROCFFT_VERSION_STRING;
    if(!buf)
        return rocfft_status_failure;
    if(len < sizeof(v))
        return rocfft_status_invalid_arg_value;
    memcpy(buf, v, sizeof(v));
    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_repo_get_unique_plan_count(size_t* count)
{
    Repo& repo = Repo::GetRepo();
    *count     = repo.GetUniquePlanCount();
    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_repo_get_total_plan_count(size_t* count)
{
    Repo& repo = Repo::GetRepo();
    *count     = repo.GetTotalPlanCount();
    return rocfft_status_success;
}

//
// Factorisation helpers
//

inline size_t get_explicity_supported_factor(rocfft_precision precision, size_t length0)
{
    auto supported  = function_pool::get_lengths(precision, CS_KERNEL_STOCKHAM);
    auto comparison = std::greater<size_t>();
    std::sort(supported.begin(), supported.end(), comparison);

    if(supported.empty())
        return 0;

    size_t factor = 0;

    if(length0 > (Large1DThreshold(precision) * Large1DThreshold(precision)))
    {
        auto supported_factor = [length0](size_t factor) -> bool {
            bool is_factor = length0 % factor == 0;
            return is_factor;
        };

        auto v     = Large1DThreshold(precision);
        auto lower = std::lower_bound(supported.cbegin(), supported.cend(), v, comparison);
        auto itr   = std::find_if(lower, supported.cend(), supported_factor);
        if(itr != supported.cend())
            factor = *itr;
    }
    else
    {
        // break into as squarish matrix as possible
        auto supported_factor = [length0, precision = precision](size_t factor) -> bool {
            bool is_factor    = length0 % factor == 0;
            bool have_kernels = function_pool::has_function(fpkey(length0 / factor, precision));
            return is_factor && have_kernels;
        };

        auto v     = (size_t)sqrt(length0);
        auto lower = std::lower_bound(supported.cbegin(), supported.cend(), v, comparison);
        if(*lower < sqrt(length0))
            lower--;

        // note: this squarish factorization isn't always the fastest.
        // for length 18816: if you start at supported.cbegin() the resulting plan is faster
        auto itr = std::find_if(lower, supported.cend(), supported_factor);
        if(itr != supported.cend())
            factor = *itr;
    }

    return factor;
}

inline bool SupportedLength(rocfft_precision precision, size_t len)
{
    // do we have an explicit kernel?
    if(function_pool::has_function(fpkey(len, precision)))
        return true;

    // can we factor with 2, 3, or 5?  note: all combinations of these
    // are explicitly generated at build time
    size_t p = len;
    while(!(p % 2))
        p /= 2;
    while(!(p % 3))
        p /= 3;
    while(!(p % 5))
        p /= 5;

    if(p == 1)
        return true;

    // do we have an explicit kernel for the remainder?
    if(function_pool::has_function(fpkey(p, precision)))
        return true;

    // finally, can we factor this length with combinations of existing kernels?
    if(get_explicity_supported_factor(precision, len) > 0)
        return true;

    return false;
}

inline size_t FindBlue(size_t len)
{
    size_t p = 1;
    while(p < len)
        p <<= 1;
    return 2 * p;
}

inline void PrintFailInfo(rocfft_precision precision,
                          size_t           length,
                          ComputeScheme    scheme,
                          size_t           kernelLength = 0,
                          ComputeScheme    kernelScheme = CS_NONE)
{
    rocfft_cerr << "Failed on Node: length " << length << " (" << precision << "): "
                << "when attempting Scheme: " << PrintScheme(scheme) << std::endl;
    if(kernelScheme != CS_NONE)
        rocfft_cerr << "\tCouldn't find the kernel of length " << kernelLength << ", with type "
                    << PrintScheme(kernelScheme) << std::endl;
}

// Tree node builders

// NB:
// Don't assign inArrayType and outArrayType when building any tree node.
// That should be done in buffer assignment stage or
// TraverseTreeAssignPlacementsLogicA().

void TreeNode::RecursiveBuildTree()
{
    // this flag can be enabled when generator can do block column fft in
    // multi-dimension cases and small 2d, 3d within one kernel
    bool MultiDimFuseKernelsAvailable = false;

    if((parent == nullptr)
       && ((inArrayType == rocfft_array_type_real) || (outArrayType == rocfft_array_type_real)))
    {
        build_real();
        return;
    }

    switch(dimension)
    {
    case 1:
        build_1D();
        break;

    case 2:
    {
        if(scheme == CS_KERNEL_TRANSPOSE)
            return;

        // First choice is 2D_SINGLE kernel, if the problem will fit into LDS.
        // Next best is CS_2D_RC. Last resort is RTRT.
        if(use_CS_2D_SINGLE())
        {
            scheme = CS_KERNEL_2D_SINGLE; // the node has all build info
            return;
        }
        else if(use_CS_2D_RC())
        {
            scheme = CS_2D_RC;
            build_CS_2D_RC();
            return;
        }
        else
        {
            scheme = CS_2D_RTRT;
            build_CS_2D_RTRT();
            return;
        }
    }
    break;

    case 3:
    {
        if(use_CS_3D_RC())
        {
            scheme = CS_3D_RC;
        }
        else if(MultiDimFuseKernelsAvailable)
        {
            // conditions to choose which scheme
            if((length[0] * length[1] * length[2]) <= 2048)
                scheme = CS_KERNEL_3D_SINGLE;
            else if(length[2] <= 256)
                scheme = CS_3D_RC;
            else
                scheme = CS_3D_RTRT;
        }
        else
        {
            // if we can get down to 3 or 4 kernels via SBRC, prefer that
            if(use_CS_3D_BLOCK_RC())
                scheme = CS_3D_BLOCK_RC;
            else
                scheme = CS_3D_RTRT;

            if(scheme == CS_3D_RTRT)
            {
                // NB:
                // Peek the 1st child but not really add it in.
                // Give up if 1st child is 2D_RTRT (means the poor RTRT_TRT)
                // Switch to TRTRTR as the last resort.
                auto child0       = TreeNode::CreateNode(this);
                child0->length    = length;
                child0->dimension = 2;
                child0->RecursiveBuildTree();
                if(child0->scheme == CS_2D_RTRT)
                {
                    scheme = CS_3D_TRTRTR;
                }
                // if we are here, the 2D sheme is either
                // 2D_SINGLE+TRT or 2D_RC+TRT
            }
        }

        switch(scheme)
        {
        case CS_3D_RTRT:
        {
            build_CS_3D_RTRT();
        }
        break;
        case CS_3D_TRTRTR:
        {
            build_CS_3D_TRTRTR();
        }
        break;
        case CS_3D_BLOCK_RC:
        {
            build_CS_3D_BLOCK_RC();
        }
        break;
        case CS_3D_RC:
        {
            build_CS_3D_RC();
        }
        break;
        case CS_KERNEL_3D_SINGLE:
        {
        }
        break;

        default:
            assert(false);
        }
    }
    break;

    default:
        assert(false);
    }
}

// check if we have an SBCC kernel along the specified dimension
static bool SBCC_dim_available(const std::vector<size_t>& length,
                               size_t                     sbcc_dim,
                               rocfft_precision           precision)
{
    // Check the C part.
    // The first R is built recursively with 2D_FFT, leave the check part to themselves
    size_t numTrans = 0;
    // do we have a purpose-built sbcc kernel
    bool have_sbcc = false;
    try
    {
        numTrans = function_pool::get_kernel(
                       fpkey(length[sbcc_dim], precision, CS_KERNEL_STOCKHAM_BLOCK_CC))
                       .batches_per_block;
        have_sbcc = true;
    }
    catch(std::out_of_range&)
    {
        try
        {
            numTrans
                = function_pool::get_kernel(fpkey(length[sbcc_dim], precision)).batches_per_block;
        }
        catch(std::out_of_range&)
        {
            return false;
        }
    }
    // if we have a size for this transform but numTrans, this must
    // be an old-generator kernel
    if(!numTrans)
    {
        size_t wgs = 0;
        DetermineSizes(length[sbcc_dim], wgs, numTrans);
    }

    // x-dim should be >= the blockwidth, or it might perform worse..
    if(length[0] < numTrans)
        return false;

    // for regular stockham kernels, ensure we are doing enough rows
    // to coalesce properly. 4 seems to be enough for
    // double-precision, whereas some sizes that do 7 rows seem to be
    // slower for single.
    if(!have_sbcc)
    {
        size_t minRows = precision == rocfft_precision_single ? 8 : 4;
        if(numTrans < minRows)
            return false;
    }

    return true;
}

// do we have an SBCC kernel for this specific size?
static bool have_SBCC_kernel(size_t length, rocfft_precision precision)
{
    return function_pool::has_function(fpkey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
}

bool TreeNode::use_CS_2D_SINGLE()
{
    // Get actual LDS size, to check if we can run a 2D_SINGLE
    // kernel that will fit the problem into LDS.
    //
    // NOTE: This is potentially problematic in a heterogeneous
    // multi-device environment.  The device we query now could
    // differ from the device we run the plan on.  That said,
    // it's vastly more common to have multiples of the same
    // device in the real world.
    int ldsSize;
    int deviceid;
    // if this fails, device 0 is a reasonable default
    if(hipGetDevice(&deviceid) != hipSuccess)
    {
        log_trace(__func__, "warning", "hipGetDevice failed - using device 0");
        deviceid = 0;
    }
    // if this fails, giving 0 to Single2DSizes will assume
    // normal size for contemporary hardware
    if(hipDeviceGetAttribute(&ldsSize, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, deviceid)
       != hipSuccess)
    {
        log_trace(__func__,
                  "warning",
                  "hipDeviceGetAttribute failed - assuming normal LDS size for current hardware");
        ldsSize = 0;
    }
    const auto single2DSizes = Single2DSizes(ldsSize, precision, GetWGSAndNT);
    if(std::find(single2DSizes.begin(), single2DSizes.end(), std::make_pair(length[0], length[1]))
       != single2DSizes.end())
        return true;

    return false;
}

bool TreeNode::use_CS_2D_RC()
{
    //   For CS_2D_RC, we are reusing SBCC kernel for 1D middle size. The
    //   current implementation of 1D SBCC supports only 64, 128, and 256.
    //   However, technically no LDS limitation along the fast dimension
    //   on upper bound for 2D SBCC cases, and even should not limit to pow
    //   of 2.

    std::set<int> sbcc_support = {50, 64, 81, 100, 128, 200, 256, 336};
    if((sbcc_support.find(length[1]) != sbcc_support.end()) && (length[0] >= 56))
    {
        return true;
    }

    return false;
}

size_t TreeNode::count_3D_SBRC_nodes()
{
    size_t sbrc_dimensions = 0;
    for(unsigned int i = 0; i < length.size(); ++i)
    {
        if(function_pool::has_function(fpkey(length[i], precision, CS_KERNEL_STOCKHAM_BLOCK_RC)))
        {
            // make sure the SBRC kernel on that dimension would be tile-aligned
            size_t bwd, wgs, lds;
            GetBlockComputeTable(length[i], bwd, wgs, lds);
            if(length[(i + 2) % length.size()] % bwd == 0)
                ++sbrc_dimensions;
        }
    }
    return sbrc_dimensions;
}

bool TreeNode::use_CS_3D_BLOCK_RC()
{
    // TODO: SBRC hasn't worked for inner batch (i/oDist == 1)
    if(iDist == 1 || oDist == 1)
        return false;

    return count_3D_SBRC_nodes() >= 2;
}

bool TreeNode::use_CS_3D_RC()
{
    // TODO: SBCC hasn't worked for inner batch (i/oDist == 1)
    if(iDist == 1 || oDist == 1)
        return false;

    try
    {
        // Check the C part.
        // The first R is built recursively with 2D_FFT, leave the check part to themselves
        auto krn
            = function_pool::get_kernel(fpkey(length[2], precision, CS_KERNEL_STOCKHAM_BLOCK_CC));

        // hack for this special case
        // this size is rejected by the following conservative threshold (#-elems)
        // however it can use 3D_RC and get much better performance
        std::vector<size_t> special_case{56, 336, 336};
        if(length == special_case && precision == rocfft_precision_double)
            return true;

        // x-dim should be >= the blockwidth, or it might perform worse..
        if(length[0] < krn.batches_per_block)
            return false;
        // we don't want a too-large 3D block, sbcc along z-dim might be bad
        if((length[0] * length[1] * length[2]) >= (128 * 128 * 128))
            return false;

        // Peek the first child
        // Give up if 1st child is 2D_RTRT (means the poor RTRT_C),
        auto child0       = TreeNode::CreateNode(this);
        child0->length    = length;
        child0->dimension = 2;
        child0->RecursiveBuildTree();
        if(child0->scheme == CS_2D_RTRT)
            return false;

        // if we are here, the 2D sheme is either
        // 2D_SINGLE+CC (2 kernels) or 2D_RC+CC (3 kernels),
        assert(child0->scheme == CS_KERNEL_2D_SINGLE || child0->scheme == CS_2D_RC);
        return true;
    }
    catch(...)
    {
        return false;
    }

    return false;
}

bool TreeNode::use_CS_KERNEL_TRANSPOSE_Z_XY()
{
    if(function_pool::has_function(fpkey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC)))
    {
        size_t bwd, wgs, lds;
        GetBlockComputeTable(length[0], bwd, wgs, lds);

        if((length[1] >= bwd) && (length[2] >= bwd) && (length[1] * length[2] % bwd == 0))
        {
            return true;
        }
    }

    return false;
}

void TreeNode::build_real()
{
    if(length[0] % 2 == 0 && inStride[0] == 1 && outStride[0] == 1)
    {
        switch(dimension)
        {
        case 1:
            build_real_even_1D();
            return;
        case 2:
            build_real_even_2D();
            return;
        case 3:
            build_real_even_3D();
            return;
        default:
            throw std::runtime_error("Invalid dimension in build_real()");
            return;
        }
    }

    // Fallback method
    build_real_embed();
}

void TreeNode::build_real_embed()
{
    // Embed the data into a full-length complex array, perform a
    // complex transform, and then extract the relevant output.

    scheme = CS_REAL_TRANSFORM_USING_CMPLX;

    auto copyHeadPlan = TreeNode::CreateNode(this);

    // head copy plan
    copyHeadPlan->dimension = dimension;
    copyHeadPlan->length    = length;
    copyHeadPlan->scheme    = (inArrayType == rocfft_array_type_real) ? CS_KERNEL_COPY_R_TO_CMPLX
                                                                      : CS_KERNEL_COPY_HERM_TO_CMPLX;
    childNodes.emplace_back(std::move(copyHeadPlan));

    // complex fft
    auto fftPlan = TreeNode::CreateNode(this);

    fftPlan->dimension = dimension;
    fftPlan->length    = length;

    fftPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(fftPlan));

    // tail copy plan
    auto copyTailPlan = TreeNode::CreateNode(this);

    copyTailPlan->dimension = dimension;
    copyTailPlan->length    = length;
    copyTailPlan->scheme    = (inArrayType == rocfft_array_type_real) ? CS_KERNEL_COPY_CMPLX_TO_HERM
                                                                      : CS_KERNEL_COPY_CMPLX_TO_R;
    childNodes.emplace_back(std::move(copyTailPlan));
}

void TreeNode::build_real_even_1D(bool fuse_pre_post_processing)
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    scheme = CS_REAL_TRANSFORM_EVEN;

    auto cfftPlan       = TreeNode::CreateNode(this);
    cfftPlan->dimension = dimension;
    cfftPlan->length    = length;
    cfftPlan->length[0] = cfftPlan->length[0] / 2;

    cfftPlan->placement = rocfft_placement_inplace;

    // cfftPlan works in-place on the input buffer for R2C, on the
    // output buffer for C2R

    // NB: the buffer is real, but we treat it as complex
    cfftPlan->RecursiveBuildTree();

    // fuse pre/post-processing into fft if it's single-kernel
    // and generated with new generator

    // NOTE: get_kernel won't throw because we would only have a
    // single kernel if it's in the function map.  We're also relying
    // on the fact that old-generator would not populate factors.
    bool singleKernelFFT
        = cfftPlan->childNodes.empty()
          && !function_pool::get_kernel(fpkey(cfftPlan->length.front(), precision)).factors.empty();
    if(!singleKernelFFT)
        fuse_pre_post_processing = false;

    switch(direction)
    {
    case -1:
    {
        // real-to-complex transform: in-place complex transform then post-process

        // insert a node that's prepared to apply the user's
        // callback, since the callback would expect reals and this
        // plan would otherwise pretend it's complex
        auto applyCallback       = TreeNode::CreateNode(this);
        applyCallback->scheme    = CS_KERNEL_APPLY_CALLBACK;
        applyCallback->dimension = dimension;
        applyCallback->length    = length;
        applyCallback->placement = rocfft_placement_inplace;
        childNodes.emplace_back(std::move(applyCallback));

        if(fuse_pre_post_processing)
            cfftPlan->ebtype = EmbeddedType::Real2C_POST;

        childNodes.emplace_back(std::move(cfftPlan));

        // add separate post-processing if we couldn't fuse
        if(!fuse_pre_post_processing)
        {
            auto postPlan       = TreeNode::CreateNode(this);
            postPlan->scheme    = CS_KERNEL_R_TO_CMPLX;
            postPlan->dimension = 1;
            postPlan->length    = length;
            postPlan->length[0] /= 2;

            childNodes.emplace_back(std::move(postPlan));
        }
        break;
    }
    case 1:
    {
        // complex-to-real transform: pre-process followed by in-place complex transform

        if(fuse_pre_post_processing)
            cfftPlan->ebtype = EmbeddedType::C2Real_PRE;
        else
        {
            // add separate pre-processing if we couldn't fuse
            auto prePlan       = TreeNode::CreateNode(this);
            prePlan->scheme    = CS_KERNEL_CMPLX_TO_R;
            prePlan->dimension = 1;
            prePlan->length    = length;
            prePlan->length[0] /= 2;

            childNodes.emplace_back(std::move(prePlan));
        }

        childNodes.emplace_back(std::move(cfftPlan));

        // insert a node that's prepared to apply the user's
        // callback, since the callback would expect reals and this
        // plan would otherwise pretend it's complex
        auto applyCallback       = TreeNode::CreateNode(this);
        applyCallback->scheme    = CS_KERNEL_APPLY_CALLBACK;
        applyCallback->dimension = dimension;
        applyCallback->length    = length;
        applyCallback->placement = rocfft_placement_inplace;
        childNodes.emplace_back(std::move(applyCallback));
        break;
    }
    default:
    {
        std::cerr << "invalid direction: plan creation failed!\n";
    }
    }
}

void TreeNode::build_real_even_2D()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    assert(inArrayType == rocfft_array_type_real || outArrayType == rocfft_array_type_real);

    scheme = CS_REAL_2D_EVEN;

    if(inArrayType == rocfft_array_type_real) //forward
    {
        // RTRT
        {
            // first row fft
            auto row1Plan = TreeNode::CreateNode(this);
            row1Plan->length.push_back(length[0]);
            row1Plan->dimension = 1;
            row1Plan->length.push_back(length[1]);
            for(size_t index = 2; index < length.size(); index++)
            {
                row1Plan->length.push_back(length[index]);
            }
            row1Plan->build_real_even_1D();
            childNodes.emplace_back(std::move(row1Plan));
        }

        {
            // first transpose
            auto trans1Plan = TreeNode::CreateNode(this);
            trans1Plan->length.push_back(length[0] / 2 + 1);
            trans1Plan->length.push_back(length[1]);
            trans1Plan->scheme    = CS_KERNEL_TRANSPOSE;
            trans1Plan->dimension = 2;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans1Plan->length.push_back(length[index]);
            }
            childNodes.emplace_back(std::move(trans1Plan));
        }

        {
            // second row fft
            auto row2Plan = TreeNode::CreateNode(this);
            row2Plan->length.push_back(length[1]);
            row2Plan->dimension = 1;
            row2Plan->length.push_back(length[0] / 2 + 1);
            for(size_t index = 2; index < length.size(); index++)
            {
                row2Plan->length.push_back(length[index]);
            }
            row2Plan->RecursiveBuildTree();
            childNodes.emplace_back(std::move(row2Plan));
        }

        {
            // second transpose
            auto trans2Plan = TreeNode::CreateNode(this);
            trans2Plan->length.push_back(length[1]);
            trans2Plan->length.push_back(length[0] / 2 + 1);
            trans2Plan->scheme    = CS_KERNEL_TRANSPOSE;
            trans2Plan->dimension = 2;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans2Plan->length.push_back(length[index]);
            }
            childNodes.emplace_back(std::move(trans2Plan));
        }
    }
    else
    {
        // TRTR

        // first transpose
        {
            auto trans1Plan = TreeNode::CreateNode(this);
            trans1Plan->length.push_back(length[0] / 2 + 1);
            trans1Plan->length.push_back(length[1]);
            trans1Plan->scheme    = CS_KERNEL_TRANSPOSE;
            trans1Plan->dimension = 2;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans1Plan->length.push_back(length[index]);
            }
            childNodes.emplace_back(std::move(trans1Plan));
        }

        // c2c row transform
        {
            auto c2cPlan       = TreeNode::CreateNode(this);
            c2cPlan->dimension = 1;
            c2cPlan->length.push_back(length[1]);
            c2cPlan->length.push_back(length[0] / 2 + 1);
            for(size_t index = 2; index < length.size(); index++)
            {
                c2cPlan->length.push_back(length[index]);
            }
            c2cPlan->RecursiveBuildTree();
            childNodes.emplace_back(std::move(c2cPlan));
        }

        // second transpose
        {
            auto trans2plan = TreeNode::CreateNode(this);
            trans2plan->length.push_back(length[1]);
            trans2plan->length.push_back(length[0] / 2 + 1);
            trans2plan->scheme    = CS_KERNEL_TRANSPOSE;
            trans2plan->dimension = 2;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans2plan->length.push_back(length[index]);
            }
            childNodes.emplace_back(std::move(trans2plan));
        }

        // c2r row transform
        {
            auto c2rPlan = TreeNode::CreateNode(this);
            c2rPlan->length.push_back(length[0]);
            c2rPlan->length.push_back(length[1]);
            c2rPlan->dimension = 1;
            for(size_t index = 2; index < length.size(); index++)
            {
                c2rPlan->length.push_back(length[index]);
            }
            c2rPlan->build_real_even_1D();
            childNodes.emplace_back(std::move(c2rPlan));
        }
    }
}

void TreeNode::build_real_even_3D()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);
    assert(inArrayType == rocfft_array_type_real || outArrayType == rocfft_array_type_real);

    scheme = CS_REAL_3D_EVEN;

    // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes.
    bool sbcc_inplace
        = SBCC_dim_available(length, 1, precision) && SBCC_dim_available(length, 2, precision);
    // ensure the fastest dimensions are big enough to get enough
    // column tiles to perform well
    if(length[0] <= 52 && length[1] <= 52)
        sbcc_inplace = false;
    // if all 3 lengths are SBRC-able, then R2C will already be 3
    // kernel.  SBRC should be slightly better since row accesses
    // should be a bit nicer in general than column accesses.
    if(function_pool::has_function(fpkey(length[0] / 2, precision, CS_KERNEL_STOCKHAM_BLOCK_RC))
       && function_pool::has_function(fpkey(length[1], precision, CS_KERNEL_STOCKHAM_BLOCK_RC))
       && function_pool::has_function(fpkey(length[2], precision, CS_KERNEL_STOCKHAM_BLOCK_RC)))
    {
        sbcc_inplace = false;
    }
    auto add_sbcc_children = [this](const std::vector<size_t>& remainingLength) {
        // SBCC along Z dimension
        auto sbccZ    = TreeNode::CreateNode(this);
        sbccZ->length = remainingLength;
        std::swap(sbccZ->length[1], sbccZ->length[2]);
        std::swap(sbccZ->length[0], sbccZ->length[1]);
        sbccZ->scheme = have_SBCC_kernel(sbccZ->length[0], precision) ? CS_KERNEL_STOCKHAM_BLOCK_CC
                                                                      : CS_KERNEL_STOCKHAM;
        childNodes.emplace_back(std::move(sbccZ));

        // SBCC along Y dimension
        auto sbccY    = TreeNode::CreateNode(this);
        sbccY->length = remainingLength;
        std::swap(sbccY->length[0], sbccY->length[1]);
        sbccY->scheme = have_SBCC_kernel(sbccY->length[0], precision) ? CS_KERNEL_STOCKHAM_BLOCK_CC
                                                                      : CS_KERNEL_STOCKHAM;
        childNodes.emplace_back(std::move(sbccY));
    };

    std::vector<size_t> remainingLength = {length[0] / 2 + 1, length[1], length[2]};
    if(inArrayType == rocfft_array_type_real) // forward
    {
        // first row fft + postproc is mandatory for fastest dimension
        {
            auto rcplan       = TreeNode::CreateNode(this);
            rcplan->length    = length;
            rcplan->dimension = 1;
            rcplan->build_real_even_1D(sbcc_inplace);
            childNodes.emplace_back(std::move(rcplan));
        }

        // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes
        if(sbcc_inplace)
        {
            add_sbcc_children(remainingLength);
        }
        // otherwise, handle remaining dimensions with TRTRT
        else
        {
            // first transpose
            {
                auto trans1plan       = TreeNode::CreateNode(this);
                trans1plan->length    = remainingLength;
                trans1plan->scheme    = CS_KERNEL_TRANSPOSE_Z_XY;
                trans1plan->dimension = 2;
                childNodes.emplace_back(std::move(trans1plan));
            }

            {
                auto c1plan       = TreeNode::CreateNode(this);
                c1plan->length    = {childNodes[childNodes.size() - 1]->length[1],
                                  childNodes[childNodes.size() - 1]->length[2],
                                  childNodes[childNodes.size() - 1]->length[0]};
                c1plan->dimension = 1;
                c1plan->placement = rocfft_placement_inplace;
                c1plan->RecursiveBuildTree();
                childNodes.emplace_back(std::move(c1plan));
            }

            // second transpose
            {
                auto trans2plan       = TreeNode::CreateNode(this);
                trans2plan->length    = childNodes[childNodes.size() - 1]->length;
                trans2plan->scheme    = CS_KERNEL_TRANSPOSE_Z_XY;
                trans2plan->dimension = 2;
                childNodes.emplace_back(std::move(trans2plan));
            }

            {
                auto c2plan       = TreeNode::CreateNode(this);
                c2plan->length    = {childNodes[childNodes.size() - 1]->length[1],
                                  childNodes[childNodes.size() - 1]->length[2],
                                  childNodes[childNodes.size() - 1]->length[0]};
                c2plan->dimension = 1;
                c2plan->placement = rocfft_placement_inplace;
                c2plan->RecursiveBuildTree();
                childNodes.emplace_back(std::move(c2plan));
            }

            // third transpose
            {
                auto trans3       = TreeNode::CreateNode(this);
                trans3->length    = childNodes[childNodes.size() - 1]->length;
                trans3->scheme    = CS_KERNEL_TRANSPOSE_Z_XY;
                trans3->dimension = 2;
                childNodes.emplace_back(std::move(trans3));
            }
        }
    }
    else
    {
        if(sbcc_inplace)
        {
            add_sbcc_children(remainingLength);
        }
        // otherwise, TRTRTR
        else
        {
            // transpose
            {
                auto trans3       = TreeNode::CreateNode(this);
                trans3->length    = remainingLength;
                trans3->scheme    = CS_KERNEL_TRANSPOSE_XY_Z;
                trans3->dimension = 2;
                childNodes.emplace_back(std::move(trans3));
            }

            {
                auto c2plan       = TreeNode::CreateNode(this);
                c2plan->length    = {childNodes[childNodes.size() - 1]->length[2],
                                  childNodes[childNodes.size() - 1]->length[0],
                                  childNodes[childNodes.size() - 1]->length[1]};
                c2plan->dimension = 1;
                c2plan->placement = rocfft_placement_inplace;
                c2plan->RecursiveBuildTree();
                childNodes.emplace_back(std::move(c2plan));
            }

            // transpose
            {
                auto trans2       = TreeNode::CreateNode(this);
                trans2->length    = childNodes[childNodes.size() - 1]->length;
                trans2->scheme    = CS_KERNEL_TRANSPOSE_XY_Z;
                trans2->dimension = 2;
                childNodes.emplace_back(std::move(trans2));
            }

            {
                auto c1plan       = TreeNode::CreateNode(this);
                c1plan->length    = {childNodes[childNodes.size() - 1]->length[2],
                                  childNodes[childNodes.size() - 1]->length[0],
                                  childNodes[childNodes.size() - 1]->length[1]};
                c1plan->dimension = 1;
                c1plan->placement = rocfft_placement_inplace;
                c1plan->RecursiveBuildTree();
                childNodes.emplace_back(std::move(c1plan));
            }

            // transpose
            {
                auto trans1       = TreeNode::CreateNode(this);
                trans1->length    = childNodes[childNodes.size() - 1]->length;
                trans1->scheme    = CS_KERNEL_TRANSPOSE_XY_Z;
                trans1->dimension = 2;
                childNodes.emplace_back(std::move(trans1));
            }
        }

        // c2r
        {
            auto crplan       = TreeNode::CreateNode(this);
            crplan->length    = length;
            crplan->dimension = 1;
            crplan->build_real_even_1D(sbcc_inplace);
            childNodes.emplace_back(std::move(crplan));
        }
    }
}

size_t TreeNode::div1DNoPo2(const size_t length0)
{
    auto factor = get_explicity_supported_factor(precision, length0);
    // return 0 means we can't find any supported kernels,
    // happens when debugging, we don't want this assert crash the program
    //assert(factor != 0);
    return (factor > 0) ? length0 / factor : 0;
}

// Compute the large twd decomposition base
// 2-Steps:
//  e.g., ( CeilPo2(10000)+ 1 ) / 2 , returns 7 : (2^7)*(2^7) = 16384 >= 10000
// 3-Steps:
//  e.g., ( CeilPo2(10000)+ 2 ) / 3 , returns 5 : (2^5)*(2^5)*(2^5) = 32768 >= 10000
size_t TreeNode::large_twiddle_base(size_t length, bool use3Steps)
{
    if(use3Steps)
        // 16^3 ~ 64^3, basically enough for 262144
        return std::min((size_t)6, std::max((size_t)4, (CeilPo2(length) + 2) / 3));
    else
        // always return 8, 256^2, if exceed 65536, it becomes 256^3...
        return 8;
}

void TreeNode::build_1D()
{
    // Build a node for a 1D FFT

    if(!SupportedLength(precision, length[0]))
    {
        build_1DBluestein();
        return;
    }

    // single: limit up to 4096, double: up to 2048
    if(length[0] <= Large1DThreshold(precision)
       && function_pool::has_function(fpkey(length[0], precision)))
    {
        scheme = CS_KERNEL_STOCKHAM;
        return;
    }

    size_t divLength1 = 1;
    bool   failed     = false;

    if(IsPo2(length[0])) // multiple kernels involving transpose
    {
        // TODO: wrap the below into a function and check with LDS size
        if(length[0] <= 262144 / PrecisionWidth(precision))
        {
            // Enable block compute under these conditions
            if(1 == PrecisionWidth(precision))
            {
                if(map1DLengthSingle.find(length[0]) != map1DLengthSingle.end())
                {
                    divLength1 = map1DLengthSingle.at(length[0]);
                }
                else
                {
                    failed = true;
                }
            }
            else
            {
                if(map1DLengthDouble.find(length[0]) != map1DLengthDouble.end())
                {
                    divLength1 = map1DLengthDouble.at(length[0]);
                }
                else
                {
                    failed = true;
                }
            }
            // up to 256cc + 256rc for the 1arge-1D, use CRT for even larger
            scheme = (length[0] <= 65536) ? CS_L1D_CC : CS_L1D_CRT;
            // scheme = (length[0] <= 65536 / PrecisionWidth(precision)) ? CS_L1D_CC : CS_L1D_CRT;
        }
        else
        {
            if(length[0] > (Large1DThreshold(precision) * Large1DThreshold(precision)))
            {
                divLength1 = length[0] / Large1DThreshold(precision);
            }
            else
            {
                size_t in_x = 0;
                size_t len  = length[0];
                while(len != 1)
                {
                    len >>= 1;
                    in_x++;
                }
                in_x /= 2;
                divLength1 = (size_t)1 << in_x;
            }
            scheme = CS_L1D_TRTRT;
        }
    }
    else // if not Pow2
    {
        if(precision == rocfft_precision_single)
        {
            if(map1DLengthSingle.find(length[0]) != map1DLengthSingle.end())
            {
                divLength1 = map1DLengthSingle.at(length[0]);
                scheme     = CS_L1D_CC;
            }
            else
            {
                failed = true;
            }
        }
        else if(precision == rocfft_precision_double)
        {
            if(map1DLengthDouble.find(length[0]) != map1DLengthDouble.end())
            {
                divLength1 = map1DLengthDouble.at(length[0]);
                scheme     = CS_L1D_CC;
            }
            else
            {
                failed = true;
            }
        }

        if(failed)
        {
            divLength1 = div1DNoPo2(length[0]);
            scheme     = CS_L1D_TRTRT;
            failed     = (divLength1 == 0);
        }
    }

    if(failed)
    {
        PrintFailInfo(precision, length[0], scheme);
        assert(false); // can't find the length in map1DLengthSingle/Double.
    }

    size_t divLength0 = length[0] / divLength1;

    switch(scheme)
    {
    case CS_L1D_TRTRT:
        build_1DCS_L1D_TRTRT(divLength0, divLength1);
        break;
    case CS_L1D_CC:
        build_1DCS_L1D_CC(divLength0, divLength1);
        break;
    case CS_L1D_CRT:
        build_1DCS_L1D_CRT(divLength0, divLength1);
        break;
    default:
        assert(false);
    }
}

void TreeNode::build_1DBluestein()
{
    // Build a node for a 1D stage using the Bluestein algorithm for
    // general transform lengths.

    scheme     = CS_BLUESTEIN;
    lengthBlue = FindBlue(length[0]);

    auto chirpPlan = TreeNode::CreateNode(this);

    chirpPlan->scheme    = CS_KERNEL_CHIRP;
    chirpPlan->dimension = 1;
    chirpPlan->length.push_back(length[0]);
    chirpPlan->lengthBlue = lengthBlue;
    chirpPlan->direction  = direction;
    chirpPlan->batch      = 1;
    chirpPlan->large1D    = 2 * length[0];
    childNodes.emplace_back(std::move(chirpPlan));

    auto padmulPlan = TreeNode::CreateNode(this);

    padmulPlan->dimension  = 1;
    padmulPlan->length     = length;
    padmulPlan->lengthBlue = lengthBlue;
    padmulPlan->scheme     = CS_KERNEL_PAD_MUL;
    childNodes.emplace_back(std::move(padmulPlan));

    auto fftiPlan = TreeNode::CreateNode(this);

    fftiPlan->dimension = 1;
    fftiPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftiPlan->length.push_back(length[index]);
    }

    fftiPlan->iOffset = 2 * lengthBlue;
    fftiPlan->oOffset = 2 * lengthBlue;
    fftiPlan->scheme  = CS_KERNEL_STOCKHAM;
    fftiPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(fftiPlan));

    auto fftcPlan = TreeNode::CreateNode(this);

    fftcPlan->dimension = 1;
    fftcPlan->length.push_back(lengthBlue);
    fftcPlan->scheme  = CS_KERNEL_STOCKHAM;
    fftcPlan->batch   = 1;
    fftcPlan->iOffset = lengthBlue;
    fftcPlan->oOffset = lengthBlue;
    fftcPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(fftcPlan));

    auto fftmulPlan = TreeNode::CreateNode(this);

    fftmulPlan->dimension = 1;
    fftmulPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftmulPlan->length.push_back(length[index]);
    }

    fftmulPlan->lengthBlue = lengthBlue;
    fftmulPlan->scheme     = CS_KERNEL_FFT_MUL;
    childNodes.emplace_back(std::move(fftmulPlan));

    auto fftrPlan = TreeNode::CreateNode(this);

    fftrPlan->dimension = 1;
    fftrPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftrPlan->length.push_back(length[index]);
    }

    fftrPlan->scheme    = CS_KERNEL_STOCKHAM;
    fftrPlan->direction = -direction;
    fftrPlan->iOffset   = 2 * lengthBlue;
    fftrPlan->oOffset   = 2 * lengthBlue;
    fftrPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(fftrPlan));

    auto resmulPlan = TreeNode::CreateNode(this);

    resmulPlan->dimension  = 1;
    resmulPlan->length     = length;
    resmulPlan->lengthBlue = lengthBlue;
    resmulPlan->scheme     = CS_KERNEL_RES_MUL;
    childNodes.emplace_back(std::move(resmulPlan));
}

void TreeNode::build_1DCS_L1D_TRTRT(const size_t divLength0, const size_t divLength1)
{
    if(!function_pool::has_function(fpkey(divLength0, precision)))
    {
        PrintFailInfo(precision, length[0], scheme, divLength0, CS_KERNEL_STOCKHAM);
        assert(false);
    }

    // first transpose
    auto trans1Plan = TreeNode::CreateNode(this);

    trans1Plan->length.push_back(divLength0);
    trans1Plan->length.push_back(divLength1);

    trans1Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans1Plan->dimension = 2;

    for(size_t index = 1; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(trans1Plan));

    // first row fft
    auto row1Plan = TreeNode::CreateNode(this);

    // twiddling is done in row2 or transpose2
    row1Plan->large1D = 0;

    row1Plan->length.push_back(divLength1);
    row1Plan->length.push_back(divLength0);

    row1Plan->scheme    = CS_KERNEL_STOCKHAM;
    row1Plan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row1Plan->length.push_back(length[index]);
    }

    row1Plan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(row1Plan));

    // second transpose
    auto trans2Plan = TreeNode::CreateNode(this);

    trans2Plan->length.push_back(divLength1);
    trans2Plan->length.push_back(divLength0);

    trans2Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans2Plan->dimension = 2;

    trans2Plan->large1D = length[0];

    for(size_t index = 1; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(trans2Plan));

    // second row fft
    auto row2Plan = TreeNode::CreateNode(this);

    row2Plan->length.push_back(divLength0);
    row2Plan->length.push_back(divLength1);

    row2Plan->scheme    = CS_KERNEL_STOCKHAM;
    row2Plan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row2Plan->length.push_back(length[index]);
    }

    // algorithm is set up in a way that row2 does not recurse
    assert(divLength0 <= Large1DThreshold(this->precision));

    childNodes.emplace_back(std::move(row2Plan));

    // third transpose
    auto trans3Plan = TreeNode::CreateNode(this);

    trans3Plan->length.push_back(divLength0);
    trans3Plan->length.push_back(divLength1);

    trans3Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans3Plan->dimension = 2;

    for(size_t index = 1; index < length.size(); index++)
    {
        trans3Plan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(trans3Plan));
}

void TreeNode::build_1DCS_L1D_CC(const size_t divLength0, const size_t divLength1)
{
    //  Note:
    //  The kernel CS_KERNEL_STOCKHAM_BLOCK_CC and CS_KERNEL_STOCKHAM_BLOCK_RC
    //  are only enabled for outplace for now. Check more details in generator.file.cpp,
    //  and in generated kernel_lunch_single_large.cpp.h
    if(!function_pool::has_function(fpkey(divLength1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC)))
    {
        PrintFailInfo(precision, length[0], scheme, divLength1, CS_KERNEL_STOCKHAM_BLOCK_CC);
        assert(false);
    }
    if(!function_pool::has_function(fpkey(divLength0, precision, CS_KERNEL_STOCKHAM_BLOCK_RC)))
    {
        PrintFailInfo(precision, length[0], scheme, divLength0, CS_KERNEL_STOCKHAM_BLOCK_RC);
        assert(false);
    }

    // first plan, column-to-column
    auto col2colPlan = TreeNode::CreateNode(this);

    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D = length[0];

    // decompose the large twd table for L1D_CC
    // exclude some exceptions that don't get benefit from 3-step LargeTwd (set in FFTKernel)
    auto krn = function_pool::get_kernel(fpkey(divLength1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
    col2colPlan->largeTwd3Steps = krn.use_3steps_large_twd;
    col2colPlan->largeTwdBase   = large_twiddle_base(length[0], col2colPlan->largeTwd3Steps);

    col2colPlan->length.push_back(divLength1);
    col2colPlan->length.push_back(divLength0);

    col2colPlan->scheme    = CS_KERNEL_STOCKHAM_BLOCK_CC;
    col2colPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        col2colPlan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(col2colPlan));

    // second plan, row-to-column
    auto row2colPlan = TreeNode::CreateNode(this);

    row2colPlan->length.push_back(divLength0);
    row2colPlan->length.push_back(divLength1);

    row2colPlan->scheme    = CS_KERNEL_STOCKHAM_BLOCK_RC;
    row2colPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row2colPlan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(row2colPlan));
}

void TreeNode::build_1DCS_L1D_CRT(const size_t divLength0, const size_t divLength1)
{
    if(!function_pool::has_function(fpkey(divLength1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC)))
    {
        PrintFailInfo(precision, length[0], scheme, divLength1, CS_KERNEL_STOCKHAM_BLOCK_CC);
        assert(false);
    }
    if(!function_pool::has_function(fpkey(divLength0, precision, CS_KERNEL_STOCKHAM)))
    {
        PrintFailInfo(precision, length[0], scheme, divLength0, CS_KERNEL_STOCKHAM);
        assert(false);
    }

    // first plan, column-to-column
    auto col2colPlan = TreeNode::CreateNode(this);

    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D = length[0];

    // decompose the large twd table for L1D_CRT
    // exclude some exceptions that don't get benefit from 3-step LargeTwd (set in FFTKernel)
    auto kernel
        = function_pool::get_kernel(fpkey(divLength1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
    col2colPlan->largeTwd3Steps = kernel.use_3steps_large_twd;
    col2colPlan->largeTwdBase   = large_twiddle_base(length[0], col2colPlan->largeTwd3Steps);

    col2colPlan->length.push_back(divLength1);
    col2colPlan->length.push_back(divLength0);

    col2colPlan->scheme    = CS_KERNEL_STOCKHAM_BLOCK_CC;
    col2colPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        col2colPlan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(col2colPlan));

    // second plan, row-to-row
    auto row2rowPlan = TreeNode::CreateNode(this);

    row2rowPlan->length.push_back(divLength0);
    row2rowPlan->length.push_back(divLength1);

    row2rowPlan->scheme    = CS_KERNEL_STOCKHAM;
    row2rowPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row2rowPlan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(row2rowPlan));

    // third plan, transpose
    auto transPlan = TreeNode::CreateNode(this);

    transPlan->length.push_back(divLength0);
    transPlan->length.push_back(divLength1);

    transPlan->scheme    = CS_KERNEL_TRANSPOSE;
    transPlan->dimension = 2;

    for(size_t index = 1; index < length.size(); index++)
    {
        transPlan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(transPlan));
}

void TreeNode::build_CS_2D_RC()
{
    if(!function_pool::has_function(fpkey(length[1], precision, CS_KERNEL_STOCKHAM_BLOCK_CC)))
    {
        PrintFailInfo(precision, length[1], scheme, length[1], CS_KERNEL_STOCKHAM_BLOCK_CC);
        assert(false);
    }

    // row fft
    auto rowPlan = TreeNode::CreateNode(this);

    rowPlan->length.push_back(length[0]);
    rowPlan->dimension = 1;
    rowPlan->length.push_back(length[1]);

    for(size_t index = 2; index < length.size(); index++)
    {
        rowPlan->length.push_back(length[index]);
    }

    rowPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(rowPlan));

    // column fft
    auto colPlan = TreeNode::CreateNode(this);

    colPlan->length.push_back(length[1]);
    colPlan->dimension = 1;
    colPlan->length.push_back(length[0]);
    colPlan->large1D = 0; // No twiddle factor in sbcc kernel

    for(size_t index = 2; index < length.size(); index++)
    {
        colPlan->length.push_back(length[index]);
    }

    colPlan->scheme = CS_KERNEL_STOCKHAM_BLOCK_CC;
    childNodes.emplace_back(std::move(colPlan));
}

void TreeNode::build_CS_2D_RTRT()
{
    // first row fft
    auto row1Plan = TreeNode::CreateNode(this);

    row1Plan->length.push_back(length[0]);
    row1Plan->dimension = 1;
    row1Plan->length.push_back(length[1]);

    for(size_t index = 2; index < length.size(); index++)
    {
        row1Plan->length.push_back(length[index]);
    }

    row1Plan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(row1Plan));

    // first transpose
    auto trans1Plan = TreeNode::CreateNode(this);

    trans1Plan->length.push_back(length[0]);
    trans1Plan->length.push_back(length[1]);

    trans1Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans1Plan->dimension = 2;

    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(trans1Plan));

    // second row fft
    auto row2Plan = TreeNode::CreateNode(this);

    row2Plan->length.push_back(length[1]);
    row2Plan->dimension = 1;
    row2Plan->length.push_back(length[0]);

    for(size_t index = 2; index < length.size(); index++)
    {
        row2Plan->length.push_back(length[index]);
    }

    row2Plan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(row2Plan));

    // second transpose
    auto trans2Plan = TreeNode::CreateNode(this);

    trans2Plan->length.push_back(length[1]);
    trans2Plan->length.push_back(length[0]);

    trans2Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans2Plan->dimension = 2;

    for(size_t index = 2; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }

    childNodes.emplace_back(std::move(trans2Plan));
}

void TreeNode::build_CS_3D_RTRT()
{
    // 2d fft
    auto xyPlan       = TreeNode::CreateNode(this);
    xyPlan->length    = length;
    xyPlan->dimension = 2;
    xyPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(xyPlan));

    // first transpose
    auto trans1Plan       = TreeNode::CreateNode(this);
    trans1Plan->length    = length;
    trans1Plan->scheme    = CS_KERNEL_TRANSPOSE_XY_Z;
    trans1Plan->dimension = 2;
    childNodes.emplace_back(std::move(trans1Plan));

    // z fft
    auto zPlan       = TreeNode::CreateNode(this);
    zPlan->dimension = 1;
    zPlan->length.push_back(length[2]);
    zPlan->length.push_back(length[0]);
    zPlan->length.push_back(length[1]);
    zPlan->RecursiveBuildTree();

    // second transpose
    auto trans2Plan       = TreeNode::CreateNode(this);
    trans2Plan->length    = zPlan->length;
    trans2Plan->scheme    = CS_KERNEL_TRANSPOSE_Z_XY;
    trans2Plan->dimension = 2;
    childNodes.emplace_back(std::move(zPlan));
    childNodes.emplace_back(std::move(trans2Plan));
}

void TreeNode::build_CS_3D_TRTRTR()
{
    scheme                         = CS_3D_TRTRTR;
    std::vector<size_t> cur_length = length;

    for(int i = 0; i < 6; i += 2)
    {
        // transpose Z_XY
        auto trans_plan       = TreeNode::CreateNode(this);
        trans_plan->length    = cur_length;
        trans_plan->scheme    = CS_KERNEL_TRANSPOSE_Z_XY;
        trans_plan->dimension = 2;
        childNodes.emplace_back(std::move(trans_plan));

        std::swap(cur_length[0], cur_length[1]);
        std::swap(cur_length[1], cur_length[2]);

        // row ffts
        auto row_plan       = TreeNode::CreateNode(this);
        row_plan->length    = cur_length;
        row_plan->dimension = 1;
        row_plan->RecursiveBuildTree();
        childNodes.emplace_back(std::move(row_plan));
    }
}

void TreeNode::build_CS_3D_BLOCK_RC()
{
    scheme                         = CS_3D_BLOCK_RC;
    std::vector<size_t> cur_length = length;

    size_t total_sbrc = 0;
    for(int i = 0; i < 3; ++i)
    {
        // if we have an sbrc kernel for this length, use it,
        // otherwise, fall back to row FFT+transpose
        bool have_sbrc = function_pool::has_function(
            fpkey(cur_length.front(), precision, CS_KERNEL_STOCKHAM_BLOCK_RC));
        // ensure the kernel would be tile-aligned
        if(have_sbrc)
        {
            size_t bwd, wgs, lds;
            GetBlockComputeTable(cur_length[0], bwd, wgs, lds);
            if(cur_length[2] % bwd != 0)
                have_sbrc = false;
        }
        if(have_sbrc)
            ++total_sbrc;

        // if we're doing in-place, we are unable to do more than 2
        // sbrc kernels.  we'd ideally want 3 sbrc, but each needs to
        // be out-of-place:
        // - kernel 1: IN -> TEMP
        // - kernel 2: TEMP -> IN
        // - kernel 3: IN -> ???
        //
        // So we have no way to put the results back into IN.  So
        // limit ourselves to 2 sbrc kernels in that case.
        if(total_sbrc >= 3 && placement == rocfft_placement_inplace)
            have_sbrc = false;

        if(have_sbrc)
        {
            auto sbrc_node    = TreeNode::CreateNode(this);
            sbrc_node->length = cur_length;
            sbrc_node->scheme = CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z;
            childNodes.emplace_back(std::move(sbrc_node));
        }
        else
        {
            // row ffts
            auto row_plan       = TreeNode::CreateNode(this);
            row_plan->length    = cur_length;
            row_plan->dimension = 1;
            row_plan->RecursiveBuildTree();
            childNodes.emplace_back(std::move(row_plan));

            // transpose XY_Z
            auto trans_plan       = TreeNode::CreateNode(this);
            trans_plan->length    = cur_length;
            trans_plan->scheme    = CS_KERNEL_TRANSPOSE_XY_Z;
            trans_plan->dimension = 2;
            childNodes.emplace_back(std::move(trans_plan));
        }

        std::swap(cur_length[2], cur_length[1]);
        std::swap(cur_length[1], cur_length[0]);
    }
}

void TreeNode::build_CS_3D_RC()
{
    // 2d fft
    auto xyPlan = TreeNode::CreateNode(this);

    xyPlan->length.push_back(length[0]);
    xyPlan->length.push_back(length[1]);
    xyPlan->dimension = 2;
    xyPlan->length.push_back(length[2]);

    for(size_t index = 3; index < length.size(); index++)
    {
        xyPlan->length.push_back(length[index]);
    }

    xyPlan->RecursiveBuildTree();
    childNodes.emplace_back(std::move(xyPlan));

    // z col fft
    auto zPlan = TreeNode::CreateNode(this);

    // make this always inplace, and let the previous one follow the root's placement
    zPlan->placement = rocfft_placement_inplace;
    zPlan->length.push_back(length[2]);
    zPlan->dimension = 1;
    zPlan->length.push_back(length[0]);
    zPlan->length.push_back(length[1]);

    for(size_t index = 3; index < length.size(); index++)
    {
        zPlan->length.push_back(length[index]);
    }

    // zPlan->scheme = CS_KERNEL_3D_STOCKHAM_BLOCK_CC;
    zPlan->scheme = CS_KERNEL_STOCKHAM_BLOCK_CC;
    childNodes.emplace_back(std::move(zPlan));
}

struct TreeNode::TraverseState
{
    TraverseState(const ExecPlan& execPlan)
        : rootPlan(execPlan.rootPlan.get())
    {
        TraverseFullSequence(rootPlan);
    }
    const TreeNode* rootPlan;
    // All nodes in the plan (leaf + non-leaf), ordered by how they
    // would be executed
    std::vector<const TreeNode*> fullSeq;

private:
    // recursively fill fullSeq
    void TraverseFullSequence(const TreeNode* node)
    {
        fullSeq.push_back(node);
        for(auto& child : node->childNodes)
            TraverseFullSequence(child.get());
    }
};

/// Buffer assignment
void TreeNode::SetInputBuffer(TraverseState& state)
{
    // find the given node in the full sequence
    auto it = std::find(state.fullSeq.begin(), state.fullSeq.end(), this);
    if(it == state.fullSeq.end())
    {
        // How did we get a node that wasn't in sequence?
        // Trigger an error in buffer assignment.
        assert(false);
        obIn = OB_UNINIT;
    }
    // Looking backwards from this node, find the closest leaf
    // node.  Exclude CS_KERNEL_CHIRP, since those effectively take
    // no inputs and output to a separate out-of-band buffer that
    // is not part of the chain.
    auto rev_begin = std::make_reverse_iterator(it);
    auto rev_end   = std::make_reverse_iterator(state.fullSeq.begin());
    auto prevLeaf  = std::find_if(rev_begin, rev_end, [](const TreeNode* n) {
        return n->childNodes.empty() && n->scheme != CS_KERNEL_CHIRP;
    });
    if(prevLeaf == rev_end)
    {
        // There is no earlier leaf node, so we should use the user's input for this node.
        obIn = state.rootPlan->obIn;
    }
    else
    {
        // There is an earlier leaf node, so we have to use its output as this node's input.
        obIn = (*prevLeaf)->obOut;
    }
}

// Assign buffers, taking into account out-of-place transposes and
// padded buffers.
// NB: this recursive function modifies the parameters in the parent call.
void TreeNode::TraverseTreeAssignBuffersLogicA(TraverseState&   state,
                                               OperatingBuffer& flipIn,
                                               OperatingBuffer& flipOut,
                                               OperatingBuffer& obOutBuf)
{
    // Input buffer for 'this' is dictated by our traversal state.
    // Either we're the first node, which means we use the input the
    // user said to use, or we use the output of the last traversed
    // node.
    //
    // obIn might have already been set in special cases during plan
    // building, so only set it if it's not already set.
    if(obIn == OB_UNINIT)
        SetInputBuffer(state);

    if(parent == nullptr)
    {
        // Set flipIn, flipOut, and oboutBuf for the root node.
        assert(flipIn == OB_UNINIT);
        assert(flipOut == OB_UNINIT);
        assert(obOutBuf == OB_UNINIT);
        switch(scheme)
        {
        case CS_REAL_TRANSFORM_USING_CMPLX:
            flipIn   = OB_TEMP_CMPLX_FOR_REAL;
            flipOut  = OB_TEMP;
            obOutBuf = OB_TEMP_CMPLX_FOR_REAL;
            break;
        case CS_BLUESTEIN:
            flipIn   = OB_TEMP_BLUESTEIN;
            flipOut  = OB_TEMP;
            obOutBuf = OB_TEMP_BLUESTEIN;
            break;
        default:
            flipIn   = OB_USER_OUT;
            flipOut  = OB_TEMP;
            obOutBuf = OB_USER_OUT;
        }
    }

#if 0
    auto        here = this;
    auto        up   = parent;
    std::string tabs;
    while(up != nullptr && here != up)
    {
        here = up;
        up   = parent->parent;
        tabs += "\t";
    }
    rocfft_cout << "TraverseTreeAssignBuffersLogicA: " << PrintScheme(scheme) << ": "
                << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << "\n"
                << tabs << "\tobIn: " << PrintOperatingBuffer(obIn) << "\n"
                << tabs << "\tobOut: " << PrintOperatingBuffer(obOut) << "\n"
                << tabs << "\tflipIn: " << PrintOperatingBuffer(flipIn) << "\n"
                << tabs << "\tflipOut: " << PrintOperatingBuffer(flipOut) << "\n"
                << tabs << "\tobOutBuf: " << PrintOperatingBuffer(obOutBuf) << std::endl;
#endif

    switch(scheme)
    {
    case CS_REAL_TRANSFORM_USING_CMPLX:
        assign_buffers_CS_REAL_TRANSFORM_USING_CMPLX(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_REAL_TRANSFORM_EVEN:
        assign_buffers_CS_REAL_TRANSFORM_EVEN(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_REAL_2D_EVEN:
        assign_buffers_CS_REAL_2D_EVEN(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_REAL_3D_EVEN:
        assign_buffers_CS_REAL_3D_EVEN(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_BLUESTEIN:
        assign_buffers_CS_BLUESTEIN(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_L1D_TRTRT:
        assign_buffers_CS_L1D_TRTRT(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_L1D_CC:
        assign_buffers_CS_L1D_CC(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_L1D_CRT:
        assign_buffers_CS_L1D_CRT(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_2D_RTRT:
    case CS_3D_RTRT:
        assign_buffers_CS_RTRT(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_2D_RC:
    case CS_3D_RC:
        assign_buffers_CS_RC(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_3D_TRTRTR:
        assign_buffers_CS_3D_TRTRTR(state, flipIn, flipOut, obOutBuf);
        break;
    case CS_3D_BLOCK_RC:
        assign_buffers_CS_3D_BLOCK_RC(state, flipIn, flipOut, obOutBuf);
        break;
    default:
        if(parent == nullptr)
        {
            obOut = obOutBuf;
        }
        else
        {
            assert(obIn != OB_UNINIT);
            assert(obOut != OB_UNINIT);
            if(obIn != obOut)
            {
                std::swap(flipIn, flipOut);
            }
        }
    }

    // Assert that all operating buffers have been assigned
    assert(obIn != OB_UNINIT);
    assert(obOut != OB_UNINIT);
    for(int i = 0; i < childNodes.size(); ++i)
    {
        assert(childNodes[i]->obIn != OB_UNINIT);
        assert(childNodes[i]->obOut != OB_UNINIT);
    }

    // Assert that the kernel chain is connected
    for(int i = 1; i < childNodes.size(); ++i)
    {
        if(childNodes[i - 1]->scheme == CS_KERNEL_CHIRP)
        {
            // The Bluestein algorithm uses a separate buffer which is
            // convoluted with the input; the chain assumption isn't true here.
            // NB: we assume that the CS_KERNEL_CHIRP is first in the chain.
            continue;
        }
        assert(childNodes[i - 1]->obOut == childNodes[i]->obIn);
    }
}

void TreeNode::assign_buffers_CS_REAL_TRANSFORM_USING_CMPLX(TraverseState&   state,
                                                            OperatingBuffer& flipIn,
                                                            OperatingBuffer& flipOut,
                                                            OperatingBuffer& obOutBuf)
{
    assert(parent == nullptr);
    assert(childNodes.size() == 3);

    assert((direction == -1 && childNodes[0]->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
           || (direction == 1 && childNodes[0]->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX));

    obOut = OB_USER_OUT;

    assert((direction == -1 && childNodes[0]->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
           || (direction == 1 && childNodes[0]->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX));

    childNodes[0]->SetInputBuffer(state);
    childNodes[0]->obOut        = OB_TEMP_CMPLX_FOR_REAL;
    childNodes[0]->inArrayType  = inArrayType;
    childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut       = flipIn;
    childNodes[1]->inArrayType = rocfft_array_type_complex_interleaved;
    //To check: we might to check childNodes[1]->outArrayType depending on flipIn
    childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
    size_t cs = childNodes[1]->childNodes.size();
    if(cs)
    {
        if(childNodes[1]->scheme == CS_BLUESTEIN)
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_BLUESTEIN);
            assert(childNodes[1]->childNodes[1]->obIn == OB_TEMP_CMPLX_FOR_REAL);
        }
        else
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_CMPLX_FOR_REAL);
        }
        assert(childNodes[1]->childNodes[cs - 1]->obOut == OB_TEMP_CMPLX_FOR_REAL);
    }

    assert((direction == -1 && childNodes[2]->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
           || (direction == 1 && childNodes[2]->scheme == CS_KERNEL_COPY_CMPLX_TO_R));
    childNodes[2]->SetInputBuffer(state);
    childNodes[2]->obOut        = obOut;
    childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
    childNodes[2]->outArrayType = outArrayType;
}

void TreeNode::assign_buffers_CS_REAL_TRANSFORM_EVEN(TraverseState&   state,
                                                     OperatingBuffer& flipIn,
                                                     OperatingBuffer& flipOut,
                                                     OperatingBuffer& obOutBuf)
{
    auto flipIn0  = flipIn;
    auto flipOut0 = flipOut;

    if(direction == -1)
    {
        // real-to-complex

        flipIn  = obIn;
        flipOut = OB_TEMP;

        // apply callback
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = obIn;
        childNodes[0]->inArrayType  = rocfft_array_type_real;
        childNodes[0]->outArrayType = rocfft_array_type_real;

        // we would have only 2 child nodes if we're able to fuse the
        // post-processing into the single FFT kernel
        bool fusedPostProcessing = childNodes[1]->ebtype == EmbeddedType::Real2C_POST;

        // complex FFT kernel
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut       = fusedPostProcessing ? obOut : obIn;
        childNodes[1]->inArrayType = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType
            = fusedPostProcessing ? outArrayType : rocfft_array_type_complex_interleaved;
        childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

        if(!fusedPostProcessing)
        {
            // real-to-complex post kernel
            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut        = obOut;
            childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
            childNodes[2]->outArrayType = outArrayType;
        }

        flipIn  = flipIn0;
        flipOut = flipOut0;
    }
    else
    {
        // we would only have 2 child nodes if we fused the
        // pre-processing into the single FFT kernel
        bool fusedPreProcessing = childNodes[0]->ebtype == EmbeddedType::C2Real_PRE;

        auto& fftNode = fusedPreProcessing ? childNodes[0] : childNodes[1];

        if(!fusedPreProcessing)
        {
            // complex-to-real

            // complex-to-real pre kernel
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut = obOut;

            childNodes[0]->inArrayType  = inArrayType;
            childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

            // NB: The case here indicates parent's input buffer is not
            //     complex_planar or hermitian_planar, so the child must
            //     be a hermitian_interleaved.
            if(inArrayType == rocfft_array_type_complex_interleaved)
            {
                childNodes[0]->inArrayType = rocfft_array_type_hermitian_interleaved;
            }
        }

        // complex FFT kernel
        fftNode->SetInputBuffer(state);
        fftNode->obOut = obOut;
        fftNode->inArrayType
            = fusedPreProcessing ? inArrayType : rocfft_array_type_complex_interleaved;
        fftNode->outArrayType = rocfft_array_type_complex_interleaved;
        fftNode->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

        // apply callback
        childNodes.back()->obIn         = obOut;
        childNodes.back()->obOut        = obOut;
        childNodes.back()->inArrayType  = rocfft_array_type_real;
        childNodes.back()->outArrayType = rocfft_array_type_real;
    }
}

void TreeNode::assign_buffers_CS_REAL_2D_EVEN(TraverseState&   state,
                                              OperatingBuffer& flipIn,
                                              OperatingBuffer& flipOut,
                                              OperatingBuffer& obOutBuf)
{
    assert(scheme == CS_REAL_2D_EVEN);
    assert(parent == nullptr);

    if(direction == -1)
    {
        // RTRT

        // real-to-complex
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = obOut;
        childNodes[0]->inArrayType  = inArrayType;
        childNodes[0]->outArrayType = outArrayType;
        childNodes[0]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

        // T
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut        = flipOut;
        childNodes[1]->inArrayType  = childNodes[0]->outArrayType;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;

        // complex-to-complex
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut = flipOut;
        childNodes[2]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
        childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[2]->outArrayType = rocfft_array_type_complex_interleaved;

        // T
        childNodes[3]->SetInputBuffer(state);
        childNodes[3]->obOut        = obOut;
        childNodes[3]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[3]->outArrayType = outArrayType;
    }
    else
    { // TRTR

        // The first transform only gets to play with the input buffer.
        auto flipIn0 = flipIn;
        flipIn       = obIn;

        // T
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = flipOut;
        childNodes[0]->inArrayType  = inArrayType;
        childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

        // complex-to-complex
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut        = flipOut;
        childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;
        childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

        // T
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut        = obIn;
        childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[2]->outArrayType = inArrayType;

        // complex-to-real
        childNodes[3]->SetInputBuffer(state);
        childNodes[3]->obOut        = obOut;
        childNodes[3]->inArrayType  = inArrayType;
        childNodes[3]->outArrayType = rocfft_array_type_real;

        flipIn = flipIn0;

        childNodes[3]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
    }
}

void TreeNode::assign_buffers_CS_REAL_3D_EVEN(TraverseState&   state,
                                              OperatingBuffer& flipIn,
                                              OperatingBuffer& flipOut,
                                              OperatingBuffer& obOutBuf)
{
    assert(scheme == CS_REAL_3D_EVEN);
    assert(parent == nullptr);

    obOut = OB_USER_OUT;

    if(direction == -1)
    {
        flipIn  = obIn;
        flipOut = OB_TEMP;

        // R: r2c
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = obOutBuf;
        childNodes[0]->inArrayType  = inArrayType;
        childNodes[0]->outArrayType = outArrayType;
        childNodes[0]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

        flipIn   = OB_TEMP;
        flipOut  = obOut;
        obOutBuf = obOut;

        // in-place SBCC for higher dimensions
        if(childNodes.size() == 3)
        {
            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = childNodes[1]->obIn;
            childNodes[1]->inArrayType  = outArrayType;
            childNodes[1]->outArrayType = outArrayType;

            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut        = childNodes[2]->obIn;
            childNodes[2]->inArrayType  = outArrayType;
            childNodes[2]->outArrayType = outArrayType;
        }
        // RTRTRT
        else if(childNodes.size() == 6)
        {

            // NB: for out-of-place transforms, we can't fit the result of the first r2c transform into
            // the input buffer.

            // T
            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = obOutBuf;
            childNodes[1]->inArrayType  = childNodes[0]->outArrayType;
            childNodes[1]->outArrayType = outArrayType;

            // R: c2c
            childNodes[2]->inArrayType  = childNodes[1]->outArrayType;
            childNodes[2]->outArrayType = outArrayType;
            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut = obOutBuf;
            flipIn               = childNodes[2]->obIn;
            flipOut              = obOutBuf;
            childNodes[2]->TraverseTreeAssignBuffersLogicA(state, flipOut, flipIn, obIn);

            // T
            childNodes[3]->SetInputBuffer(state);
            childNodes[3]->obOut        = OB_TEMP;
            childNodes[3]->inArrayType  = childNodes[2]->outArrayType;
            childNodes[3]->outArrayType = rocfft_array_type_complex_interleaved;

            // R: c2c
            childNodes[4]->SetInputBuffer(state);
            childNodes[4]->obOut = OB_TEMP;
            childNodes[4]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
            childNodes[4]->inArrayType  = childNodes[3]->outArrayType;
            childNodes[4]->outArrayType = rocfft_array_type_complex_interleaved;

            // T
            childNodes[5]->inArrayType  = childNodes[4]->outArrayType;
            childNodes[5]->outArrayType = outArrayType;
            childNodes[5]->SetInputBuffer(state);
            childNodes[5]->obOut = obOutBuf;
        }
    }
    else
    {
        // in-place SBCC for higher dimensions
        if(childNodes.size() == 3)
        {
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut        = childNodes[0]->obIn;
            childNodes[0]->inArrayType  = inArrayType;
            childNodes[0]->outArrayType = inArrayType;

            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = childNodes[1]->obIn;
            childNodes[1]->inArrayType  = inArrayType;
            childNodes[1]->outArrayType = inArrayType;
        }
        // TRTR
        else
        {
            // NB: only c2r can fit into the output buffer for out-of-place transforms.

            // Transpose
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut        = OB_TEMP;
            childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

            // c2c
            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = OB_USER_IN;
            childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
            childNodes[1]->outArrayType = inArrayType;
            childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

            // Transpose
            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut        = OB_TEMP;
            childNodes[2]->inArrayType  = childNodes[1]->outArrayType;
            childNodes[2]->outArrayType = rocfft_array_type_complex_interleaved;

            // c2c
            childNodes[3]->SetInputBuffer(state);
            childNodes[3]->obOut        = OB_USER_IN;
            childNodes[3]->inArrayType  = rocfft_array_type_complex_interleaved;
            childNodes[3]->outArrayType = inArrayType;
            childNodes[3]->TraverseTreeAssignBuffersLogicA(state, flipOut, flipIn, obOutBuf);

            // Transpose
            childNodes[4]->SetInputBuffer(state);
            childNodes[4]->obOut        = OB_TEMP;
            childNodes[4]->inArrayType  = childNodes[3]->outArrayType;
            childNodes[4]->outArrayType = rocfft_array_type_complex_interleaved;
        }

        // c2r
        auto& previousNode = childNodes[childNodes.size() - 2];
        childNodes.back()->SetInputBuffer(state);
        childNodes.back()->obOut        = obOutBuf;
        childNodes.back()->inArrayType  = previousNode->outArrayType;
        childNodes.back()->outArrayType = outArrayType;
        childNodes.back()->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

        obOut = childNodes.back()->obOut;
    }

#if 0
    rocfft_cout << PrintScheme(scheme) << std::endl;
    for(int i = 0; i < childNodes.size(); ++i)
    {
        rocfft_cout << i << ": " << PrintScheme(childNodes[i]->scheme) << " : "
                  << PrintOperatingBuffer(childNodes[i]->obIn) << " -> "
                  << PrintOperatingBuffer(childNodes[i]->obOut) << std::endl;
    }
#endif
}

void TreeNode::assign_buffers_CS_BLUESTEIN(TraverseState&   state,
                                           OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    assert(childNodes.size() == 7);

    OperatingBuffer savFlipIn  = flipIn;
    OperatingBuffer savFlipOut = flipOut;
    OperatingBuffer savOutBuf  = obOutBuf;

    flipIn   = OB_TEMP_BLUESTEIN;
    flipOut  = OB_TEMP;
    obOutBuf = OB_TEMP_BLUESTEIN;

    // CS_KERNEL_CHIRP effectively takes no inputs and does not
    // connect to the previous kernel in the chain, so don't assign
    // obIn using SetInputBuffer.
    assert(childNodes[0]->scheme == CS_KERNEL_CHIRP);
    childNodes[0]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[0]->obOut = OB_TEMP_BLUESTEIN;

    assert(childNodes[1]->scheme == CS_KERNEL_PAD_MUL);
    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut = OB_TEMP_BLUESTEIN;

    childNodes[2]->SetInputBuffer(state);
    childNodes[2]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[2]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

    childNodes[3]->SetInputBuffer(state);
    childNodes[3]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[3]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

    assert(childNodes[4]->scheme == CS_KERNEL_FFT_MUL);
    childNodes[4]->SetInputBuffer(state);
    childNodes[4]->obOut = OB_TEMP_BLUESTEIN;

    childNodes[5]->SetInputBuffer(state);
    childNodes[5]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[5]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

    assert(childNodes[6]->scheme == CS_KERNEL_RES_MUL);
    childNodes[6]->SetInputBuffer(state);
    childNodes[6]->obOut = (parent == nullptr) ? OB_USER_OUT : obOut;

    obOut = childNodes[6]->obOut;

    flipIn   = savFlipIn;
    flipOut  = savFlipOut;
    obOutBuf = savOutBuf;
}
void TreeNode::assign_buffers_CS_L1D_TRTRT(TraverseState&   state,
                                           OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
    {
        obOutBuf = OB_USER_OUT;

        // T
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut = flipOut;

        // R
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut = obOut == flipIn ? flipOut : flipIn;
        if(childNodes[1]->childNodes.size())
        {
            childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
        }

        // T
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut = obOut == flipIn ? flipIn : flipOut;

        // R
        childNodes[3]->SetInputBuffer(state);
        childNodes[3]->obOut = obOut == flipIn ? flipOut : flipIn;

        // T
        childNodes[4]->SetInputBuffer(state);
        childNodes[4]->obOut = obOutBuf;
    }
    else
    {

        // T
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut = childNodes[0]->obIn == flipIn ? flipOut : flipIn;

        // R
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut = obOut == flipIn ? flipOut : flipIn;
        if(childNodes[1]->childNodes.size())
        {
            childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
        }

        // T
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut = obOut == flipIn ? flipIn : flipOut;

        // R
        childNodes[3]->SetInputBuffer(state);
        childNodes[3]->obOut = obOut == flipIn ? flipOut : flipIn;

        // T
        childNodes[4]->SetInputBuffer(state);
        childNodes[4]->obOut = obOut;
    }
}

void TreeNode::assign_buffers_CS_L1D_CC(TraverseState&   state,
                                        OperatingBuffer& flipIn,
                                        OperatingBuffer& flipOut,
                                        OperatingBuffer& obOutBuf)
{
    if(obOut == OB_UNINIT)
    {
        if(parent == nullptr)
        {
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut = OB_TEMP;

            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut = obOutBuf;
        }
        else
        {

            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut = flipOut;

            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut = flipIn;
        }

        obOut = childNodes[1]->obOut;
    }
    else
    {
        childNodes[0]->SetInputBuffer(state);
        // childNodes[1] must be out-of-place:
        childNodes[0]->obOut = flipOut == obOut ? flipIn : flipOut;

        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut = obOut;
    }
}

void TreeNode::assign_buffers_CS_L1D_CRT(TraverseState&   state,
                                         OperatingBuffer& flipIn,
                                         OperatingBuffer& flipOut,
                                         OperatingBuffer& obOutBuf)
{
    if(obOut == OB_UNINIT)
    {
        if(parent == nullptr)
        {
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut = OB_TEMP;

            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut = OB_TEMP;

            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut = obOutBuf;
        }
        else
        {
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut = flipOut;

            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut = flipOut;

            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut = flipIn;
        }

        obOut = childNodes[2]->obOut;
    }
    else
    {
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut = flipOut;

        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut = obOut == flipIn ? flipOut : flipIn;

        // Last node is a transpose and must be out-of-place:
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut = obOut;
    }
}

void TreeNode::assign_buffers_CS_RTRT(TraverseState&   state,
                                      OperatingBuffer& flipIn,
                                      OperatingBuffer& flipOut,
                                      OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
    {
        obOut = OB_USER_OUT;
    }

    // Copy the flip buffers, which are swapped by recursive calls.
    auto flipIn0  = flipIn;
    auto flipOut0 = flipOut;

    // Transform:
    childNodes[0]->SetInputBuffer(state);
    childNodes[0]->obOut = flipIn;
    childNodes[0]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

    // Transpose:
    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut = flipOut0;

    // Stockham:
    childNodes[2]->SetInputBuffer(state);
    childNodes[2]->obOut = flipOut0;
    childNodes[2]->TraverseTreeAssignBuffersLogicA(state, flipOut0, flipIn0, obOutBuf);

    // Transpose:
    childNodes[3]->SetInputBuffer(state);
    childNodes[3]->obOut = obOut;

    // Transposes must be out-of-place:
    assert(childNodes[1]->obIn != childNodes[1]->obOut);
    assert(childNodes[3]->obIn != childNodes[3]->obOut);
}

void TreeNode::assign_buffers_CS_RC(TraverseState&   state,
                                    OperatingBuffer& flipIn,
                                    OperatingBuffer& flipOut,
                                    OperatingBuffer& obOutBuf)
{
    childNodes[0]->SetInputBuffer(state);
    childNodes[0]->obOut = flipOut;
    childNodes[0]->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut = obOutBuf;
    childNodes[1]->TraverseTreeAssignBuffersLogicA(state, flipOut, flipIn, obOutBuf);

    obIn  = childNodes[0]->obIn;
    obOut = childNodes[1]->obOut;
}

void TreeNode::assign_buffers_CS_3D_TRTRTR(TraverseState&   state,
                                           OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    assert(scheme == CS_3D_TRTRTR);
    assert(childNodes.size() == 6);

    for(int i = 0; i < 6; i += 2)
    {
        auto& trans_plan = childNodes[i];

        // T
        trans_plan->SetInputBuffer(state);
        trans_plan->obOut        = OB_TEMP;
        trans_plan->inArrayType  = (i == 0) ? inArrayType : childNodes[i - 1]->outArrayType;
        trans_plan->outArrayType = rocfft_array_type_complex_interleaved;

        auto& row_plan = childNodes[i + 1];
        row_plan->SetInputBuffer(state);
        row_plan->obOut        = obOutBuf;
        row_plan->inArrayType  = rocfft_array_type_complex_interleaved;
        row_plan->outArrayType = outArrayType;
        row_plan->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
    }

    obOut = childNodes[childNodes.size() - 1]->obOut;
}

void TreeNode::assign_buffers_CS_3D_BLOCK_RC(TraverseState&   state,
                                             OperatingBuffer& flipIn,
                                             OperatingBuffer& flipOut,
                                             OperatingBuffer& obOutBuf)
{
    assert(scheme == CS_3D_BLOCK_RC);

    for(size_t i = 0; i < childNodes.size(); ++i)
    {
        auto& node = childNodes[i];
        node->SetInputBuffer(state);
        node->inArrayType = (i == 0) ? inArrayType : childNodes[i - 1]->outArrayType;
        node->obOut       = flipOut == OB_USER_OUT && placement == rocfft_placement_notinplace
                                ? OB_USER_IN
                                : flipOut;

        // temp is interleaved, in/out might not be
        switch(node->obOut)
        {
        case OB_USER_IN:
            node->outArrayType = inArrayType;
            break;
        case OB_USER_OUT:
            node->outArrayType = outArrayType;
            break;
        default:
            node->outArrayType = rocfft_array_type_complex_interleaved;
        }

        node->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);
    }

    obOut                           = obOutBuf;
    childNodes.back()->obOut        = obOut;
    childNodes.back()->outArrayType = outArrayType;
}

///////////////////////////////////////////////////////////////////////////////
/// Set placement variable and in/out array types, if not already set.
void TreeNode::TraverseTreeAssignPlacementsLogicA(const rocfft_array_type rootIn,
                                                  const rocfft_array_type rootOut)
{
    if(parent != nullptr)
    {
        placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;
        // if (this->scheme == CS_KERNEL_TRANSPOSE)
        // {
        //     rocfft_cout << " obIn " << obIn << ", obOut " << obOut << " rootIn " << rootIn
        //               << ", rootOut " << rootOut << " inArrayType " << inArrayType
        //               << ", outArrayType " << outArrayType << std::endl;
        // }

        if(inArrayType == rocfft_array_type_unset)
        {
            switch(obIn)
            {
            case OB_USER_IN:
                // NB:
                // There are some cases that 2D/3D even length r2c with
                // child node ***BLOCK_CC. The child node can not detect
                // the correct array type from its direct parent, which
                // has to get the info from root node.
                // On the other hand, some cases that 1D even length r2c
                // with children should use the array type from the parent
                // instead of root node.
                inArrayType = (rootIn == rocfft_array_type_complex_planar
                               || rootIn == rocfft_array_type_hermitian_planar)
                                  ? rootIn
                                  : parent->inArrayType;
                break;
            case OB_USER_OUT:
                inArrayType = (rootOut == rocfft_array_type_complex_planar
                               || rootOut == rocfft_array_type_hermitian_planar)
                                  ? rootOut
                                  : parent->outArrayType;
                break;
            case OB_TEMP:
                inArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_CMPLX_FOR_REAL:
                inArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_BLUESTEIN:
                inArrayType = rocfft_array_type_complex_interleaved;
                if(parent->iOffset != 0)
                    iOffset = parent->iOffset;
                break;
            default:
                inArrayType = rocfft_array_type_complex_interleaved;
            }
        }

        if(outArrayType == rocfft_array_type_unset)
        {
            switch(obOut)
            {
            case OB_USER_IN:
                outArrayType = (rootIn == rocfft_array_type_complex_planar
                                || rootIn == rocfft_array_type_hermitian_planar)
                                   ? rootIn
                                   : parent->inArrayType;
                break;
            case OB_USER_OUT:
                outArrayType = (rootOut == rocfft_array_type_complex_planar
                                || rootOut == rocfft_array_type_hermitian_planar)
                                   ? rootOut
                                   : parent->outArrayType;
                break;
            case OB_TEMP:
                outArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_CMPLX_FOR_REAL:
                outArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_BLUESTEIN:
                outArrayType = rocfft_array_type_complex_interleaved;
                if(parent->oOffset != 0)
                    oOffset = parent->oOffset;
                break;
            default:
                outArrayType = rocfft_array_type_complex_interleaved;
            }
        }
    }

    for(auto children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
    {
        (*children_p)->TraverseTreeAssignPlacementsLogicA(rootIn, rootOut);
    }
}

// Set strides and distances

void TreeNode::TraverseTreeAssignParamsLogicA()
{
#if 0
    // Debug output information
    auto        here = this;
    auto        up   = parent;
    std::string tabs;
    while(up != nullptr && here != up)
    {
        here = up;
        up   = parent->parent;
        tabs += "\t";
    }
    rocfft_cout << tabs << "TraverseTreeAssignParamsLogicA: " << PrintScheme(scheme) << std::endl;
    rocfft_cout << tabs << "\tlength:";
    for(auto i : length)
        rocfft_cout << i << " ";
    rocfft_cout << std::endl;
    rocfft_cout << tabs << "\tistride:";
    for(auto i : inStride)
        rocfft_cout << i << " ";
    rocfft_cout << std::endl;
    rocfft_cout << tabs << "\tostride:";
    for(auto i : outStride)
        rocfft_cout << i << " ";
    rocfft_cout << std::endl;
#endif

    assert(length.size() == inStride.size());
    assert(length.size() == outStride.size());

    switch(scheme)
    {
    case CS_REAL_TRANSFORM_USING_CMPLX:
        assign_params_CS_REAL_TRANSFORM_USING_CMPLX();
        break;
    case CS_REAL_TRANSFORM_EVEN:
        assign_params_CS_REAL_TRANSFORM_EVEN();
        break;
    case CS_REAL_2D_EVEN:
        assign_params_CS_REAL_2D_EVEN();
        break;
    case CS_REAL_3D_EVEN:
        assign_params_CS_REAL_3D_EVEN();
        break;
    case CS_BLUESTEIN:
        assign_params_CS_BLUESTEIN();
        break;
    case CS_L1D_TRTRT:
        assign_params_CS_L1D_TRTRT();
        break;
    case CS_L1D_CC:
        assign_params_CS_L1D_CC();
        break;
    case CS_L1D_CRT:
        assign_params_CS_L1D_CRT();
        break;
    case CS_2D_RTRT:
        assign_params_CS_2D_RTRT();
        break;
    case CS_2D_RC:
    case CS_2D_STRAIGHT:
        assign_params_CS_2D_RC_STRAIGHT();
        break;
    case CS_3D_RTRT:
        assign_params_CS_3D_RTRT();
        break;
    case CS_3D_TRTRTR:
        assign_params_CS_3D_TRTRTR();
        break;
    case CS_3D_BLOCK_RC:
        assign_params_CS_3D_BLOCK_RC();
        break;
    case CS_3D_RC:
    case CS_3D_STRAIGHT:
        assign_params_CS_3D_RC_STRAIGHT();
        break;
    default:
        return;
    }
}

void TreeNode::assign_params_CS_REAL_TRANSFORM_USING_CMPLX()
{
    assert(childNodes.size() == 3);
    auto& copyHeadPlan = childNodes[0];
    auto& fftPlan      = childNodes[1];
    auto& copyTailPlan = childNodes[2];

    copyHeadPlan->inStride = inStride;
    copyHeadPlan->iDist    = iDist;

    copyHeadPlan->outStride.push_back(1);
    copyHeadPlan->oDist = copyHeadPlan->length[0];
    for(size_t index = 1; index < length.size(); index++)
    {
        copyHeadPlan->outStride.push_back(copyHeadPlan->oDist);
        copyHeadPlan->oDist *= length[index];
    }

    fftPlan->inStride  = copyHeadPlan->outStride;
    fftPlan->iDist     = copyHeadPlan->oDist;
    fftPlan->outStride = fftPlan->inStride;
    fftPlan->oDist     = fftPlan->iDist;

    fftPlan->TraverseTreeAssignParamsLogicA();

    copyTailPlan->inStride = fftPlan->outStride;
    copyTailPlan->iDist    = fftPlan->oDist;

    copyTailPlan->outStride = outStride;
    copyTailPlan->oDist     = oDist;
}

void TreeNode::assign_params_CS_REAL_TRANSFORM_EVEN()
{
    // definitely will have FFT + apply callback.  pre/post processing
    // might be fused into the FFT or separate.
    assert(childNodes.size() == 2 || childNodes.size() == 3);

    if(direction == -1)
    {
        // forward transform, r2c

        // iDist is in reals, subplan->iDist is in complexes
        auto& applyCallback      = childNodes[0];
        applyCallback->inStride  = inStride;
        applyCallback->iDist     = iDist;
        applyCallback->outStride = inStride;
        applyCallback->oDist     = iDist;

        auto& fftPlan     = childNodes[1];
        fftPlan->inStride = inStride;
        for(int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            fftPlan->inStride[i] /= 2;
        }
        fftPlan->iDist     = iDist / 2;
        fftPlan->outStride = inStride;
        for(int i = 1; i < fftPlan->outStride.size(); ++i)
        {
            fftPlan->outStride[i] /= 2;
        }
        fftPlan->oDist = iDist / 2;
        fftPlan->TraverseTreeAssignParamsLogicA();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        if(childNodes.size() == 3)
        {
            auto& postPlan = childNodes[2];
            assert(postPlan->scheme == CS_KERNEL_R_TO_CMPLX
                   || postPlan->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE);
            postPlan->inStride = inStride;
            for(int i = 1; i < postPlan->inStride.size(); ++i)
            {
                postPlan->inStride[i] /= 2;
            }
            postPlan->iDist     = iDist / 2;
            postPlan->outStride = outStride;
            postPlan->oDist     = oDist;

            assert(postPlan->length.size() == postPlan->inStride.size());
            assert(postPlan->length.size() == postPlan->outStride.size());
        }
        else
        {
            // we fused post-proc into the FFT kernel, so give the correct out strides
            fftPlan->outStride = outStride;
            fftPlan->oDist     = oDist;
        }
    }
    else
    {

        // backward transform, c2r
        bool fusedPreProcessing = childNodes[0]->ebtype == EmbeddedType::C2Real_PRE;

        // oDist is in reals, subplan->oDist is in complexes

        if(!fusedPreProcessing)
        {
            auto& prePlan = childNodes[0];
            assert(prePlan->scheme == CS_KERNEL_CMPLX_TO_R);

            prePlan->iDist = iDist;
            prePlan->oDist = oDist / 2;

            // Strides are actually distances for multimensional transforms.
            // Only the first value is used, but we require dimension values.
            prePlan->inStride  = inStride;
            prePlan->outStride = outStride;
            // Strides are in complex types
            for(int i = 1; i < prePlan->outStride.size(); ++i)
            {
                prePlan->outStride[i] /= 2;
            }
            assert(prePlan->length.size() == prePlan->inStride.size());
            assert(prePlan->length.size() == prePlan->outStride.size());
        }

        auto& fftPlan = fusedPreProcessing ? childNodes[0] : childNodes[1];
        // Transform the strides from real to complex.

        fftPlan->inStride  = fusedPreProcessing ? inStride : outStride;
        fftPlan->iDist     = fusedPreProcessing ? iDist : oDist / 2;
        fftPlan->outStride = outStride;
        fftPlan->oDist     = oDist / 2;
        // The strides must be translated from real to complex.
        for(int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            if(!fusedPreProcessing)
                fftPlan->inStride[i] /= 2;
            fftPlan->outStride[i] /= 2;
        }

        fftPlan->TraverseTreeAssignParamsLogicA();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        // we apply callbacks on the root plan's output
        TreeNode* rootPlan = this;
        while(rootPlan->parent != nullptr)
            rootPlan = rootPlan->parent;

        auto& applyCallback      = childNodes.back();
        applyCallback->inStride  = rootPlan->outStride;
        applyCallback->iDist     = rootPlan->oDist;
        applyCallback->outStride = rootPlan->outStride;
        applyCallback->oDist     = rootPlan->oDist;
    }
}

void TreeNode::assign_params_CS_L1D_CC()
{
    auto& col2colPlan = childNodes[0];
    auto& row2colPlan = childNodes[1];

    if((obOut == OB_USER_OUT) || (obOut == OB_TEMP_CMPLX_FOR_REAL))
    {
        // B -> T
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        col2colPlan->outStride.push_back(col2colPlan->length[1]);
        col2colPlan->outStride.push_back(1);
        col2colPlan->oDist = length[0];

        for(size_t index = 1; index < length.size(); index++)
        {
            col2colPlan->inStride.push_back(inStride[index]);
            col2colPlan->outStride.push_back(col2colPlan->oDist);
            col2colPlan->oDist *= length[index];
        }

        // T -> B
        row2colPlan->inStride.push_back(1);
        row2colPlan->inStride.push_back(row2colPlan->length[0]);
        row2colPlan->iDist = length[0];

        row2colPlan->outStride.push_back(outStride[0] * row2colPlan->length[1]);
        row2colPlan->outStride.push_back(outStride[0]);
        row2colPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
        {
            row2colPlan->inStride.push_back(row2colPlan->iDist);
            row2colPlan->iDist *= length[index];
            row2colPlan->outStride.push_back(outStride[index]);
        }
    }
    else
    {
        // here we don't have B info right away, we get it through its parent

        // TODO: what is this assert for?
        assert(parent->obOut == OB_USER_IN || parent->obOut == OB_USER_OUT
               || parent->obOut == OB_TEMP_CMPLX_FOR_REAL
               || parent->scheme == CS_REAL_TRANSFORM_EVEN);

        // T-> B
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        for(size_t index = 1; index < length.size(); index++)
            col2colPlan->inStride.push_back(inStride[index]);

        if(parent->scheme == CS_L1D_TRTRT)
        {
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]);
            col2colPlan->outStride.push_back(parent->outStride[0]);
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]
                                             * col2colPlan->length[0]);
            col2colPlan->oDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                col2colPlan->outStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            assert(parent->outStride[0] == 1);

            col2colPlan->outStride.push_back(col2colPlan->length[1]);
            col2colPlan->outStride.push_back(1);
            col2colPlan->oDist = col2colPlan->length[1] * col2colPlan->length[0];

            for(size_t index = 1; index < length.size(); index++)
            {
                col2colPlan->outStride.push_back(col2colPlan->oDist);
                col2colPlan->oDist *= length[index];
            }
        }

        // B -> T
        if(parent->scheme == CS_L1D_TRTRT)
        {
            row2colPlan->inStride.push_back(parent->outStride[0]);
            row2colPlan->inStride.push_back(parent->outStride[0] * row2colPlan->length[0]);
            row2colPlan->inStride.push_back(parent->outStride[0] * row2colPlan->length[0]
                                            * row2colPlan->length[1]);
            row2colPlan->iDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                row2colPlan->inStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            row2colPlan->inStride.push_back(1);
            row2colPlan->inStride.push_back(row2colPlan->length[0]);
            row2colPlan->iDist = row2colPlan->length[0] * row2colPlan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                row2colPlan->inStride.push_back(row2colPlan->iDist);
                row2colPlan->iDist *= length[index];
            }
        }

        row2colPlan->outStride.push_back(outStride[0] * row2colPlan->length[1]);
        row2colPlan->outStride.push_back(outStride[0]);
        row2colPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            row2colPlan->outStride.push_back(outStride[index]);
    }
}

void TreeNode::assign_params_CS_L1D_CRT()
{
    auto& col2colPlan = childNodes[0];
    auto& row2rowPlan = childNodes[1];
    auto& transPlan   = childNodes[2];

    if((obOut == OB_USER_OUT) || (obOut == OB_TEMP_CMPLX_FOR_REAL))
    {
        // B -> T
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        col2colPlan->outStride.push_back(col2colPlan->length[1]);
        col2colPlan->outStride.push_back(1);
        col2colPlan->oDist = length[0];

        for(size_t index = 1; index < length.size(); index++)
        {
            col2colPlan->inStride.push_back(inStride[index]);
            col2colPlan->outStride.push_back(col2colPlan->oDist);
            col2colPlan->oDist *= length[index];
        }

        // T -> T
        row2rowPlan->inStride.push_back(1);
        row2rowPlan->inStride.push_back(row2rowPlan->length[0]);
        row2rowPlan->iDist = length[0];

        for(size_t index = 1; index < length.size(); index++)
        {
            row2rowPlan->inStride.push_back(row2rowPlan->iDist);
            row2rowPlan->iDist *= length[index];
        }

        row2rowPlan->outStride = row2rowPlan->inStride;
        row2rowPlan->oDist     = row2rowPlan->iDist;

        // T -> B
        transPlan->inStride = row2rowPlan->outStride;
        transPlan->iDist    = row2rowPlan->oDist;

        transPlan->outStride.push_back(outStride[0]);
        transPlan->outStride.push_back(outStride[0] * (transPlan->length[1]));
        transPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            transPlan->outStride.push_back(outStride[index]);
    }
    else
    {
        // T -> B
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        for(size_t index = 1; index < length.size(); index++)
            col2colPlan->inStride.push_back(inStride[index]);

        if(parent->scheme == CS_L1D_TRTRT)
        {
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]);
            col2colPlan->outStride.push_back(parent->outStride[0]);
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]
                                             * col2colPlan->length[0]);
            col2colPlan->oDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                col2colPlan->outStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            assert(parent->outStride[0] == 1);
            for(size_t index = 1; index < parent->length.size(); index++)
                assert(parent->outStride[index]
                       == (parent->outStride[index - 1] * parent->length[index - 1]));

            col2colPlan->outStride.push_back(col2colPlan->length[1]);
            col2colPlan->outStride.push_back(1);
            col2colPlan->oDist = col2colPlan->length[1] * col2colPlan->length[0];

            for(size_t index = 1; index < length.size(); index++)
            {
                col2colPlan->outStride.push_back(col2colPlan->oDist);
                col2colPlan->oDist *= length[index];
            }
        }

        // B -> B
        if(parent->scheme == CS_L1D_TRTRT)
        {
            row2rowPlan->inStride.push_back(parent->outStride[0]);
            row2rowPlan->inStride.push_back(parent->outStride[0] * row2rowPlan->length[0]);
            row2rowPlan->inStride.push_back(parent->outStride[0] * row2rowPlan->length[0]
                                            * row2rowPlan->length[1]);
            row2rowPlan->iDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                row2rowPlan->inStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            row2rowPlan->inStride.push_back(1);
            row2rowPlan->inStride.push_back(row2rowPlan->length[0]);
            row2rowPlan->iDist = row2rowPlan->length[0] * row2rowPlan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                row2rowPlan->inStride.push_back(row2rowPlan->iDist);
                row2rowPlan->iDist *= length[index];
            }
        }

        row2rowPlan->outStride = row2rowPlan->inStride;
        row2rowPlan->oDist     = row2rowPlan->iDist;

        // B -> T
        transPlan->inStride = row2rowPlan->outStride;
        transPlan->iDist    = row2rowPlan->oDist;

        transPlan->outStride.push_back(outStride[0]);
        transPlan->outStride.push_back(outStride[0] * transPlan->length[1]);
        transPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            transPlan->outStride.push_back(outStride[index]);
    }
}

void TreeNode::assign_params_CS_BLUESTEIN()
{
    auto& chirpPlan  = childNodes[0];
    auto& padmulPlan = childNodes[1];
    auto& fftiPlan   = childNodes[2];
    auto& fftcPlan   = childNodes[3];
    auto& fftmulPlan = childNodes[4];
    auto& fftrPlan   = childNodes[5];
    auto& resmulPlan = childNodes[6];

    chirpPlan->inStride.push_back(1);
    chirpPlan->iDist = chirpPlan->lengthBlue;
    chirpPlan->outStride.push_back(1);
    chirpPlan->oDist = chirpPlan->lengthBlue;

    padmulPlan->inStride = inStride;
    padmulPlan->iDist    = iDist;

    padmulPlan->outStride.push_back(1);
    padmulPlan->oDist = padmulPlan->lengthBlue;
    for(size_t index = 1; index < length.size(); index++)
    {
        padmulPlan->outStride.push_back(padmulPlan->oDist);
        padmulPlan->oDist *= length[index];
    }

    fftiPlan->inStride  = padmulPlan->outStride;
    fftiPlan->iDist     = padmulPlan->oDist;
    fftiPlan->outStride = fftiPlan->inStride;
    fftiPlan->oDist     = fftiPlan->iDist;

    fftiPlan->TraverseTreeAssignParamsLogicA();

    fftcPlan->inStride  = chirpPlan->outStride;
    fftcPlan->iDist     = chirpPlan->oDist;
    fftcPlan->outStride = fftcPlan->inStride;
    fftcPlan->oDist     = fftcPlan->iDist;

    fftcPlan->TraverseTreeAssignParamsLogicA();

    fftmulPlan->inStride  = fftiPlan->outStride;
    fftmulPlan->iDist     = fftiPlan->oDist;
    fftmulPlan->outStride = fftmulPlan->inStride;
    fftmulPlan->oDist     = fftmulPlan->iDist;

    fftrPlan->inStride  = fftmulPlan->outStride;
    fftrPlan->iDist     = fftmulPlan->oDist;
    fftrPlan->outStride = fftrPlan->inStride;
    fftrPlan->oDist     = fftrPlan->iDist;

    fftrPlan->TraverseTreeAssignParamsLogicA();

    resmulPlan->inStride  = fftrPlan->outStride;
    resmulPlan->iDist     = fftrPlan->oDist;
    resmulPlan->outStride = outStride;
    resmulPlan->oDist     = oDist;
}

void TreeNode::assign_params_CS_L1D_TRTRT()
{
    const size_t biggerDim  = std::max(childNodes[0]->length[0], childNodes[0]->length[1]);
    const size_t smallerDim = std::min(childNodes[0]->length[0], childNodes[0]->length[1]);
    const size_t padding
        = ((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512) ? 64 : 0;

    auto& trans1Plan = childNodes[0];
    auto& row1Plan   = childNodes[1];
    auto& trans2Plan = childNodes[2];
    auto& row2Plan   = childNodes[3];
    auto& trans3Plan = childNodes[4];

    trans1Plan->inStride.push_back(inStride[0]);
    trans1Plan->inStride.push_back(trans1Plan->length[0] * inStride[0]);
    trans1Plan->iDist = iDist;
    for(size_t index = 1; index < length.size(); index++)
        trans1Plan->inStride.push_back(inStride[index]);

    if(trans1Plan->obOut == OB_TEMP)
    {
        trans1Plan->outStride.push_back(1);
        trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
        trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];

        for(size_t index = 1; index < length.size(); index++)
        {
            trans1Plan->outStride.push_back(trans1Plan->oDist);
            trans1Plan->oDist *= length[index];
        }
    }
    else
    {
        if(parent->scheme == CS_L1D_TRTRT)
        {
            trans1Plan->outStride.push_back(outStride[0]);
            trans1Plan->outStride.push_back(outStride[0] * (trans1Plan->length[1]));
            trans1Plan->oDist = oDist;

            for(size_t index = 1; index < length.size(); index++)
                trans1Plan->outStride.push_back(outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            assert((parent->obOut == OB_USER_OUT) || (parent->obOut == OB_TEMP_CMPLX_FOR_REAL));

            assert(parent->outStride[0] == 1);
            // CS_REAL_2D_EVEN pads the lengths/strides, and mixes
            // counts between reals and complexes, so the math for
            // the assert below doesn't work out
            if(parent->scheme != CS_REAL_2D_EVEN)
            {
                for(size_t index = 1; index < parent->length.size(); index++)
                    assert(parent->outStride[index]
                           == (parent->outStride[index - 1] * parent->length[index - 1]));
            }

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1]);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                trans1Plan->outStride.push_back(trans1Plan->oDist);
                trans1Plan->oDist *= length[index];
            }
        }
    }

    row1Plan->inStride = trans1Plan->outStride;
    row1Plan->iDist    = trans1Plan->oDist;

    if(row1Plan->placement == rocfft_placement_inplace)
    {
        row1Plan->outStride = row1Plan->inStride;
        row1Plan->oDist     = row1Plan->iDist;
    }
    else
    {
        row1Plan->outStride.push_back(outStride[0]);
        row1Plan->outStride.push_back(outStride[0] * row1Plan->length[0]);
        row1Plan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            row1Plan->outStride.push_back(outStride[index]);
    }

    row1Plan->TraverseTreeAssignParamsLogicA();

    trans2Plan->inStride = row1Plan->outStride;
    trans2Plan->iDist    = row1Plan->oDist;

    if(trans2Plan->obOut == OB_TEMP)
    {
        trans2Plan->outStride.push_back(1);
        trans2Plan->outStride.push_back(trans2Plan->length[1] + padding);
        trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->outStride[1];

        for(size_t index = 1; index < length.size(); index++)
        {
            trans2Plan->outStride.push_back(trans2Plan->oDist);
            trans2Plan->oDist *= length[index];
        }
    }
    else
    {
        if((parent == NULL) || (parent && (parent->scheme == CS_L1D_TRTRT)))
        {
            trans2Plan->outStride.push_back(outStride[0]);
            trans2Plan->outStride.push_back(outStride[0] * (trans2Plan->length[1]));
            trans2Plan->oDist = oDist;

            for(size_t index = 1; index < length.size(); index++)
                trans2Plan->outStride.push_back(outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            trans2Plan->outStride.push_back(1);
            trans2Plan->outStride.push_back(trans2Plan->length[1]);
            trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                trans2Plan->outStride.push_back(trans2Plan->oDist);
                trans2Plan->oDist *= length[index];
            }
        }
    }

    row2Plan->inStride = trans2Plan->outStride;
    row2Plan->iDist    = trans2Plan->oDist;

    if(row2Plan->obIn == row2Plan->obOut)
    {
        row2Plan->outStride = row2Plan->inStride;
        row2Plan->oDist     = row2Plan->iDist;
    }
    else if(row2Plan->obOut == OB_TEMP)
    {
        row2Plan->outStride.push_back(1);
        row2Plan->outStride.push_back(row2Plan->length[0] + padding);
        row2Plan->oDist = row2Plan->length[1] * row2Plan->outStride[1];

        for(size_t index = 1; index < length.size(); index++)
        {
            row2Plan->outStride.push_back(row2Plan->oDist);
            row2Plan->oDist *= length[index];
        }
    }
    else
    {
        if((parent == NULL) || (parent && (parent->scheme == CS_L1D_TRTRT)))
        {
            row2Plan->outStride.push_back(outStride[0]);
            row2Plan->outStride.push_back(outStride[0] * (row2Plan->length[0]));
            row2Plan->oDist = oDist;

            for(size_t index = 1; index < length.size(); index++)
                row2Plan->outStride.push_back(outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            row2Plan->outStride.push_back(1);
            row2Plan->outStride.push_back(row2Plan->length[0]);
            row2Plan->oDist = row2Plan->length[0] * row2Plan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                row2Plan->outStride.push_back(row2Plan->oDist);
                row2Plan->oDist *= length[index];
            }
        }
    }

    trans3Plan->inStride = row2Plan->outStride;
    trans3Plan->iDist    = row2Plan->oDist;

    trans3Plan->outStride.push_back(outStride[0]);
    trans3Plan->outStride.push_back(outStride[0] * (trans3Plan->length[1]));
    trans3Plan->oDist = oDist;

    for(size_t index = 1; index < length.size(); index++)
        trans3Plan->outStride.push_back(outStride[index]);
}

void TreeNode::assign_params_CS_2D_RTRT()
{
    const size_t biggerDim  = std::max(length[0], length[1]);
    const size_t smallerDim = std::min(length[0], length[1]);
    const size_t padding
        = ((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512) ? 64 : 0;

    auto& row1Plan      = childNodes[0];
    row1Plan->inStride  = inStride;
    row1Plan->iDist     = iDist;
    row1Plan->outStride = outStride;
    row1Plan->oDist     = oDist;
    row1Plan->TraverseTreeAssignParamsLogicA();

    auto& trans1Plan     = childNodes[1];
    trans1Plan->inStride = row1Plan->outStride;
    trans1Plan->iDist    = row1Plan->oDist;
    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
    trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    auto& row2Plan      = childNodes[2];
    row2Plan->inStride  = trans1Plan->outStride;
    row2Plan->iDist     = trans1Plan->oDist;
    row2Plan->outStride = row2Plan->inStride;
    row2Plan->oDist     = row2Plan->iDist;
    row2Plan->TraverseTreeAssignParamsLogicA();

    auto& trans2Plan      = childNodes[3];
    trans2Plan->inStride  = row2Plan->outStride;
    trans2Plan->iDist     = row2Plan->oDist;
    trans2Plan->outStride = outStride;
    trans2Plan->oDist     = oDist;
}

void TreeNode::assign_params_CS_REAL_2D_EVEN()
{
    const size_t biggerDim  = std::max(length[0], length[1]);
    const size_t smallerDim = std::min(length[0], length[1]);
    const size_t padding
        = (((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512)) ? 64 : 0;

    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& row1Plan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            row1Plan->inStride = inStride;
            row1Plan->iDist    = iDist;

            row1Plan->outStride = outStride;
            row1Plan->oDist     = oDist;

            row1Plan->TraverseTreeAssignParamsLogicA();
        }

        auto& trans1Plan = childNodes[1];
        {
            // B -> T
            trans1Plan->inStride = row1Plan->outStride;
            trans1Plan->iDist    = row1Plan->oDist;

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
        }

        auto& row2Plan = childNodes[2];
        {
            // T -> T
            row2Plan->inStride = trans1Plan->outStride;
            row2Plan->iDist    = trans1Plan->oDist;

            row2Plan->outStride = row2Plan->inStride;
            row2Plan->oDist     = row2Plan->iDist;

            row2Plan->TraverseTreeAssignParamsLogicA();
        }

        auto& trans2Plan = childNodes[3];
        {
            // T -> B
            trans2Plan->inStride = row2Plan->outStride;
            trans2Plan->iDist    = row2Plan->oDist;

            trans2Plan->outStride = outStride;
            trans2Plan->oDist     = oDist;
        }
    }
    else
    {
        auto& trans1Plan = childNodes[0];
        {
            trans1Plan->inStride = inStride;
            trans1Plan->iDist    = iDist;

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
        }
        auto& c2cPlan = childNodes[1];
        {
            c2cPlan->inStride = trans1Plan->outStride;
            c2cPlan->iDist    = trans1Plan->oDist;

            c2cPlan->outStride = c2cPlan->inStride;
            c2cPlan->oDist     = c2cPlan->iDist;

            c2cPlan->TraverseTreeAssignParamsLogicA();
        }
        auto& trans2Plan = childNodes[2];
        {
            trans2Plan->inStride = trans1Plan->outStride;
            trans2Plan->iDist    = trans1Plan->oDist;

            trans2Plan->outStride = trans1Plan->inStride;
            trans2Plan->oDist     = trans2Plan->length[0] * trans2Plan->outStride[1];
        }
        auto& c2rPlan = childNodes[3];
        {
            c2rPlan->inStride = trans2Plan->outStride;
            c2rPlan->iDist    = trans2Plan->oDist;

            c2rPlan->outStride = outStride;
            c2rPlan->oDist     = oDist;

            c2rPlan->TraverseTreeAssignParamsLogicA();
        }
    }
}

void TreeNode::assign_params_CS_REAL_3D_EVEN()
{
    // TODO: add padding?

    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& rcplan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            rcplan->inStride  = inStride;
            rcplan->iDist     = iDist;
            rcplan->outStride = outStride;
            rcplan->oDist     = oDist;
            rcplan->dimension = 1;
            rcplan->TraverseTreeAssignParamsLogicA();
        }

        // in-place SBCC for higher dims
        if(childNodes.size() == 3)
        {
            auto& sbccZ     = childNodes[1];
            sbccZ->inStride = outStride;
            // SBCC along Z dim
            std::swap(sbccZ->inStride[1], sbccZ->inStride[2]);
            std::swap(sbccZ->inStride[0], sbccZ->inStride[1]);
            sbccZ->iDist     = oDist;
            sbccZ->outStride = sbccZ->inStride;
            sbccZ->oDist     = oDist;
            sbccZ->TraverseTreeAssignParamsLogicA();

            auto& sbccY     = childNodes[2];
            sbccY->inStride = outStride;
            // SBCC along Y dim
            std::swap(sbccY->inStride[0], sbccY->inStride[1]);
            sbccY->iDist     = oDist;
            sbccY->outStride = sbccY->inStride;
            sbccY->oDist     = oDist;
            sbccY->TraverseTreeAssignParamsLogicA();
        }
        // TRTRTR
        else
        {
            auto& trans1 = childNodes[1];
            {
                trans1->inStride = rcplan->outStride;
                trans1->iDist    = rcplan->oDist;
                trans1->outStride.push_back(1);
                trans1->outStride.push_back(trans1->length[1]);
                trans1->outStride.push_back(trans1->length[2] * trans1->outStride[1]);
                trans1->oDist = trans1->iDist;
            }

            auto& c1plan = childNodes[2];
            {
                c1plan->inStride  = trans1->outStride;
                c1plan->iDist     = trans1->oDist;
                c1plan->outStride = c1plan->inStride;
                c1plan->oDist     = c1plan->iDist;
                c1plan->dimension = 1;
                c1plan->TraverseTreeAssignParamsLogicA();
            }

            auto& trans2 = childNodes[3];
            {
                trans2->inStride = c1plan->outStride;
                trans2->iDist    = c1plan->oDist;
                trans2->outStride.push_back(1);
                trans2->outStride.push_back(trans2->length[1]);
                trans2->outStride.push_back(trans2->length[2] * trans2->outStride[1]);
                trans2->oDist = trans2->iDist;
            }

            auto& c2plan = childNodes[4];
            {
                c2plan->inStride  = trans2->outStride;
                c2plan->iDist     = trans2->oDist;
                c2plan->outStride = c2plan->inStride;
                c2plan->oDist     = c2plan->iDist;
                c2plan->dimension = 1;
                c2plan->TraverseTreeAssignParamsLogicA();
            }

            auto& trans3 = childNodes[5];
            {
                trans3->inStride  = c2plan->outStride;
                trans3->iDist     = c2plan->oDist;
                trans3->outStride = outStride;
                trans3->oDist     = oDist;
            }
        }
    }
    else
    {
        // input strides for last c2r node
        std::vector<size_t> c2r_inStride = inStride;
        size_t              c2r_iDist    = iDist;

        // in-place SBCC for higher dimensions
        if(childNodes.size() == 3)
        {
            auto& sbccZ     = childNodes[0];
            sbccZ->inStride = inStride;
            // SBCC along Z dim
            std::swap(sbccZ->inStride[1], sbccZ->inStride[2]);
            std::swap(sbccZ->inStride[0], sbccZ->inStride[1]);
            sbccZ->iDist     = iDist;
            sbccZ->outStride = sbccZ->inStride;
            sbccZ->oDist     = iDist;
            sbccZ->TraverseTreeAssignParamsLogicA();

            auto& sbccY     = childNodes[1];
            sbccY->inStride = inStride;
            // SBCC along Y dim
            std::swap(sbccY->inStride[0], sbccY->inStride[1]);
            sbccY->iDist     = iDist;
            sbccY->outStride = sbccY->inStride;
            sbccY->oDist     = iDist;
            sbccY->TraverseTreeAssignParamsLogicA();
        }
        // RTRTRT
        else
        {
            {
                auto& trans3     = childNodes[0];
                trans3->inStride = inStride;
                trans3->iDist    = iDist;
                trans3->outStride.push_back(1);
                trans3->outStride.push_back(trans3->outStride[0] * trans3->length[2]);
                trans3->outStride.push_back(trans3->outStride[1] * trans3->length[0]);
                trans3->oDist = trans3->iDist;
            }

            {
                auto& ccplan      = childNodes[1];
                ccplan->inStride  = childNodes[0]->outStride;
                ccplan->iDist     = childNodes[0]->oDist;
                ccplan->outStride = ccplan->inStride;
                ccplan->oDist     = ccplan->iDist;
                ccplan->dimension = 1;
                ccplan->TraverseTreeAssignParamsLogicA();
            }

            {
                auto& trans2     = childNodes[2];
                trans2->inStride = childNodes[1]->outStride;
                trans2->iDist    = childNodes[1]->oDist;
                trans2->outStride.push_back(1);
                trans2->outStride.push_back(trans2->outStride[0] * trans2->length[2]);
                trans2->outStride.push_back(trans2->outStride[1] * trans2->length[0]);
                trans2->oDist = trans2->iDist;
            }

            {
                auto& ccplan      = childNodes[3];
                ccplan->inStride  = childNodes[2]->outStride;
                ccplan->iDist     = childNodes[2]->oDist;
                ccplan->outStride = ccplan->inStride;
                ccplan->oDist     = ccplan->iDist;
                ccplan->dimension = 1;
                ccplan->TraverseTreeAssignParamsLogicA();
            }

            {
                auto& trans1     = childNodes[4];
                trans1->inStride = childNodes[3]->outStride;
                trans1->iDist    = childNodes[3]->oDist;
                trans1->outStride.push_back(1);
                trans1->outStride.push_back(trans1->outStride[0] * trans1->length[2]);
                trans1->outStride.push_back(trans1->outStride[1] * trans1->length[0]);
                trans1->oDist = trans1->iDist;
                c2r_inStride  = trans1->outStride;
                c2r_iDist     = trans1->oDist;
            }
        }

        auto& crplan = childNodes.back();
        {
            crplan->inStride  = c2r_inStride;
            crplan->iDist     = c2r_iDist;
            crplan->outStride = outStride;
            crplan->oDist     = oDist;
            crplan->dimension = 1;
            crplan->TraverseTreeAssignParamsLogicA();
        }
    }
}

void TreeNode::assign_params_CS_2D_RC_STRAIGHT()
{
    auto& rowPlan = childNodes[0];
    auto& colPlan = childNodes[1];

    // B -> B
    // assert((rowPlan->obOut == OB_USER_OUT) || (rowPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
    //        || (rowPlan->obOut == OB_TEMP_BLUESTEIN));
    rowPlan->inStride = inStride;
    rowPlan->iDist    = iDist;

    // row plan is in-place, so keep same strides in case parent's
    // in/out strides are incompatible for the same buffer
    rowPlan->outStride = inStride;
    rowPlan->oDist     = iDist;

    rowPlan->TraverseTreeAssignParamsLogicA();

    // B -> B
    assert((colPlan->obOut == OB_USER_OUT) || (colPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (colPlan->obOut == OB_TEMP_BLUESTEIN));
    colPlan->inStride = rowPlan->outStride;
    std::swap(colPlan->inStride[0], colPlan->inStride[1]);

    colPlan->iDist = rowPlan->oDist;

    colPlan->outStride = outStride;
    std::swap(colPlan->outStride[0], colPlan->outStride[1]);
    colPlan->oDist = oDist;
}

void TreeNode::assign_params_CS_3D_RTRT()
{
    assert(childNodes.size() == 4);

    const size_t biggerDim  = std::max(length[0] * length[1], length[2]);
    const size_t smallerDim = std::min(length[0] * length[1], length[2]);
    const size_t padding
        = ((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512) ? 64 : 0;

    // B -> B
    auto& xyPlan = childNodes[0];
    assert((xyPlan->obOut == OB_USER_OUT) || (xyPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (xyPlan->obOut == OB_TEMP_BLUESTEIN));
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->TraverseTreeAssignParamsLogicA();

    // B -> T
    auto& trans1Plan = childNodes[1];
    assert(trans1Plan->obOut == OB_TEMP);
    trans1Plan->inStride = xyPlan->outStride;
    trans1Plan->iDist    = xyPlan->oDist;

    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[2] + padding);
    trans1Plan->outStride.push_back(trans1Plan->length[0] * trans1Plan->outStride[1]);
    trans1Plan->oDist = trans1Plan->length[1] * trans1Plan->outStride[2];

    for(size_t index = 3; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    // T -> T
    auto& zPlan = childNodes[2];
    assert(zPlan->obOut == OB_TEMP);
    zPlan->inStride = trans1Plan->outStride;
    zPlan->iDist    = trans1Plan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;

    zPlan->TraverseTreeAssignParamsLogicA();

    // T -> B
    auto& trans2Plan = childNodes[3];
    assert((trans2Plan->obOut == OB_USER_OUT) || (trans2Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (trans2Plan->obOut == OB_TEMP_BLUESTEIN));
    trans2Plan->inStride = zPlan->outStride;
    trans2Plan->iDist    = zPlan->oDist;

    trans2Plan->outStride = outStride;
    trans2Plan->oDist     = oDist;
}

void TreeNode::assign_params_CS_3D_TRTRTR()
{
    assert(scheme == CS_3D_TRTRTR);
    assert(childNodes.size() == 6);

    for(int i = 0; i < 6; i += 2)
    {
        auto& trans_plan = childNodes[i];
        if(i == 0)
        {
            trans_plan->inStride = inStride;
            trans_plan->iDist    = iDist;
        }
        else
        {
            trans_plan->inStride = childNodes[i - 1]->outStride;
            trans_plan->iDist    = childNodes[i - 1]->oDist;
        }

        trans_plan->outStride.push_back(1);
        trans_plan->outStride.push_back(trans_plan->outStride[0] * trans_plan->length[1]);
        trans_plan->outStride.push_back(trans_plan->outStride[1] * trans_plan->length[2]);
        trans_plan->oDist = trans_plan->outStride[2] * trans_plan->length[0];

        auto& row_plan     = childNodes[i + 1];
        row_plan->inStride = trans_plan->outStride;
        row_plan->iDist    = trans_plan->oDist;

        if(i == 4)
        {
            row_plan->outStride = outStride;
            row_plan->oDist     = oDist;
        }
        else
        {
            row_plan->outStride.push_back(1);
            row_plan->outStride.push_back(row_plan->outStride[0] * row_plan->length[0]);
            row_plan->outStride.push_back(row_plan->outStride[1] * row_plan->length[1]);
            row_plan->oDist = row_plan->outStride[2] * row_plan->length[2];
        }
        row_plan->TraverseTreeAssignParamsLogicA();
    }
}

void TreeNode::assign_params_CS_3D_BLOCK_RC()
{
    assert(scheme == CS_3D_BLOCK_RC);
    // could go as low as 3 kernels if all dimensions are SBRC-able,
    // but less than 6.  If we ended up with 6 we should have just
    // done 3D_TRTRTR instead.
    assert(childNodes.size() >= 3 && childNodes.size() < 6);

    childNodes.front()->inStride = inStride;
    childNodes.front()->iDist    = iDist;

    std::vector<size_t> prev_outStride;
    size_t              prev_oDist = 0;
    for(auto& node : childNodes)
    {
        // set initial inStride + iDist, or connect it to previous node
        if(prev_outStride.empty())
        {
            node->inStride = inStride;
            node->iDist    = iDist;
        }
        else
        {
            node->inStride = prev_outStride;
            node->iDist    = prev_oDist;
        }

        // each node is either:
        // - a fused sbrc+transpose node (for dimensions we have SBRC kernels for)
        // - a transpose node or a row FFT node (otherwise)
        switch(node->scheme)
        {
        case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->length[2]);
            node->outStride.push_back(node->outStride[1] * node->length[0]);
            node->oDist = node->outStride[2] * node->length[1];
            break;
        }
        case CS_KERNEL_STOCKHAM:
        {
            node->outStride = node->inStride;
            node->oDist     = node->iDist;
            node->TraverseTreeAssignParamsLogicA();
            break;
        }
        case CS_KERNEL_TRANSPOSE_XY_Z:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->outStride[0] * node->length[2]);
            node->outStride.push_back(node->outStride[1] * node->length[0]);
            node->oDist = node->iDist;
            break;
        }
        default:
            // build_CS_3D_BLOCK_RC should not have created any other node types
            assert(false);
        }
        prev_outStride = node->outStride;
        prev_oDist     = node->oDist;
    }
    childNodes.back()->outStride = outStride;
    childNodes.back()->oDist     = oDist;
}

void TreeNode::assign_params_CS_3D_RC_STRAIGHT()
{
    auto& xyPlan = childNodes[0];
    auto& zPlan  = childNodes[1];

    // B -> B
    assert((xyPlan->obOut == OB_USER_OUT) || (xyPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (xyPlan->obOut == OB_TEMP_BLUESTEIN));
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->TraverseTreeAssignParamsLogicA();

    // B -> B
    assert((zPlan->obOut == OB_USER_OUT) || (zPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (zPlan->obOut == OB_TEMP_BLUESTEIN));
    zPlan->inStride.push_back(inStride[2]);
    zPlan->inStride.push_back(inStride[0]);
    zPlan->inStride.push_back(inStride[1]);
    for(size_t index = 3; index < length.size(); index++)
        zPlan->inStride.push_back(inStride[index]);

    zPlan->iDist = xyPlan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;
}

///////////////////////////////////////////////////////////////////////////////
/// Collect leaf node and calculate work memory requirements

void TreeNode::TraverseTreeCollectLeafsLogicA(std::vector<TreeNode*>& seq,
                                              size_t&                 tmpBufSize,
                                              size_t&                 cmplxForRealSize,
                                              size_t&                 blueSize,
                                              size_t&                 chirpSize)
{
    if(childNodes.size() == 0)
    {
        if(scheme == CS_KERNEL_CHIRP)
        {
            chirpSize = std::max(2 * lengthBlue, chirpSize);
        }
        if(obOut == OB_TEMP_BLUESTEIN)
        {
            blueSize = std::max(oDist * batch, blueSize);
        }
        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
        {
            cmplxForRealSize = std::max(oDist * batch, cmplxForRealSize);
        }
        if(obOut == OB_TEMP)
        {
            tmpBufSize = std::max(oDist * batch, tmpBufSize);
        }
        seq.push_back(this);
    }
    else
    {
        for(auto children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
        {
            (*children_p)
                ->TraverseTreeCollectLeafsLogicA(
                    seq, tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
        }
    }
}

void TreeNode::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << "\n" << indentStr.c_str() << "scheme: " << PrintScheme(scheme).c_str();
    os << "\n" << indentStr.c_str();
    os << "dimension: " << dimension;
    os << "\n" << indentStr.c_str();
    os << "batch: " << batch;
    os << "\n" << indentStr.c_str();
    os << "length: ";
    for(size_t i = 0; i < length.size(); i++)
    {
        os << length[i] << " ";
    }

    os << "\n" << indentStr.c_str() << "iStrides: ";
    for(size_t i = 0; i < inStride.size(); i++)
        os << inStride[i] << " ";

    os << "\n" << indentStr.c_str() << "oStrides: ";
    for(size_t i = 0; i < outStride.size(); i++)
        os << outStride[i] << " ";

    if(iOffset)
    {
        os << "\n" << indentStr.c_str();
        os << "iOffset: " << iOffset;
    }
    if(oOffset)
    {
        os << "\n" << indentStr.c_str();
        os << "oOffset: " << oOffset;
    }

    os << "\n" << indentStr.c_str();
    os << "iDist: " << iDist;
    os << "\n" << indentStr.c_str();
    os << "oDist: " << oDist;

    os << "\n" << indentStr.c_str();
    os << "direction: " << direction;

    os << "\n" << indentStr.c_str();
    os << ((placement == rocfft_placement_inplace) ? "inplace" : "not inplace");

    os << "\n" << indentStr.c_str();

    os << ((precision == rocfft_precision_single) ? "single-precision" : "double-precision");

    os << std::endl << indentStr.c_str();
    os << "array type: ";
    switch(inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        os << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        os << "complex planar";
        break;
    case rocfft_array_type_real:
        os << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        os << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        os << "hermitian planar";
        break;
    default:
        os << "unset";
        break;
    }
    os << " -> ";
    switch(outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        os << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        os << "complex planar";
        break;
    case rocfft_array_type_real:
        os << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        os << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        os << "hermitian planar";
        break;
    default:
        os << "unset";
        break;
    }
    if(large1D)
    {
        os << "\n" << indentStr.c_str() << "large1D: " << large1D;
        os << "\n" << indentStr.c_str() << "largeTwdBase: " << largeTwdBase;
    }
    if(lengthBlue)
        os << "\n" << indentStr.c_str() << "lengthBlue: " << lengthBlue;
    os << "\n";
    switch(ebtype)
    {
    case EmbeddedType::NONE:
        break;
    case EmbeddedType::C2Real_PRE:
        os << indentStr.c_str() << "EmbeddedType: C2Real_PRE\n";
        break;
    case EmbeddedType::Real2C_POST:
        os << indentStr.c_str() << "EmbeddedType: Real2C_POST\n";
        break;
    }

    os << indentStr << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << "\n";
    os << indentStr << PrintOperatingBufferCode(obIn) << " -> " << PrintOperatingBufferCode(obOut)
       << "\n";
    for(const auto& c : comments)
    {
        os << indentStr << "comment: " << c << "\n";
    }

    if(childNodes.size())
    {
        for(auto& children_p : childNodes)
        {
            children_p->Print(os, indent + 1);
        }
    }
    std::cout << std::flush;
}

void TreeNode::RecursiveRemoveNode(TreeNode* node)
{
    for(auto& child : childNodes)
        child->RecursiveRemoveNode(node);
    childNodes.erase(std::remove_if(childNodes.begin(),
                                    childNodes.end(),
                                    [node](const std::unique_ptr<TreeNode>& child) {
                                        return child.get() == node;
                                    }),
                     childNodes.end());
}

// remove a leaf node from the plan completely - plan optimization
// can remove unnecessary nodes to skip unnecessary work.
void RemoveNode(ExecPlan& execPlan, TreeNode* node)
{
    auto& execSeq = execPlan.execSeq;
    // remove it from the non-owning leaf nodes
    execSeq.erase(std::remove(execSeq.begin(), execSeq.end(), node), execSeq.end());

    // remove it from the tree structure
    execPlan.rootPlan->RecursiveRemoveNode(node);
}

static rocfft_result_placement EffectivePlacement(OperatingBuffer         obIn,
                                                  OperatingBuffer         obOut,
                                                  rocfft_result_placement rootPlacement)
{
    if(rootPlacement == rocfft_placement_inplace)
    {
        // in == out
        if((obIn == OB_USER_IN || obIn == OB_USER_OUT)
           && (obOut == OB_USER_IN || obOut == OB_USER_OUT))
            return rocfft_placement_inplace;
    }
    // otherwise just check if the buffers look different
    return obIn == obOut ? rocfft_placement_inplace : rocfft_placement_notinplace;
}

void Optimize_Transpose_With_Strides(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    // If this is a stockham fft that does multiple rows in one
    // kernel, followed by a 2d transpose, adjust output strides to
    // replace the transpose.  Multiple rows will ensure that the
    // transposed column writes are coalesced.
    for(auto it = execSeq.begin(); it != execSeq.end(); ++it)
    {
        auto stockham = *it;
        if(stockham->scheme != CS_KERNEL_STOCKHAM)
            continue;

        size_t wgs, numTrans;
        DetermineSizes(stockham->length[0], wgs, numTrans);
        if(numTrans < 2)
            continue;

        auto next = it + 1;
        if(next == execSeq.end())
            break;
        auto transpose = *next;
        if(transpose->scheme == CS_KERNEL_TRANSPOSE
           && stockham->length == transpose->length
           // can't get rid of a transpose that also does twiddle multiplication
           && transpose->large1D == 0
           // This is a transpose, which must be out-of-place
           && EffectivePlacement(stockham->obIn, transpose->obOut, execPlan.rootPlan->placement)
                  == rocfft_placement_notinplace)
        {
            stockham->outStride = transpose->outStride;
            std::swap(stockham->outStride[0], stockham->outStride[1]);
            stockham->obOut        = transpose->obOut;
            stockham->outArrayType = transpose->outArrayType;
            stockham->placement    = rocfft_placement_notinplace;
            stockham->oDist        = transpose->oDist;
            RemoveNode(execPlan, transpose);
        }
    }

    // Similarly, if we have a 2D transpose followed by a stockham
    // fft that does multiple rows in one kernel, adjust input
    // strides to replace the transpose.  Multiple rows will ensure
    // that the transposed column reads are coalesced.
    for(auto it = execSeq.begin(); it != execSeq.end(); ++it)
    {
        auto transpose = *it;
        // can't get rid of a transpose that also does twiddle multiplication
        if(transpose->scheme != CS_KERNEL_TRANSPOSE || transpose->large1D != 0)
            continue;

        auto next = it + 1;
        if(next == execSeq.end())
            break;
        auto stockham = *next;

        if(stockham->scheme != CS_KERNEL_STOCKHAM)
            continue;

        size_t wgs, numTrans;
        DetermineSizes(stockham->length[0], wgs, numTrans);
        if(numTrans < 2)
            continue;

        // This is a transpose, which must be out-of-place
        if(EffectivePlacement(transpose->obIn, stockham->obOut, execPlan.rootPlan->placement)
           == rocfft_placement_notinplace)
        {
            stockham->inStride = transpose->inStride;
            std::swap(stockham->inStride[0], stockham->inStride[1]);
            stockham->obIn        = transpose->obIn;
            stockham->inArrayType = transpose->inArrayType;
            stockham->placement   = rocfft_placement_notinplace;
            stockham->iDist       = transpose->iDist;
            RemoveNode(execPlan, transpose);
        }
    }
}

void Optimize_R_TO_CMPLX_TRANSPOSE(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    // combine R_TO_CMPLX and following transpose
    auto it = std::find_if(execSeq.begin(), execSeq.end(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_R_TO_CMPLX;
    });
    if(it != execSeq.end())
    {
        auto r_to_cmplx = *it;
        // check that next node is transpose
        ++it;
        if(it == execSeq.end())
            return;
        auto transpose = *it;
        if(transpose->scheme == CS_KERNEL_TRANSPOSE
           || transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
        {
            r_to_cmplx->obOut        = transpose->obOut;
            r_to_cmplx->scheme       = CS_KERNEL_R_TO_CMPLX_TRANSPOSE;
            r_to_cmplx->outArrayType = transpose->outArrayType;
            r_to_cmplx->placement    = EffectivePlacement(
                r_to_cmplx->obIn, r_to_cmplx->obOut, execPlan.rootPlan->placement);
            // transpose must be out-of-place
            assert(r_to_cmplx->placement == rocfft_placement_notinplace);
            r_to_cmplx->outStride = transpose->outStride;
            r_to_cmplx->oDist     = transpose->oDist;
            r_to_cmplx->comments.push_back("fused " + PrintScheme(CS_KERNEL_R_TO_CMPLX)
                                           + " and following " + PrintScheme(transpose->scheme));
            RemoveNode(execPlan, transpose);
        }
    }
}

void Optimize_TRANSPOSE_CMPLX_TO_R(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    // combine CMPLX_TO_R with preceding transpose
    auto cmplx_to_r = std::find_if(execSeq.rbegin(), execSeq.rend(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_CMPLX_TO_R;
    });
    // should be a stockham or bluestein kernel following the CMPLX_TO_R, so
    // CMPLX_TO_R can't be the last node either
    if(cmplx_to_r != execSeq.rend() && cmplx_to_r != execSeq.rbegin())
    {
        auto following = cmplx_to_r - 1;
        if((*following)->scheme == CS_KERNEL_CHIRP)
            following = following - 1; // skip CHIRP
        auto transpose = cmplx_to_r + 1;
        if(transpose != execSeq.rend()
           && ((*transpose)->scheme == CS_KERNEL_TRANSPOSE
               || (*transpose)->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
           && cmplx_to_r != execSeq.rbegin())
        {
            // connect the transpose operation to the following
            // transform by default
            (*cmplx_to_r)->obIn  = (*transpose)->obIn;
            (*cmplx_to_r)->obOut = (*following)->obIn;

            // but transpose needs to be out-of-place, so bring the
            // temp buffer in if the operation would be effectively
            // in-place.
            if(EffectivePlacement(
                   (*cmplx_to_r)->obIn, (*cmplx_to_r)->obOut, execPlan.rootPlan->placement)
               == rocfft_placement_inplace)
            {
                (*cmplx_to_r)->obOut    = OB_TEMP;
                (*following)->obIn      = OB_TEMP;
                (*following)->placement = EffectivePlacement(
                    (*following)->obIn, (*following)->obOut, execPlan.rootPlan->placement);
            }
            (*cmplx_to_r)->placement = rocfft_placement_notinplace;

            (*cmplx_to_r)->scheme      = CS_KERNEL_TRANSPOSE_CMPLX_TO_R;
            (*cmplx_to_r)->inArrayType = (*transpose)->inArrayType;
            (*cmplx_to_r)->inStride    = (*transpose)->inStride;
            (*cmplx_to_r)->length      = (*transpose)->length;
            (*cmplx_to_r)->iDist       = (*transpose)->iDist;
            (*cmplx_to_r)
                ->comments.push_back("fused " + PrintScheme(CS_KERNEL_CMPLX_TO_R)
                                     + " and preceding " + PrintScheme((*transpose)->scheme));
            RemoveNode(execPlan, *transpose);
        }
    }
}

// combine CS_KERNEL_STOCKHAM and following CS_KERNEL_TRANSPOSE_Z_XY if possible
void Optimize_STOCKHAM_TRANSPOSE_Z_XY(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    for(auto it = execSeq.rbegin() + 1; it != execSeq.rend(); ++it)
    {

        if((it != execSeq.rend()) && ((*it)->scheme == CS_KERNEL_STOCKHAM)
           && ((*(it - 1))->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
           && ((*(it - 1))->use_CS_KERNEL_TRANSPOSE_Z_XY()) // kernel available
           && ((*it)->obIn
               != ((*(it - 1))->obOut)) // "in-place" doesn't work without manipulating buffers
           && (!(((it + 1) != execSeq.rend())
                 && (*(it + 1))->scheme
                        == CS_KERNEL_TRANSPOSE_XY_Z)) // don't touch case XY_Z -> FFT -> Z_XY
        )
        {
            auto stockham  = it;
            auto transpose = it - 1;

            (*stockham)->obOut        = (*transpose)->obOut;
            (*stockham)->scheme       = CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY;
            (*stockham)->outArrayType = (*transpose)->outArrayType;
            (*stockham)->placement    = EffectivePlacement(
                (*stockham)->obIn, (*stockham)->obOut, execPlan.rootPlan->placement);
            // transpose must be out-of-place
            assert((*stockham)->placement == rocfft_placement_notinplace);
            (*stockham)->outStride = (*transpose)->outStride;
            (*stockham)->oDist     = (*transpose)->oDist;

            (*stockham)->comments.push_back("fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                                            + " and following "
                                            + PrintScheme((*transpose)->scheme));
            RemoveNode(execPlan, *transpose);
        }
    }
}

// combine one CS_KERNEL_STOCKHAM and following CS_KERNEL_TRANSPOSE_XY_Z in 3D complex to real
// NB: this should be replaced by combining CS_KERNEL_TRANSPOSE_XY_Z and the following
//     CS_KERNEL_STOCKHAM eventually, in which we might fuse 2 pairs of TR.
void Optimize_STOCKHAM_TRANSPOSE_XY_Z(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    auto trans_cmplx_to_r = std::find_if(execSeq.rbegin(), execSeq.rend(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R;
    });
    if(trans_cmplx_to_r != execSeq.rend() && trans_cmplx_to_r != execSeq.rbegin())
    {
        auto stockham2  = trans_cmplx_to_r + 1;
        auto transpose2 = trans_cmplx_to_r + 2;
        auto stockham1  = trans_cmplx_to_r + 3;
        if(stockham1 != execSeq.rend() && (*stockham2)->scheme == CS_KERNEL_STOCKHAM
           && (*transpose2)->scheme == CS_KERNEL_TRANSPOSE_XY_Z
           && (*stockham1)->scheme == CS_KERNEL_STOCKHAM
           && (function_pool::has_function(fpkey((*transpose2)->length[0],
                                                 (*transpose2)->precision,
                                                 CS_KERNEL_STOCKHAM_BLOCK_RC))) // kernel available
           && ((*transpose2)->length[0]
               == (*transpose2)->length[2]) // limit to original "cubic" case
           && ((*transpose2)->length[0] / 2 + 1 == (*transpose2)->length[1])
           && (!IsPo2((*transpose2)->length[0]))) // Need more investigation for diagonal transpose
        {
            (*transpose2)->scheme       = CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z;
            (*transpose2)->obIn         = (*stockham1)->obIn;
            (*transpose2)->obOut        = (*stockham2)->obOut;
            (*transpose2)->inArrayType  = (*stockham1)->inArrayType;
            (*transpose2)->outArrayType = (*stockham2)->outArrayType;

            (*stockham2)->obIn        = (*stockham2)->obOut;
            (*stockham2)->inArrayType = (*stockham2)->outArrayType;
            (*stockham2)->placement   = rocfft_placement_inplace;

            (*transpose2)
                ->comments.push_back("fused " + PrintScheme(CS_KERNEL_TRANSPOSE_XY_Z)
                                     + " and preceding " + PrintScheme((*stockham1)->scheme));
            RemoveNode(execPlan, *stockham1);
        }
    }
}

// combine CS_KERNEL_R_TO_CMPLX_TRANSPOSE with preceding CS_KERNEL_STOCKHAM
void Optimize_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY(ExecPlan&               execPlan,
                                                 std::vector<TreeNode*>& execSeq)
{
    auto r_to_cmplx_transpose = std::find_if(execSeq.rbegin(), execSeq.rend(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE;
    });
    if(r_to_cmplx_transpose != execSeq.rend() && r_to_cmplx_transpose != execSeq.rbegin())
    {
        auto stockham = r_to_cmplx_transpose + 1;
        if((*stockham)->scheme == CS_KERNEL_STOCKHAM
           && (function_pool::has_function(fpkey((*stockham)->length[0],
                                                 (*stockham)->precision,
                                                 CS_KERNEL_STOCKHAM_BLOCK_RC))) // kernel available
           && ((*stockham)->length[0] * 2
               == (*stockham)->length[1]) // limit to original "cubic" case
           && (((*stockham)->length.size() == 2)
               || ((*stockham)->length[1] == (*stockham)->length[2])))
        {
            (*stockham)->scheme       = CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY;
            (*stockham)->obOut        = (*r_to_cmplx_transpose)->obOut;
            (*stockham)->outArrayType = (*r_to_cmplx_transpose)->outArrayType;
            (*stockham)->placement    = rocfft_placement_notinplace;
            (*stockham)->outStride    = (*r_to_cmplx_transpose)->outStride;
            (*stockham)->oDist        = (*r_to_cmplx_transpose)->oDist;

            // NB:
            //    The generated CS_KERNEL_R_TO_CMPLX_TRANSPOSE kernel is in 3D fashion.
            //    We just need extend length and strides to make it work for 2D case.
            if((*stockham)->length.size() == 2)
            {
                (*stockham)->length.push_back(1);
                (*stockham)->inStride.push_back((*stockham)->inStride[1]);
                (*stockham)->outStride.push_back((*stockham)->outStride[1]);
            }

            (*stockham)->comments.push_back("fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                                            + " and following "
                                            + PrintScheme((*r_to_cmplx_transpose)->scheme));

            // NB:
            //   Don't remove "*stockham" kernel beacause the combined 2 twiddle tables
            //   are stored there.
            RemoveNode(execPlan, *r_to_cmplx_transpose);
        }
    }
}

static void OptimizePlan(ExecPlan& execPlan)
{
    auto passes = {
        Optimize_R_TO_CMPLX_TRANSPOSE,
        Optimize_TRANSPOSE_CMPLX_TO_R,
        Optimize_STOCKHAM_TRANSPOSE_Z_XY,
        Optimize_STOCKHAM_TRANSPOSE_XY_Z,
        Optimize_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY,
        Optimize_Transpose_With_Strides,
    };
    for(auto pass : passes)
    {
        // Give each optimization pass its own copy of execSeq, so
        // that it can call RemoveNode without worrying about
        // invalidating iterators it might be using.  But it can also
        // modify nodes willy-nilly via the TreeNode*'s in the vector.
        auto localExecSeq = execPlan.execSeq;
        pass(execPlan, localExecSeq);
    }
}

void ProcessNode(ExecPlan& execPlan)
{
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->dimension);

    execPlan.rootPlan->RecursiveBuildTree();

    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

    // initialize root plan input/output location if not already done
    if(execPlan.rootPlan->obOut == OB_UNINIT)
        execPlan.rootPlan->obOut = OB_USER_OUT;
    if(execPlan.rootPlan->obIn == OB_UNINIT)
        execPlan.rootPlan->obIn
            = execPlan.rootPlan->placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
    // initialize traverse state so we can initialize obIn + obOut for all nodes
    TreeNode::TraverseState state(execPlan);
    OperatingBuffer         flipIn = OB_UNINIT, flipOut = OB_UNINIT, obOutBuf = OB_UNINIT;
    execPlan.rootPlan->TraverseTreeAssignBuffersLogicA(state, flipIn, flipOut, obOutBuf);

    execPlan.rootPlan->TraverseTreeAssignPlacementsLogicA(execPlan.rootPlan->inArrayType,
                                                          execPlan.rootPlan->outArrayType);
    execPlan.rootPlan->TraverseTreeAssignParamsLogicA();

    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->TraverseTreeCollectLeafsLogicA(
        execPlan.execSeq, tmpBufSize, cmplxForRealSize, blueSize, chirpSize);

    OptimizePlan(execPlan);

    execPlan.workBufSize      = tmpBufSize + cmplxForRealSize + blueSize + chirpSize;
    execPlan.tmpWorkBufSize   = tmpBufSize;
    execPlan.copyWorkBufSize  = cmplxForRealSize;
    execPlan.blueWorkBufSize  = blueSize;
    execPlan.chirpWorkBufSize = chirpSize;
}

void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan)
{
    os << "**********************************************************************"
          "*********"
       << std::endl;

    const size_t N = std::accumulate(execPlan.rootPlan->length.begin(),
                                     execPlan.rootPlan->length.end(),
                                     execPlan.rootPlan->batch,
                                     std::multiplies<size_t>());
    os << "Work buffer size: " << execPlan.workBufSize << std::endl;
    os << "Work buffer ratio: " << (double)execPlan.workBufSize / (double)N << std::endl;

    if(execPlan.execSeq.size() > 1)
    {
        std::vector<TreeNode*>::const_iterator prev_p = execPlan.execSeq.begin();
        std::vector<TreeNode*>::const_iterator curr_p = prev_p + 1;
        while(curr_p != execPlan.execSeq.end())
        {
            if((*curr_p)->placement == rocfft_placement_inplace)
            {
                for(size_t i = 0; i < (*curr_p)->inStride.size(); i++)
                {
                    const int infact  = (*curr_p)->inArrayType == rocfft_array_type_real ? 1 : 2;
                    const int outfact = (*curr_p)->outArrayType == rocfft_array_type_real ? 1 : 2;
                    if(outfact * (*curr_p)->inStride[i] != infact * (*curr_p)->outStride[i])
                    {
                        os << "error in stride assignments" << std::endl;
                    }
                    if(outfact * (*curr_p)->iDist != infact * (*curr_p)->oDist)
                    {
                        os << "error in dist assignments" << std::endl;
                    }
                }
            }

            if((*prev_p)->scheme != CS_KERNEL_CHIRP && (*curr_p)->scheme != CS_KERNEL_CHIRP)
            {
                if((*prev_p)->obOut != (*curr_p)->obIn)
                {
                    os << "error in buffer assignments" << std::endl;
                }
            }

            prev_p = curr_p;
            curr_p++;
        }
    }

    execPlan.rootPlan->Print(os, 0);

    os << "GridParams\n";
    for(const auto& gp : execPlan.gridParam)
    {
        os << "  b[" << gp.b_x << "," << gp.b_y << "," << gp.b_z << "] tpb[" << gp.tpb_x << ","
           << gp.tpb_y << "," << gp.tpb_z << "]\n";
    }
    os << "End GridParams\n";

    os << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
