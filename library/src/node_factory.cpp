// Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "node_factory.h"
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "radix_table.h"
#include "tree_node_1D.h"
#include "tree_node_2D.h"
#include "tree_node_3D.h"
#include "tree_node_bluestein.h"
#include "tree_node_real.h"

#include <functional>
#include <set>

// TODO:
//   - better data structure, and more elements for non pow of 2
//   - validate corresponding functions existing in function pool or not
NodeFactory::Map1DLength const NodeFactory::map1DLengthSingle
    = {{8192, 64}, // pow of 2: CC (64cc + 128rc)
       {16384, 64}, //          CC (64cc + 256rc) // 128x128 no faster
       {32768, 128}, //         CC (128cc + 256rc)
       {65536, 256}, //         CC (256cc + 256rc)
       {131072, 64}, //         CRT(64cc + 2048 + transpose)
       {262144, 64}, //         CRT(64cc + 4096 + transpose)
       {6561, 81}, // pow of 3: CC (81cc + 81rc)
       {10000, 100}, // mixed:  CC (100cc + 100rc)
       {40000, 200}}; //        CC (200cc + 200rc)

NodeFactory::Map1DLength const NodeFactory::map1DLengthDouble
    = {{4096, 64}, // pow of 2: CC (64cc + 64rc)
       {8192, 64}, //           CC (64cc + 128rc)
       {16384, 128}, //         CC (128cc + 128rc) // faster than 64x256
       {32768, 128}, //         CC (128cc + 256rc)
       {65536, 256}, //         CC (256cc + 256rc) // {65536, 64}
       {131072, 64}, //         CRT(64cc + 2048 + transpose)
       {6561, 81}, // pow of 3: CC (81cc + 81rc)
       {2500, 50}, // mixed:    CC (50cc + 50rc)
       {10000, 100}, //         CC (100cc + 100rc)
       {40000, 200}}; //        CC (200cc + 200rc)

//
// Factorisation helpers
//

// Return true if the order of factors in a decomposition should be
// reversed.  This improves performance for some lengths.
inline bool reverse_factors(size_t length)
{
    std::set<size_t> reverse_factors_lengths = {32256, 43008};
    return reverse_factors_lengths.count(length) == 1;
}

// Search function pool for length where is_supported_factor(length) returns true.
inline size_t search_pool(rocfft_precision            precision,
                          size_t                      length,
                          std::function<bool(size_t)> is_supported_factor)
{
    // query supported lengths from function pool, largest to smallest
    auto supported  = function_pool::get_lengths(precision, CS_KERNEL_STOCKHAM);
    auto comparison = std::greater<size_t>();
    std::sort(supported.begin(), supported.end(), comparison);

    if(supported.empty())
        return 0;

    // start search slightly smaller than sqrt(length)
    auto v     = (size_t)sqrt(length);
    auto lower = std::lower_bound(supported.cbegin(), supported.cend(), v, comparison);
    if(*lower < sqrt(length))
        lower--;

    auto upper = supported.cend();

    // search!
    auto itr = std::find_if(lower, upper, is_supported_factor);
    if(itr != supported.cend())
        return *itr;

    return 0;
}

// Return largest factor that has BOTH functions in the pool.
inline size_t get_explicitly_supported_factor(rocfft_precision precision, size_t length)
{
    auto supported_factor = [length, precision = precision](size_t factor) -> bool {
        bool is_factor        = length % factor == 0;
        bool has_other_kernel = function_pool::has_function(fpkey(length / factor, precision));
        return is_factor && has_other_kernel;
    };
    auto factor = search_pool(precision, length, supported_factor);
    if(factor > 0 && reverse_factors(length))
        return length / factor;
    return factor;
}

// Return largest factor that has a function in the pool.
inline size_t get_largest_supported_factor(rocfft_precision precision, size_t length)
{
    auto supported_factor = [length](size_t factor) -> bool {
        bool is_factor = length % factor == 0;
        return is_factor;
    };
    return search_pool(precision, length, supported_factor);
}

inline bool SupportedLength(rocfft_precision precision, size_t len)
{
    // do we have an explicit kernel?
    if(function_pool::has_function(fpkey(len, precision)))
        return true;

    // can we factor with using only base radix?
    size_t p = len;
    while(!(p % 2))
        p /= 2;
    while(!(p % 3))
        p /= 3;
    while(!(p % 5))
        p /= 5;
    while(!(p % 7))
        p /= 7;
    while(!(p % 11))
        p /= 11;
    while(!(p % 13))
        p /= 13;
    while(!(p % 17))
        p /= 17;

    if(p == 1)
        return true;

    // do we have an explicit kernel for the remainder?
    if(function_pool::has_function(fpkey(p, precision)))
        return true;

    // finally, can we factor this length with combinations of existing kernels?
    if(get_explicitly_supported_factor(precision, len) > 0)
        return true;

    return false;
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

// std::unique_ptr<TreeNode> NodeFactory::CreateNode(TreeNode* parentNode)
// {
//     return std::unique_ptr<TreeNode>(new TreeNode(parentNode));
// }

std::unique_ptr<TreeNode> NodeFactory::CreateNodeFromScheme(ComputeScheme s, TreeNode* parent)
{
    switch(s)
    {
    // Internal Node
    case CS_REAL_TRANSFORM_USING_CMPLX:
        return std::unique_ptr<RealTransCmplxNode>(new RealTransCmplxNode(parent));
    case CS_REAL_TRANSFORM_EVEN:
        return std::unique_ptr<RealTransEvenNode>(new RealTransEvenNode(parent));
    case CS_REAL_2D_EVEN:
        return std::unique_ptr<Real2DEvenNode>(new Real2DEvenNode(parent));
    case CS_REAL_3D_EVEN:
        return std::unique_ptr<Real3DEvenNode>(new Real3DEvenNode(parent));
    case CS_BLUESTEIN:
        return std::unique_ptr<BluesteinNode>(new BluesteinNode(parent));
    case CS_L1D_TRTRT:
        return std::unique_ptr<TRTRT1DNode>(new TRTRT1DNode(parent));
    case CS_L1D_CC:
        return std::unique_ptr<CC1DNode>(new CC1DNode(parent));
    case CS_L1D_CRT:
        return std::unique_ptr<CRT1DNode>(new CRT1DNode(parent));
    case CS_2D_RTRT:
        return std::unique_ptr<RTRT2DNode>(new RTRT2DNode(parent));
    case CS_2D_RC:
        return std::unique_ptr<RC2DNode>(new RC2DNode(parent));
    case CS_3D_RTRT:
        return std::unique_ptr<RTRT3DNode>(new RTRT3DNode(parent));
    case CS_3D_TRTRTR:
        return std::unique_ptr<TRTRTR3DNode>(new TRTRTR3DNode(parent));
    case CS_3D_BLOCK_RC:
        return std::unique_ptr<BLOCKRC3DNode>(new BLOCKRC3DNode(parent));
    case CS_3D_BLOCK_CR:
        return std::unique_ptr<BLOCKCR3DNode>(new BLOCKCR3DNode(parent));
    case CS_3D_RC:
        return std::unique_ptr<RC3DNode>(new RC3DNode(parent));

    // Leaf Node that need to check external kernel file
    case CS_KERNEL_STOCKHAM:
        return std::unique_ptr<Stockham1DNode>(new Stockham1DNode(parent, s));
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
        return std::unique_ptr<SBCCNode>(new SBCCNode(parent, s));
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
        return std::unique_ptr<SBRCNode>(new SBRCNode(parent, s));
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
        return std::unique_ptr<SBCRNode>(new SBCRNode(parent, s));
    case CS_KERNEL_2D_SINGLE:
        return std::unique_ptr<Single2DNode>(new Single2DNode(parent, s));
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        return std::unique_ptr<SBRCTransXY_ZNode>(new SBRCTransXY_ZNode(parent, s));
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        return std::unique_ptr<SBRCTransZ_XYNode>(new SBRCTransZ_XYNode(parent, s));
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
        return std::unique_ptr<RealCmplxTransZ_XYNode>(new RealCmplxTransZ_XYNode(parent, s));

    // Leaf Node that doesn't need to check external kernel file
    case CS_KERNEL_R_TO_CMPLX:
    case CS_KERNEL_R_TO_CMPLX_TRANSPOSE:
    case CS_KERNEL_CMPLX_TO_R:
    case CS_KERNEL_TRANSPOSE_CMPLX_TO_R:
        return std::unique_ptr<PrePostKernelNode>(new PrePostKernelNode(parent, s));
    case CS_KERNEL_TRANSPOSE:
    case CS_KERNEL_TRANSPOSE_XY_Z:
    case CS_KERNEL_TRANSPOSE_Z_XY:
        return std::unique_ptr<TransposeNode>(new TransposeNode(parent, s));
    case CS_KERNEL_COPY_R_TO_CMPLX:
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
    case CS_KERNEL_COPY_CMPLX_TO_R:
    case CS_KERNEL_APPLY_CALLBACK:
        return std::unique_ptr<RealTransDataCopyNode>(new RealTransDataCopyNode(parent, s));
    case CS_KERNEL_CHIRP:
    case CS_KERNEL_PAD_MUL:
    case CS_KERNEL_FFT_MUL:
    case CS_KERNEL_RES_MUL:
        return std::unique_ptr<BluesteinComponentNode>(new BluesteinComponentNode(parent, s));
    default:
        throw std::runtime_error("Scheme assertion failed, node not implemented:" + PrintScheme(s));
        return nullptr;
    }
}

std::unique_ptr<TreeNode> NodeFactory::CreateExplicitNode(NodeMetaData& nodeData, TreeNode* parent)
{
    // TreeNode*     p = dummyNode->parent;
    ComputeScheme s = DecideNodeScheme(nodeData, parent);
    assert(s != CS_NONE);
    auto node = CreateNodeFromScheme(s, parent);
    node->CopyNodeData(nodeData);
    return node;
}

ComputeScheme NodeFactory::DecideNodeScheme(NodeMetaData& nodeData, TreeNode* parent)
{
    if((parent == nullptr)
       && ((nodeData.inArrayType == rocfft_array_type_real)
           || (nodeData.outArrayType == rocfft_array_type_real)))
    {
        return DecideRealScheme(nodeData);
    }

    switch(nodeData.dimension)
    {
    case 1:
        return Decide1DScheme(nodeData);
    case 2:
        return Decide2DScheme(nodeData);
    case 3:
        return Decide3DScheme(nodeData);
    default:
        throw std::runtime_error("Invalid dimension");
        assert(false);
    }

    return CS_NONE;
}

ComputeScheme NodeFactory::DecideRealScheme(NodeMetaData& nodeData)
{
    if(nodeData.length[0] % 2 == 0 && nodeData.inStride[0] == 1 && nodeData.outStride[0] == 1)
    {
        switch(nodeData.dimension)
        {
        case 1:
            return CS_REAL_TRANSFORM_EVEN;
        case 2:
            return CS_REAL_2D_EVEN;
        case 3:
            return CS_REAL_3D_EVEN;
        default:
            throw std::runtime_error("Invalid dimension");
        }
    }
    // Fallback method
    return CS_REAL_TRANSFORM_USING_CMPLX;
}

ComputeScheme NodeFactory::Decide1DScheme(NodeMetaData& nodeData)
{
    ComputeScheme scheme = CS_NONE;

    // Build a node for a 1D FFT
    if(!SupportedLength(nodeData.precision, nodeData.length[0]))
        return CS_BLUESTEIN;

    if(function_pool::has_function(fpkey(nodeData.length[0], nodeData.precision)))
    {
        return CS_KERNEL_STOCKHAM;
    }

    size_t divLength1 = 1;
    bool   failed     = false;

    if(IsPo2(nodeData.length[0])) // multiple kernels involving transpose
    {
        // TODO: wrap the below into a function and check with LDS size
        auto block_threshold = 262144;
        if(nodeData.precision == rocfft_precision_double)
            block_threshold /= 2;
        if(nodeData.length[0] <= block_threshold)
        {
            // Enable block compute under these conditions
            if(nodeData.precision == rocfft_precision_single)
            {
                if(map1DLengthSingle.find(nodeData.length[0]) != map1DLengthSingle.end())
                {
                    divLength1 = map1DLengthSingle.at(nodeData.length[0]);
                }
                else
                {
                    failed = true;
                }
            }
            else
            {
                if(map1DLengthDouble.find(nodeData.length[0]) != map1DLengthDouble.end())
                {
                    divLength1 = map1DLengthDouble.at(nodeData.length[0]);
                }
                else
                {
                    failed = true;
                }
            }
            // up to 256cc + 256rc for the 1arge-1D, use CRT for even larger
            scheme = (nodeData.length[0] <= 65536) ? CS_L1D_CC : CS_L1D_CRT;
        }
        else
        {
            auto largest = function_pool::get_largest_length(nodeData.precision);
            if(nodeData.length[0] > largest * largest)
            {
                divLength1 = nodeData.length[0] / largest;
            }
            else
            {
                size_t in_x = 0;
                size_t len  = nodeData.length[0];
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
        if(nodeData.precision == rocfft_precision_single)
        {
            if(map1DLengthSingle.find(nodeData.length[0]) != map1DLengthSingle.end())
            {
                divLength1 = map1DLengthSingle.at(nodeData.length[0]);
                scheme     = CS_L1D_CC;
            }
            else
            {
                failed = true;
            }
        }
        else if(nodeData.precision == rocfft_precision_double)
        {
            if(map1DLengthDouble.find(nodeData.length[0]) != map1DLengthDouble.end())
            {
                divLength1 = map1DLengthDouble.at(nodeData.length[0]);
                scheme     = CS_L1D_CC;
            }
            else
            {
                failed = true;
            }
        }

        if(failed)
        {
            scheme     = CS_L1D_TRTRT;
            divLength1 = get_explicitly_supported_factor(nodeData.precision, nodeData.length[0]);
            if(divLength1 == 0)
            {
                // We need to recurse.  Note, for CS_L1D_TRTRT,
                // divLength0 has to be explictly supported
                auto divLength0
                    = get_largest_supported_factor(nodeData.precision, nodeData.length[0]);
                divLength1 = (divLength0 == 0) ? 0 : nodeData.length[0] / divLength0;
            }
            failed = divLength1 == 0;
        }
    }

    if(failed)
    {
        PrintFailInfo(nodeData.precision, nodeData.length[0], scheme);
        assert(false); // can't find the length in map1DLengthSingle/Double.
        return CS_NONE;
    }

    // NOTE: we temporarily save the divLength1 at the end of length vector
    // and then get and pop later when building node
    // size_t divLength0 = length[0] / divLength1;
    nodeData.length.emplace_back(divLength1);

    return scheme;
}

ComputeScheme NodeFactory::Decide2DScheme(NodeMetaData& nodeData)
{
    // First choice is 2D_SINGLE kernel, if the problem will fit into LDS.
    // Next best is CS_2D_RC. Last resort is RTRT.
    if(use_CS_2D_SINGLE(nodeData))
        return CS_KERNEL_2D_SINGLE; // the node has all build info
    else if(use_CS_2D_RC(nodeData))
        return CS_2D_RC;
    else
        return CS_2D_RTRT;
}

ComputeScheme NodeFactory::Decide3DScheme(NodeMetaData& nodeData)
{
    // this flag can be enabled when generator can do block column fft in
    // multi-dimension cases and small 2d, 3d within one kernel
    bool MultiDimFuseKernelsAvailable = false;

    if(use_CS_3D_RC(nodeData))
    {
        return CS_3D_RC;
    }
    else if(MultiDimFuseKernelsAvailable)
    {
        // conditions to choose which scheme
        if((nodeData.length[0] * nodeData.length[1] * nodeData.length[2]) <= 2048)
            return CS_KERNEL_3D_SINGLE;
        else if(nodeData.length[2] <= 256)
            return CS_3D_RC;
        else
            return CS_3D_RTRT;
    }
    else
    {
        // if we can get down to 3 or 4 kernels via SBRC, prefer that
        if(use_CS_3D_BLOCK_RC(nodeData))
            return CS_3D_BLOCK_RC;

        // else, 3D_RTRT
        // NB:
        // Peek the 1st child but not really add it in.
        // Give up if 1st child is 2D_RTRT (means the poor RTRT_TRT)
        // Switch to TRTRTR as the last resort.
        NodeMetaData child0 = nodeData;
        child0.length       = nodeData.length;
        child0.dimension    = 2;
        auto childScheme    = DecideNodeScheme(child0, nullptr);
        if(childScheme == CS_2D_RTRT)
        {
            return CS_3D_TRTRTR;
        }

        return CS_3D_RTRT;
    }
    // TODO: CS_KERNEL_3D_SINGLE?
}

bool NodeFactory::use_CS_2D_SINGLE(NodeMetaData& nodeData)
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
    const auto single2DSizes = Single2DSizes(ldsSize, nodeData.precision, GetWGSAndNT);
    if(std::find(single2DSizes.begin(),
                 single2DSizes.end(),
                 std::make_pair(nodeData.length[0], nodeData.length[1]))
       != single2DSizes.end())
        return true;

    return false;
}

bool NodeFactory::use_CS_2D_RC(NodeMetaData& nodeData)
{
    //   For CS_2D_RC, we are reusing SBCC kernel for 1D middle size. The
    //   current implementation of 1D SBCC supports only 64, 128, and 256.
    //   However, technically no LDS limitation along the fast dimension
    //   on upper bound for 2D SBCC cases, and even should not limit to pow
    //   of 2.

    std::set<int> sbcc_support = {50, 64, 81, 100, 128, 200, 256, 336};
    if((sbcc_support.find(nodeData.length[1]) != sbcc_support.end()) && (nodeData.length[0] >= 56))
    {
        return true;
    }

    return false;
}

size_t NodeFactory::count_3D_SBRC_nodes(NodeMetaData& nodeData)
{
    size_t sbrc_dimensions = 0;
    for(unsigned int i = 0; i < nodeData.length.size(); ++i)
    {
        if(function_pool::has_SBRC_kernel(nodeData.length[i], nodeData.precision))
        {
            // make sure the SBRC kernel on that dimension would be tile-aligned
            size_t bwd, wgs, lds;
            GetBlockComputeTable(nodeData.length[i], bwd, wgs, lds);
            if(nodeData.length[(i + 2) % nodeData.length.size()] % bwd == 0)
                ++sbrc_dimensions;
        }
    }
    return sbrc_dimensions;
}

bool NodeFactory::use_CS_3D_BLOCK_RC(NodeMetaData& nodeData)
{
    // TODO: SBRC hasn't worked for inner batch (i/oDist == 1)
    if(nodeData.iDist == 1 || nodeData.oDist == 1)
        return false;

    return count_3D_SBRC_nodes(nodeData) >= 2;
}

bool NodeFactory::use_CS_3D_RC(NodeMetaData& nodeData)
{
    // TODO: SBCC hasn't worked for inner batch (i/oDist == 1)
    if(nodeData.iDist == 1 || nodeData.oDist == 1)
        return false;

    try
    {
        // Check the C part.
        // The first R is built recursively with 2D_FFT, leave the check part to themselves
        auto krn = function_pool::get_kernel(
            fpkey(nodeData.length[2], nodeData.precision, CS_KERNEL_STOCKHAM_BLOCK_CC));

        // hack for this special case
        // this size is rejected by the following conservative threshold (#-elems)
        // however it can use 3D_RC and get much better performance
        std::vector<size_t> special_case{56, 336, 336};
        if(nodeData.length == special_case && nodeData.precision == rocfft_precision_double)
            return true;

        // x-dim should be >= the blockwidth, or it might perform worse..
        if(nodeData.length[0] < krn.batches_per_block)
            return false;
        // we don't want a too-large 3D block, sbcc along z-dim might be bad
        if((nodeData.length[0] * nodeData.length[1] * nodeData.length[2]) >= (128 * 128 * 128))
            return false;

        // Peek the first child
        // Give up if 1st child is 2D_RTRT (means the poor RTRT_C),
        NodeMetaData child0 = nodeData;
        child0.length       = nodeData.length;
        child0.dimension    = 2;
        auto childScheme    = DecideNodeScheme(child0, nullptr);
        if(childScheme == CS_2D_RTRT)
            return false;

        // if we are here, the 2D sheme is either
        // 2D_SINGLE+CC (2 kernels) or 2D_RC+CC (3 kernels),
        assert(childScheme == CS_KERNEL_2D_SINGLE || childScheme == CS_2D_RC);
        return true;
    }
    catch(...)
    {
        return false;
    }

    return false;
}