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

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "../../../shared/gpubuf.h"
#include "../device/kernels/callback.h"
#include "../device/kernels/common.h"
#include "kargs.h"
#include "rocfft_ostream.hpp"
#include "twiddles.h"
#include <hip/hip_runtime_api.h>

enum OperatingBuffer
{
    OB_UNINIT,
    OB_USER_IN,
    OB_USER_OUT,
    OB_TEMP,
    OB_TEMP_CMPLX_FOR_REAL,
    OB_TEMP_BLUESTEIN,
};

enum ComputeScheme
{
    CS_NONE,
    CS_KERNEL_STOCKHAM,
    CS_KERNEL_STOCKHAM_BLOCK_CC,
    CS_KERNEL_STOCKHAM_BLOCK_RC,
    CS_KERNEL_STOCKHAM_BLOCK_CR, // not implemented yet
    CS_KERNEL_TRANSPOSE,
    CS_KERNEL_TRANSPOSE_XY_Z,
    CS_KERNEL_TRANSPOSE_Z_XY,

    CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
    CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
    CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY,

    CS_REAL_TRANSFORM_USING_CMPLX,
    CS_KERNEL_COPY_R_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_HERM,
    CS_KERNEL_COPY_HERM_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_R,

    CS_REAL_TRANSFORM_EVEN,
    CS_KERNEL_R_TO_CMPLX,
    CS_KERNEL_R_TO_CMPLX_TRANSPOSE,
    CS_KERNEL_CMPLX_TO_R,
    CS_KERNEL_TRANSPOSE_CMPLX_TO_R,
    CS_REAL_2D_EVEN,
    CS_REAL_3D_EVEN,
    CS_KERNEL_APPLY_CALLBACK,

    CS_BLUESTEIN,
    CS_KERNEL_CHIRP,
    CS_KERNEL_PAD_MUL,
    CS_KERNEL_FFT_MUL,
    CS_KERNEL_RES_MUL,

    CS_L1D_TRTRT,
    CS_L1D_CC,
    CS_L1D_CRT,

    CS_2D_STRAIGHT, // not implemented yet
    CS_2D_RTRT,
    CS_2D_RC,
    CS_KERNEL_2D_STOCKHAM_BLOCK_CC, // not implemented yet
    CS_KERNEL_2D_SINGLE,

    CS_3D_STRAIGHT, // not implemented yet
    CS_3D_TRTRTR,
    CS_3D_RTRT,
    CS_3D_BLOCK_RC,
    CS_3D_BLOCK_CR,
    CS_3D_RC,
    CS_KERNEL_3D_STOCKHAM_BLOCK_CC, // not implemented yet
    CS_KERNEL_3D_SINGLE // not implemented yet
};

enum NodeType
{
    NT_UNDEFINED, // un init
    NT_INTERNAL, // an internal node contains childrens
    NT_LEAF, // a leaf node represents a kernel and has no childrens
};

std::string PrintScheme(ComputeScheme cs);
std::string PrintOperatingBuffer(const OperatingBuffer ob);
std::string PrintOperatingBufferCode(const OperatingBuffer ob);
std::string PrintSBRCTransposeType(const SBRC_TRANSPOSE_TYPE ty);

typedef void (*DevFnCall)(const void*, void*);

struct GridParam
{
    unsigned int b_x, b_y, b_z; // in HIP, the data type of dimensions of work
    // items, work groups is unsigned int
    unsigned int tpb_x, tpb_y, tpb_z;
    unsigned int lds_bytes; // dynamic LDS allocation size

    GridParam()
        : b_x(1)
        , b_y(1)
        , b_z(1)
        , tpb_x(1)
        , tpb_y(1)
        , tpb_z(1)
        , lds_bytes(0)
    {
    }
};

static bool is_diagonal_sbrc_3D_length(size_t len)
{
    // SBRC diagonal-transpose dimensions are currently 128, 256
    return len == 128 || len == 256;
}

static bool is_cube_size(const std::vector<size_t>& length)
{
    return length.size() == 3 && length[0] == length[1] && length[1] == length[2];
}

class TreeNode;

// The mininal tree node data needed to decide the scheme
struct NodeMetaData
{
    size_t                  batch     = 1;
    size_t                  dimension = 1;
    std::vector<size_t>     length;
    std::vector<size_t>     inStride, outStride;
    size_t                  iDist = 0, oDist = 0;
    size_t                  iOffset = 0, oOffset = 0;
    int                     direction    = -1;
    rocfft_result_placement placement    = rocfft_placement_inplace;
    rocfft_precision        precision    = rocfft_precision_single;
    rocfft_array_type       inArrayType  = rocfft_array_type_unset;
    rocfft_array_type       outArrayType = rocfft_array_type_unset;

    NodeMetaData(TreeNode* refNode);
};

class TreeNode
{
    friend class NodeFactory;

protected:
    TreeNode(TreeNode* p)
        : parent(p)
    {
        if(p != nullptr)
        {
            precision = p->precision;
            batch     = p->batch;
            direction = p->direction;
        }
    }

    // Compute the large twd decomposition base
    size_t large_twiddle_base(size_t length, bool use3Steps);

public:
    // node type: internal node or leaf node, or un-defined (un-init)
    NodeType nodeType = NT_UNDEFINED;

    // Batch size
    size_t batch = 1;

    // Transform dimension - note this can be different from data dimension, user
    // provided
    size_t dimension = 1;

    // Length of the FFT in each dimension, internal value
    std::vector<size_t> length;

    // Stride of the FFT in each dimension
    std::vector<size_t> inStride, outStride;

    // Distance between consecutive batch members:
    size_t iDist = 0, oDist = 0;

    // Offsets to start of data in buffer:
    size_t iOffset = 0, oOffset = 0;

    // Direction of the transform (-1: forward, +1: inverse)
    int direction = -1;

    // The number of padding at the end of each row in lds
    unsigned int lds_padding = 0;

    // Data format parameters:
    rocfft_result_placement placement    = rocfft_placement_inplace;
    rocfft_precision        precision    = rocfft_precision_single;
    rocfft_array_type       inArrayType  = rocfft_array_type_unset;
    rocfft_array_type       outArrayType = rocfft_array_type_unset;

    // Extra twiddle multiplication for large 1D
    size_t large1D = 0;
    // decompose large twiddle to product of 256(8) or 128(7) or 64(6)...or 16(4)
    // default is 8, and sbcc could be dynamically decomposed
    size_t largeTwdBase = 8;
    // flag indicating if using the 3-step decomp. for large twiddle? (16^3, 32^3, 64^3)
    // if false, always use 8 as the base (256*256*256....)
    bool largeTwd3Steps = false;

    // embedded C2R/R2C pre/post processing
    EmbeddedType ebtype = EmbeddedType::NONE;

    // Tree structure:
    // non-owning pointer to parent node, may be null
    TreeNode* parent = nullptr;
    // owned pointers to children
    std::vector<std::unique_ptr<TreeNode>> childNodes;

    // FIXME: document
    ComputeScheme   scheme = CS_NONE;
    OperatingBuffer obIn = OB_UNINIT, obOut = OB_UNINIT;

    // FIXME: document
    size_t lengthBlue = 0;

    // Device pointers:
    gpubuf           twiddles;
    gpubuf           twiddles_large;
    gpubuf_t<size_t> devKernArg;

    // callback parameters
    UserCallbacks callbacks;

    // comments inserted by optimization passes to explain changes done
    // to the node
    std::vector<std::string> comments;

public:
    // Disallow copy constructor:
    TreeNode(const TreeNode&) = delete;

    // for the derived class
    virtual ~TreeNode() = default;

    // Disallow assignment operator:
    TreeNode& operator=(const TreeNode&) = delete;

    // Copy data from another node (to a fused node)
    void CopyNodeData(const TreeNode& srcNode);

    // Copy data from the NodeMetaData (after deciding scheme)
    void CopyNodeData(const NodeMetaData& data);

    bool isRootNode();
    bool isLeafNode();

    virtual void RecursiveBuildTree(); // Main tree builder: override by child
    virtual void SanityCheck();

    // fusing CS_KERNEL_STOCKHAM and CS_KERNEL_TRANSPOSE_Z_XY ?
    bool fuse_CS_KERNEL_TRANSPOSE_Z_XY();

    // State maintained while traversing the tree.
    //
    // Preparation and execution of the tree basically involves a
    // depth-first traversal.  At each step, the logic working on a
    // node could want to know details of:
    //
    // 1. the node itself (i.e. this)
    // 2. the node's parent (i.e. this->parent), if present
    // 3. the most recently traversed leaf node, which may be:
    //    - not present, or
    //    - an earlier sibling of this node, or
    //    - the last leaf visited from some other parent
    // 4. the root node's input/output parameters
    //
    // The TraverseState struct stores 3 and 4.
    struct TraverseState;
    // Assign the input buffer for this kernel
    void SetInputBuffer(TraverseState& state);

    // Buffer assignment:
    virtual void AssignBuffers(TraverseState&   state,
                               OperatingBuffer& flipIn,
                               OperatingBuffer& flipOut,
                               OperatingBuffer& obOutBuf);

    // Set placement variable and in/out array types
    virtual void TraverseTreeAssignPlacementsLogicA(rocfft_array_type rootIn,
                                                    rocfft_array_type rootOut);

    // Set strides and distances:
    virtual void AssignParams();

    // Determine work memory requirements:
    void TraverseTreeCollectLeafsLogicA(std::vector<TreeNode*>& seq,
                                        size_t&                 tmpBufSize,
                                        size_t&                 cmplxForRealSize,
                                        size_t&                 blueSize,
                                        size_t&                 chirpSize);

    // Output plan information for debug purposes:
    void Print(rocfft_ostream& os = rocfft_cout, int indent = 0) const;

    // logic B - using in-place transposes, todo
    //void RecursiveBuildTreeLogicB();

    void RecursiveRemoveNode(TreeNode* node);

    // insert a newNode before the node "pos"
    void RecursiveInsertNode(TreeNode* pos, std::unique_ptr<TreeNode>& newNode);

    virtual bool CreateDevKernelArgs()                                     = 0;
    virtual bool CreateTwiddleTableResource()                              = 0;
    virtual void SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp) = 0;

protected:
    virtual void BuildTree_internal() = 0;
    virtual void AssignBuffers_internal(TraverseState&   state,
                                        OperatingBuffer& flipIn,
                                        OperatingBuffer& flipOut,
                                        OperatingBuffer& obOutBuf)
        = 0;
    virtual void AssignParams_internal() = 0;
};

class InternalNode : public TreeNode
{
    friend class NodeFactory;

protected:
    InternalNode(TreeNode* p)
        : TreeNode(p)
    {
        nodeType = NT_INTERNAL;
    }

    virtual bool CreateDevKernelArgs()
    {
        throw std::runtime_error("Shouldn't call CreateDevKernelArgs in a non-LeafNode");
        return false;
    }

    virtual bool CreateTwiddleTableResource()
    {
        throw std::runtime_error("Shouldn't call CreateTwiddleTableResource in a non-LeafNode");
        return false;
    }

    virtual void SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp)
    {
        throw std::runtime_error("Shouldn't call SetupGridParamAndFuncPtr in a non-LeafNode");
    }
};

class LeafNode : public InternalNode
{
    friend class NodeFactory;

protected:
    LeafNode(TreeNode* p, ComputeScheme s)
        : InternalNode(p)
    {
        nodeType = NT_LEAF;
        scheme   = s;
    }

    bool                externalKernel = false;
    bool                need_twd_table = false;
    bool                twd_no_radices = false;
    bool                twd_attach_2N  = false;
    std::vector<size_t> kernelFactors  = {};
    size_t              bwd            = 1; // bwd, wgs, lds are for grid param lds_bytes
    size_t              wgs            = 0;
    size_t              lds            = 0;

    void           BuildTree_internal() final {} // nothing to do in leaf node
    void           AssignBuffers_internal(TraverseState&   state,
                                          OperatingBuffer& flipIn,
                                          OperatingBuffer& flipOut,
                                          OperatingBuffer& obOutBuf) override;
    void           AssignParams_internal() final {} // nothing to do in leaf node
    bool           CreateLargeTwdTable();
    virtual size_t GetTwiddleTableLength();
    virtual void   KernelCheck();
    virtual void   SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) = 0;

public:
    void SanityCheck() override;
    bool CreateDevKernelArgs() override;
    bool CreateTwiddleTableResource() override;
    void SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp) override;
};

/*****************************************************
 * CS_KERNEL_TRANSPOSE
 * CS_KERNEL_TRANSPOSE_XY_Z
 * CS_KERNEL_TRANSPOSE_Z_XY
 *****************************************************/
class TransposeNode : public LeafNode
{
    friend class NodeFactory;

protected:
    TransposeNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;
};

struct ExecPlan
{
    // shared pointer allows for ExecPlans to be copyable
    std::shared_ptr<TreeNode> rootPlan;

    // non-owning pointers to the leaf-node children of rootPlan, which
    // are the nodes that do actual work
    std::vector<TreeNode*> execSeq;

    std::vector<DevFnCall> devFnCall;
    std::vector<GridParam> gridParam;

    hipDeviceProp_t deviceProp;

    // these sizes count in complex elements
    size_t workBufSize      = 0;
    size_t tmpWorkBufSize   = 0;
    size_t copyWorkBufSize  = 0;
    size_t blueWorkBufSize  = 0;
    size_t chirpWorkBufSize = 0;

    size_t WorkBufBytes(size_t base_type_size)
    {
        // base type is the size of one real, work buf counts in
        // complex numbers
        return workBufSize * 2 * base_type_size;
    }
};

void ProcessNode(ExecPlan& execPlan);
void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan);

#endif // TREE_NODE_H
