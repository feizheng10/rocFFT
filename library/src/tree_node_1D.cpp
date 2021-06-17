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

#include "tree_node_1D.h"
#include "function_pool.h"
#include "node_factory.h"
#include "radix_table.h"

/*****************************************************
 * L1D_TRTRT  *
 *****************************************************/
void TRTRT1DNode::BuildTree_internal()
{
    size_t lenFactor1 = length.back();
    size_t lenFactor0 = length[0] / lenFactor1;
    assert(lenFactor0 * lenFactor1 == length[0]);
    length.pop_back();

    // first transpose
    auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans1Plan->length.push_back(lenFactor0);
    trans1Plan->length.push_back(lenFactor1);
    trans1Plan->dimension = 2;
    for(size_t index = 1; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }

    // first row fft
    NodeMetaData row1PlanData(this);
    row1PlanData.length.push_back(lenFactor1);
    row1PlanData.length.push_back(lenFactor0);
    row1PlanData.dimension = 1;
    for(size_t index = 1; index < length.size(); index++)
    {
        row1PlanData.length.push_back(length[index]);
    }
    auto row1Plan     = NodeFactory::CreateExplicitNode(row1PlanData, this);
    row1Plan->large1D = 0;
    row1Plan->RecursiveBuildTree();

    // second transpose
    auto trans2Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans2Plan->length.push_back(lenFactor1);
    trans2Plan->length.push_back(lenFactor0);
    trans2Plan->dimension = 2;
    trans2Plan->large1D   = length[0];
    for(size_t index = 1; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }

    // second row fft
    auto row2Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM, this);
    row2Plan->length.push_back(lenFactor0);
    row2Plan->length.push_back(lenFactor1);
    row2Plan->dimension = 1;
    for(size_t index = 1; index < length.size(); index++)
    {
        row2Plan->length.push_back(length[index]);
    }

    // third transpose
    auto trans3Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans3Plan->length.push_back(lenFactor0);
    trans3Plan->length.push_back(lenFactor1);
    trans3Plan->dimension = 2;
    for(size_t index = 1; index < length.size(); index++)
    {
        trans3Plan->length.push_back(length[index]);
    }

    // TRTRT
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(row1Plan));
    childNodes.emplace_back(std::move(trans2Plan));
    childNodes.emplace_back(std::move(row2Plan));
    childNodes.emplace_back(std::move(trans3Plan));
}

void TRTRT1DNode::AssignParams_internal()
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

    row1Plan->AssignParams();

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

void TRTRT1DNode::AssignBuffers_internal(TraverseState&   state,
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
            childNodes[1]->AssignBuffers(state, flipIn, flipOut, obOutBuf);
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
            childNodes[1]->AssignBuffers(state, flipIn, flipOut, obOutBuf);
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

/*****************************************************
 * L1D_CC  *
 *****************************************************/
void CC1DNode::BuildTree_internal()
{
    size_t lenFactor1 = length.back();
    size_t lenFactor0 = length[0] / lenFactor1;
    assert(lenFactor0 * lenFactor1 == length[0]);
    length.pop_back();

    //  Note:
    //  The kernel CS_KERNEL_STOCKHAM_BLOCK_CC and CS_KERNEL_STOCKHAM_BLOCK_RC
    //  are only enabled for outplace for now. Check more details in generator.file.cpp,
    //  and in generated kernel_lunch_single_large.cpp.h

    // first plan, column-to-column
    auto col2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D = length[0];
    if(function_pool::has_SBCC_kernel(lenFactor1, precision))
    {
        // decompose the large twd table for L1D_CC
        // exclude some exceptions that don't get benefit from 3-step LargeTwd (set in FFTKernel)
        auto krn
            = function_pool::get_kernel(fpkey(lenFactor1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
        col2colPlan->largeTwd3Steps = krn.use_3steps_large_twd;
        col2colPlan->largeTwdBase   = large_twiddle_base(length[0], col2colPlan->largeTwd3Steps);
    }
    col2colPlan->length.push_back(lenFactor1);
    col2colPlan->length.push_back(lenFactor0);
    col2colPlan->dimension = 1;
    for(size_t index = 1; index < length.size(); index++)
    {
        col2colPlan->length.push_back(length[index]);
    }

    // second plan, row-to-column
    auto row2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_RC, this);
    row2colPlan->length.push_back(lenFactor0);
    row2colPlan->length.push_back(lenFactor1);
    row2colPlan->dimension = 1;
    for(size_t index = 1; index < length.size(); index++)
    {
        row2colPlan->length.push_back(length[index]);
    }

    // CC , RC
    childNodes.emplace_back(std::move(col2colPlan));
    childNodes.emplace_back(std::move(row2colPlan));
}

void CC1DNode::AssignParams_internal()
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

void CC1DNode::AssignBuffers_internal(TraverseState&   state,
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

/*****************************************************
 * L1D_CRT  *
 *****************************************************/
void CRT1DNode::BuildTree_internal()
{
    size_t lenFactor1 = length.back();
    size_t lenFactor0 = length[0] / lenFactor1;
    assert(lenFactor0 * lenFactor1 == length[0]);
    length.pop_back();

    // first plan, column-to-column
    auto col2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D = length[0];
    if(function_pool::has_SBCC_kernel(lenFactor1, precision))
    {
        // decompose the large twd table for L1D_CRT
        // exclude some exceptions that don't get benefit from 3-step LargeTwd (set in FFTKernel)
        auto kernel
            = function_pool::get_kernel(fpkey(lenFactor1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
        col2colPlan->largeTwd3Steps = kernel.use_3steps_large_twd;
        col2colPlan->largeTwdBase   = large_twiddle_base(length[0], col2colPlan->largeTwd3Steps);
    }
    col2colPlan->length.push_back(lenFactor1);
    col2colPlan->length.push_back(lenFactor0);
    col2colPlan->dimension = 1;
    for(size_t index = 1; index < length.size(); index++)
    {
        col2colPlan->length.push_back(length[index]);
    }

    // second plan, row-to-row
    auto row2rowPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM, this);
    row2rowPlan->length.push_back(lenFactor0);
    row2rowPlan->length.push_back(lenFactor1);
    row2rowPlan->dimension = 1;
    for(size_t index = 1; index < length.size(); index++)
    {
        row2rowPlan->length.push_back(length[index]);
    }

    // third plan, transpose
    auto transPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    transPlan->length.push_back(lenFactor0);
    transPlan->length.push_back(lenFactor1);
    transPlan->dimension = 2;
    for(size_t index = 1; index < length.size(); index++)
    {
        transPlan->length.push_back(length[index]);
    }

    // CRT
    childNodes.emplace_back(std::move(col2colPlan));
    childNodes.emplace_back(std::move(row2rowPlan));
    childNodes.emplace_back(std::move(transPlan));
}

void CRT1DNode::AssignParams_internal()
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

void CRT1DNode::AssignBuffers_internal(TraverseState&   state,
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

// Leaf Node..
/*****************************************************
 * CS_KERNEL_STOCKHAM  *
 *****************************************************/
void Stockham1DNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    // get working group size and number of transforms
    size_t batch_accum = batch;
    for(size_t j = 1; j < length.size(); j++)
        batch_accum *= length[j];

    auto kernel = function_pool::get_kernel(fpkey(length[0], precision));
    fnPtr       = kernel.device_function;

    if(ebtype != EmbeddedType::NONE)
        lds_padding = 1;

    // generated by new generator
    if(kernel.threads_per_block > 0)
    {
        gp.b_x   = (batch_accum + kernel.batches_per_block - 1) / kernel.batches_per_block;
        gp.tpb_x = kernel.threads_per_block;
        lds      = (length[0] + lds_padding) * kernel.batches_per_block;
        bwd      = kernel.batches_per_block;
    }
    else
    {
        size_t workGroupSize;
        size_t numTransforms;
        GetWGSAndNT(length[0], workGroupSize, numTransforms);
        gp.b_x   = (batch_accum % numTransforms) ? 1 + (batch_accum / numTransforms)
                                                 : (batch_accum / numTransforms);
        gp.tpb_x = workGroupSize;
        bwd      = numTransforms;
    }

    return;
}

bool Stockham1DNode::CreateTwiddleTableResource()
{
    twd_attach_2N = (ebtype != EmbeddedType::NONE);
    return LeafNode::CreateTwiddleTableResource();
}

/*****************************************************
 * SBCC  *
 *****************************************************/
void SBCCNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    auto kernel = function_pool::get_kernel(fpkey(length[0], precision, scheme));
    fnPtr       = kernel.device_function;
    gp.b_x      = ((length[1]) - 1) / kernel.batches_per_block + 1;
    // repeat for higher dimensions + batch
    gp.b_x *= std::accumulate(length.begin() + 2, length.end(), batch, std::multiplies<size_t>());
    gp.tpb_x = kernel.threads_per_block;
    lds      = length[0] * kernel.batches_per_block;
    bwd      = kernel.batches_per_block;

    return;
}

/*****************************************************
 * SBRC  *
 *****************************************************/
void SBRCNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    fnPtr = function_pool::get_function(fpkey(length[0], precision, scheme));
    GetBlockComputeTable(length[0], bwd, wgs, lds);
    gp.b_x = (length[1]) / bwd;
    // repeat for higher dimensions + batch
    gp.b_x *= std::accumulate(length.begin() + 2, length.end(), batch, std::multiplies<size_t>());
    gp.tpb_x = wgs;

    return;
}

/*****************************************************
 * SBCR  *
 *****************************************************/
void SBCRNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    auto kernel = function_pool::get_kernel(fpkey(length[0], precision, scheme));
    fnPtr       = kernel.device_function;
    gp.b_x      = ((length[1]) - 1) / kernel.batches_per_block + 1;
    // repeat for higher dimensions + batch
    gp.b_x *= std::accumulate(length.begin() + 2, length.end(), batch, std::multiplies<size_t>());
    gp.tpb_x = kernel.threads_per_block;
    lds      = length[0] * kernel.batches_per_block;
    bwd      = kernel.batches_per_block;

    return;
}