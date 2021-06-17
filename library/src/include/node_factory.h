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

#ifndef TREE_NODE_FACTORY_H
#define TREE_NODE_FACTORY_H

#include "tree_node.h"

class NodeFactory
{
private:
    // Maps from length[0] to divLength1 for 1D transforms in
    // single and double precision using block computing.
    typedef std::map<size_t, size_t> Map1DLength;
    static const Map1DLength         map1DLengthSingle;
    static const Map1DLength         map1DLengthDouble;

public:
    // Create node (user level) using this function
    // static std::unique_ptr<TreeNode> CreateNode(TreeNode* parentNode = nullptr);
    static std::unique_ptr<TreeNode> CreateNodeFromScheme(ComputeScheme s,
                                                          TreeNode*     parent = nullptr);
    static std::unique_ptr<TreeNode> CreateExplicitNode(NodeMetaData& nodeData, TreeNode* parent);

    // Decide scheme from the node meta node
    static ComputeScheme DecideNodeScheme(NodeMetaData& nodeData, TreeNode* parent);
    static ComputeScheme DecideRealScheme(NodeMetaData& nodeData);
    static ComputeScheme Decide1DScheme(NodeMetaData& nodeData);
    static ComputeScheme Decide2DScheme(NodeMetaData& nodeData);
    static ComputeScheme Decide3DScheme(NodeMetaData& nodeData);

    // determine function:
    static bool use_CS_2D_SINGLE(NodeMetaData& nodeData); // using scheme CS_KERNEL_2D_SINGLE or not
    static bool use_CS_2D_RC(NodeMetaData& nodeData); // using scheme CS_2D_RC or not
    static bool use_CS_3D_BLOCK_RC(NodeMetaData& nodeData);
    static bool use_CS_3D_RC(NodeMetaData& nodeData);
    // how many SBRC kernels can we put into a 3D transform?
    static size_t count_3D_SBRC_nodes(NodeMetaData& nodeData);
};
#endif // TREE_NODE_FACTORY_H
