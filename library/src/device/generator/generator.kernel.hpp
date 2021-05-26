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

#pragma once
#if !defined(_generator_kernel_H)
#define _generator_kernel_H
#include "../../include/plan.h"
#include "../kernels/common.h"
#include "generator.param.h"
#include "generator.pass.hpp"
#include "generator.stockham.h"
#include <iterator>
#include <list>
#include <stdio.h>

// FFT Stockham Autosort Method
//
//   Each pass does one digit reverse in essence. Hence by the time all passes
//   are done, complete
//   digit reversal is done and output FFT is in correct order. Intermediate
//   FFTs are stored in natural order,
//   which is not the case with basic Cooley-Tukey algorithm. Natural order in
//   intermediate data makes it
//   convenient for stitching together passes with different radices.
//
//  Basic FFT algorithm:
//
//        Pass loop
//        {
//            Outer loop
//            {
//                Inner loop
//                {
//                }
//            }
//        }
//
//  The sweeps of the outer and inner loop resemble matrix indexing, this matrix
//  changes shape with every pass as noted below
//
//   FFT pass diagram (radix 2)
//
//                k            k+R                                    k
//            * * * * * * * * * * * * * * * *                     * * * * * * *
//            *
//            *   |             |           *                     *   | *
//            *   |             |           *                     *   | *
//            *   |             |           * LS        -->       *   | *
//            *   |             |           *                     *   | *
//            *   |             |           *                     *   | *
//            * * * * * * * * * * * * * * * *                     *   | *
//                         RS                                     *   | * L
//                                                                *   | *
//                                                                *   | *
//                                                                *   | *
//                                                                *   | *
//                                                                *   | *
//                                                                *   | *
//                                                                *   | *
//                                                                * * * * * * *
//                                                                *
//                                                                       R
//
//
//    With every pass, the matrix doubles in height and halves in length
//
//
//  N = 2^T = Length of FFT
//  q = pass loop index
//  k = outer loop index = (0 ... R-1)
//  j = inner loop index = (0 ... LS-1)
//
//  Tables shows how values change as we go through the passes
//
//    q | LS   |  R   |  L  | RS
//   ___|______|______|_____|___
//    0 |  1   | N/2  |  2  | N
//    1 |  2   | N/4  |  4  | N/2
//    2 |  4   | N/8  |  8  | N/4
//    . |  .   | .    |  .  | .
//  T-1 |  N/2 | 1    |  N  | 2
//
//
//   Data Read Order
//     Radix 2: k*LS + j, (k+R)*LS + j
//     Radix 3: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j
//     Radix 4: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j
//     Radix 5: k*LS + j, (k+R)*LS + j, (k+2R)*LS + j, (k+3R)*LS + j, (k+4R)*LS
//     + j
//
//   Data Write Order
//       Radix 2: k*L + j, k*L + j + LS
//       Radix 3: k*L + j, k*L + j + LS, k*L + j + 2*LS
//       Radix 4: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS
//       Radix 5: k*L + j, k*L + j + LS, k*L + j + 2*LS, k*L + j + 3*LS, k*L + j
//       + 4*LS
//

namespace StockhamGenerator
{

    class KernelCoreSpecs
    {

        typedef typename std::map<size_t, SpecRecord> SpecTable;
        SpecTable                                     specTable;
        std::vector<SpecRecord>                       specRecord;
        size_t                                        tableLength;

    public:
        KernelCoreSpecs()
        {
            specRecord = GetRecord();
            // reform an array to a map, the table is store in ../include/radix_table.h
            tableLength = specRecord.size();
            for(size_t i = 0; i < tableLength; i++)
                specTable[specRecord[i].length] = specRecord[i];
        }

        std::vector<size_t> GetRadices(size_t length)
        {

            std::vector<size_t> radices;

            // printf("tableLength=%d\n", tableLength);
            for(int i = 0; i < tableLength; i++)
            {

                if(length == specRecord[i].length)
                { // if find the matched size

                    size_t numPasses = specRecord[i].numPasses;
                    // printf("numPasses=%d, table item %d \n", numPasses, i);
                    for(int j = 0; j < numPasses; j++)
                    {
                        radices.push_back((specRecord[i].radices)[j]);
                    }
                    break;
                }
            }
            return radices;
        }

        // get working group size and number of transforms
        void GetWGSAndNT(size_t length, size_t& workGroupSize, size_t& numTransforms) const
        {
            workGroupSize = 0;
            numTransforms = 0;

            typename SpecTable::const_iterator it = specTable.find(length);
            if(it != specTable.end())
            {
                workGroupSize = it->second.workGroupSize;
                numTransforms = it->second.numTransforms;
            }
            // if not in the predefined table, then use the algorithm to determine
            if(workGroupSize == 0)
            {
                DetermineSizes(length, workGroupSize, numTransforms);
            }
        }
    };

    // FFT kernel generator
    // Kernel calls butterfly and pass
    template <rocfft_precision PR>
    class Kernel
    {
    public:
        size_t length; // Length of FFT
        size_t workGroupSize; // Work group size
        size_t cnPerWI; // complex numbers per work-item

        size_t numTrans; // The maximum number of FFT-transforms per work-group,
        // internal varaible
        size_t                workGroupSizePerTrans; // Work group subdivision per transform
        size_t                numPasses; // Number of FFT passes
        std::vector<size_t>   radices; // Base radix at each pass
        std::vector<Pass<PR>> passes; // Array of pass objects

        bool halfLds; // LDS used to store one component (either real or imaginary) at
        // a time
        // for passing intermediate data between the passes, if this is set true
        // then each pass-function should accept same set of registers

        bool linearRegs; // scalar registers

        // Future optimization ideas
        // bool limitRegs;                            // TODO: Incrementally write to
        // LDS, thereby using same set of registers for more than 1 butterflies
        // bool combineReadTwMul;                    // TODO: Combine reading into
        // registers and Twiddle multiply

        bool r2c2r = false; // real to complex or complex to real transform
        bool r2c = false, c2r = false;
        bool rcFull   = false;
        bool rcSimple = false;

        bool blockCompute; // When we have to compute FFT in blocks (either read or
        // write is along columns, optimization in radix-2 FFTs)
        size_t           blockWidth, blockWGS, blockLDS, blockLdsRowPadding;
        BlockComputeType blockComputeType;

        bool realSpecial; // controls related to large1D real FFTs.

        const FFTKernelGenKeyParams params; // key params

        std::string name_suffix; // use to specify kernel & device functions names to
        // avoid naming conflict.

        bool NeedsLargeTwiddles() // sbcc kernel needs large twiddle table parameter
        {
            return (blockCompute && blockComputeType == BCT_C2C) ? true : false;
        }

        // // FIXME: after we have tuning framework, we don't have to block 81 and 200
        // bool MoveLTWDtoLDS()
        // {
        //     // NOTE: need to be consistent with kernel_generator.py:
        //     // function.meta.use_3steps_large_twd -> large 1D -> FFTKernel
        //     return (length != 81) && (length != 200);
        // }

    private:
        inline std::string IterRegs(const std::string& pfx, bool initComma = true)
        {
            std::string str = "";

            if(linearRegs)
            {
                if(initComma)
                    str += ", ";

                for(size_t i = 0; i < cnPerWI; i++)
                {
                    if(i != 0)
                        str += ", ";
                    str += pfx;
                    str += "R";
                    str += std::to_string(i);
                }
            }

            return str;
        }

        inline bool IsGroupedReadWritePossible() // TODO
        {
            bool possible = true;

            if(r2c2r)
                return false;

            if(realSpecial)
                return false;

            for(size_t i = 0; i < params.fft_DataDim - 1; i++) // if it is power of 2
            {
                if(params.fft_N[i] % 2)
                {
                    possible = false;
                    break;
                }
            }

            return possible;
        }

        /*
        OffsetCalcBlockCompute when blockCompute is set as true
        it calculates the offset to the memory

        offset_name
            can be ioOffset, iOffset or oOffset, they are size_t type
        stride_name
            can be stride_in or stride_out, they are vector<size_t> type
        output
            if true, offset_name2, stride_name2 are enabled
        else not enabled
        */

        // since it is batching process mutiple matrices by default, calculate the
        // offset block
        inline std::string OffsetCalcBlockCompute(const std::string& offset_name1,
                                                  const std::string& stride_name1,
                                                  const std::string& offset_name2,
                                                  const std::string& stride_name2,
                                                  bool               input,
                                                  bool               output)
        {
            std::string str;

            if(blockComputeType == BCT_R2C)
            {
                str += "\t// for SBRC 3D kernels: number of blocks needed to deal with one 3D "
                       "batch\n";
                str += "\tunsigned int blocks_per_batch;\n";
                str += "\tif(Tsbrc == SBRC_3D_FFT_TRANS_XY_Z)\n";
                str += "\t\tblocks_per_batch = lengths[1] * ((lengths[2] + "
                       + std::to_string(blockWidth) + " - 1) / " + std::to_string(blockWidth)
                       + ");\n";
                str += "\telse if(Tsbrc == SBRC_3D_FFT_TRANS_Z_XY || Tsbrc == "
                       "SBRC_3D_FFT_ERC_TRANS_Z_XY)\n";
                str += "\t\tblocks_per_batch = lengths[2] * ((lengths[1] + "
                       + std::to_string(blockWidth) + " - 1) / " + std::to_string(blockWidth)
                       + ");\n";

                str += "\t// index of this block in the current 3D batch\n";
                str += "\tunsigned int block_in_batch = batch % blocks_per_batch;\n";
                str += "\n\tif(Tsbrc == SBRC_2D)\n";
            }
            str += "\t{\n";

            str += "\t\tunsigned int counter_mod = batch;\n";

            std::string loop;
            loop += "\t\tfor(int i = dim; i>2; i--){\n"; // dim is a runtime variable
            loop += "\t\t\tint currentLength = 1;\n";
            loop += "\t\t\tfor(int j=2; j<i; j++){\n";
            loop += "\t\t\t\tcurrentLength *= lengths[j];\n";
            loop += "\t\t\t}\n";
            loop += "\t\t\tcurrentLength *= (lengths[1]/" + std::to_string(blockWidth) + ");\n";
            loop += "\n";
            loop += "\t\t\t" + offset_name1 + " += (counter_mod/currentLength)*" + stride_name1
                    + "[i];\n";
            if(output == true)
                loop += "\t\t\t" + offset_name2 + " += (counter_mod/currentLength)*" + stride_name2
                        + "[i];\n";
            loop += "\t\t\tcounter_mod = (counter_mod % currentLength); \n";
            loop += "\t\t}\n";

            loop += "\n\t\t// We handle a 2D tile block with one work-group threads.\n";
            loop += "\t\t// In the below, '_x' moves along the fast dimension of the tile.\n";

            std::string sub_string = "(lengths[1]/" + std::to_string(blockWidth)
                                     + ")"; // in FFT it is how many unrolls
            loop += "\t\tunsigned int tileIdx_x, tileIdx_y, tileOffset_x, tileOffset_y;\n";
            loop += "\n\t\t// Calc input tile start offset\n";
            loop += "\t\ttileIdx_y\t\t= (counter_mod / " + sub_string + ");\n";
            loop += "\t\ttileOffset_y\t= " + stride_name1 + "[2];\n";
            loop += "\t\ttileIdx_x\t\t= (counter_mod % " + sub_string + ");\n";

            if(blockComputeType == BCT_R2C) // only for input
                loop += "\t\ttileOffset_x\t= " + std::to_string(blockWidth) + "*lengths[0];\n";
            else
                // the most inner part of offset calc needs to count stride[1] for SBCC
                loop += "\t\ttileOffset_x\t= " + std::to_string(blockWidth) + " * " + stride_name1
                        + "[1];\n\n";

            loop += "\t\t" + offset_name1
                    + " += tileIdx_y * tileOffset_y + tileIdx_x * tileOffset_x;\n";

            if(output == true)
            {
                loop += "\t\t// Calc output tile start offset\n";
                loop += "\t\ttileOffset_y\t= " + stride_name2 + "[2];\n";

                // the most inner part of offset calc needs to count stride[1] for block compute
                if(blockComputeType == BCT_C2R) // only for output
                    loop += "\t\ttileOffset_x\t= " + std::to_string(blockWidth) + "*lengths[0];\n";
                else
                    loop += "\t\ttileOffset_x\t= " + std::to_string(blockWidth) + " * "
                            + stride_name2 + "[1];\n\n";

                loop += "\t\t" + offset_name2
                        + " += tileIdx_y * tileOffset_y + tileIdx_x * tileOffset_x;\n";
            }
            loop += "\t}\n";
            if(blockComputeType == BCT_R2C)
            {
                loop += "\telse if(Tsbrc == SBRC_3D_FFT_TRANS_XY_Z)\n";
                loop += "\t{\n";
                loop += "\t\tunsigned int readTileIdx_x = batch % lengths[1];\n";
                loop += "\t\tunsigned int readTileIdx_y = batch % blocks_per_batch / lengths[1];\n";

                loop += "\t\tif(Ttranspose == DIAGONAL)\n";
                loop += "\t\t{\n";
                loop += "\t\t// diagonal transpose for power of 2 length\n";

                loop += "\t\tunsigned int bid = readTileIdx_x + " + std::to_string(length)
                        + " * readTileIdx_y;\n";
                loop += "\t\tunsigned int tileBlockIdx_y = bid % "
                        + std::to_string(blockWGS / blockWidth) + ";\n";
                loop += "\t\tunsigned int tileBlockIdx_x = ((bid / "
                        + std::to_string(blockWGS / blockWidth) + ") + tileBlockIdx_y) % "
                        + std::to_string(length) + ";\n";

                loop += "\t\t" + offset_name1 + " += tileBlockIdx_y * ("
                        + std::to_string(blockWidth) + " * " + stride_name1
                        + "[2]) + tileBlockIdx_x  "
                          "* "
                        + stride_name1 + "[1] + batch / blocks_per_batch * " + stride_name1
                        + "[3];\n";
                if(output)
                {
                    loop += "\t\tunsigned int writeTileIdx_x = tileBlockIdx_y;\n";
                    loop += "\t\tunsigned int writeTileIdx_y = tileBlockIdx_x;\n";
                    loop += "\t\t" + offset_name2 + " += writeTileIdx_y * " + stride_name2
                            + "[2] + writeTileIdx_x * " + std::to_string(blockWidth) + " * "
                            + stride_name2 + "[0] + batch / blocks_per_batch * " + stride_name2
                            + "[3];\n";
                }
                loop += "\t\t}\n";
                loop += "\t\telse\n";
                loop += "\t\t{\n";
                loop += "\t\t" + offset_name1 + " += readTileIdx_y * (" + std::to_string(blockWidth)
                        + " * " + stride_name1 + "[2]) + readTileIdx_x  * " + stride_name1
                        + "[1] + batch / blocks_per_batch * " + stride_name1 + "[3];\n";
                loop += "\n";
                if(output)
                {
                    loop += "\t\tunsigned int writeTileIdx_x = readTileIdx_y;\n";
                    loop += "\t\tunsigned int writeTileIdx_y = readTileIdx_x;\n";
                    loop += "\n";
                    loop += "\t\t" + offset_name2 + " += writeTileIdx_y * " + stride_name2
                            + "[2] + writeTileIdx_x * " + std::to_string(blockWidth) + " * "
                            + stride_name2 + "[0] + batch / blocks_per_batch * " + stride_name2
                            + "[3];\n";
                }
                loop += "\t\t}\n";
                loop += "\t}\n";

                loop += "\telse if(Tsbrc == SBRC_3D_FFT_TRANS_Z_XY || Tsbrc == "
                        "SBRC_3D_FFT_ERC_TRANS_Z_XY)\n";
                loop += "\t{\n";
                loop += "\t\tdim3 tgs; // tile grid size\n";
                loop += "\t\ttgs.x = 1;\n";
                loop += "\t\ttgs.y = lengths[1] * lengths[2] / " + std::to_string(blockWidth)
                        + ";\n";
                loop += "\t\tunsigned int blocks_per_batch = tgs.x * tgs.y;\n";
                loop += "\t\tunsigned int readTileIdx_x = 0; // batch % tgs.x;\n";
                loop += "\t\tunsigned int readTileIdx_y = (batch % blocks_per_batch) / tgs.x;\n";

                loop += "\t\t" + offset_name1 + " += readTileIdx_y * (" + std::to_string(blockWidth)
                        + " * " + stride_name1 + "[1]) + readTileIdx_x  * " + stride_name1
                        + "[1] + batch / blocks_per_batch * " + stride_name1 + "[3];\n";
                loop += "\n";
                if(output)
                {
                    loop += "\t\tunsigned int writeTileIdx_x = readTileIdx_y;\n";
                    loop += "\t\tunsigned int writeTileIdx_y = readTileIdx_x;\n";
                    loop += "\n";
                    loop += "\t\t" + offset_name2 + " += writeTileIdx_y * " + stride_name2
                            + "[3] + writeTileIdx_x * " + std::to_string(blockWidth) + " * "
                            + stride_name2 + "[0] + batch / blocks_per_batch * " + stride_name2
                            + "[3];\n";
                }
                loop += "\t}\n";
            }

            str += loop;
            return str;
        }

        // Contcatenate global read string for block computing
        inline std::string GlobalReadStrCat(const std::string&      global_offset,
                                            rocfft_result_placement placeness,
                                            bool                    inInterleaved)
        {
            std::string ret = "\t\t";
            if(blockComputeType == BCT_R2C)
                ret += "\t";

            if(inInterleaved)
            {
                std::string buffer = (placeness == rocfft_placement_inplace) ? "gb" : "gbIn";
                ret += "R0 = load_cb(" + buffer + ", " + global_offset
                       + ", load_cb_data, nullptr);\n";
            }
            else
            {
                ret += "R0.x = ";
                ret += (placeness == rocfft_placement_inplace) ? "gbRe" : "gbInRe";
                ret += "[" + global_offset + "];\n";
                if(blockComputeType == BCT_R2C)
                    ret += "\t";
                ret += "\t\tR0.y = ";
                ret += (placeness == rocfft_placement_inplace) ? "gbIm" : "gbInIm";
                ret += "[" + global_offset + "];\n";
            }
            return ret;
        }

        // Contcatenate global write string for block computing
        inline std::string GlobalWriteStrCat(const std::string&      global_offset,
                                             rocfft_result_placement placeness,
                                             bool                    inInterleaved)
        {
            std::string ret = "\t\t";
            if(blockComputeType == BCT_R2C)
                ret += "\t";

            if(inInterleaved)
            {
                std::string buffer = (placeness == rocfft_placement_inplace) ? "gb" : "gbOut";
                ret += "store_cb(" + buffer + ", " + global_offset
                       + ", R0, store_cb_data, nullptr );\n";
            }
            else
            {
                ret += (placeness == rocfft_placement_inplace) ? "gbRe" : "gbOutRe";
                ret += "[" + global_offset + "] = ";
                ret += "R0.x;\n\t\t";
                if(blockComputeType == BCT_R2C)
                    ret += "\t";
                ret += (placeness == rocfft_placement_inplace) ? "gbIm" : "gbOutIm";
                ret += "[" + global_offset + "] = ";
                ret += "R0.y;\n";
            }
            return ret;
        }

        /*
        OffsetCalc calculates the offset to the memory

        offset_name
            can be ioOffset, iOffset or oOffset, they are size_t type
        stride_name
            can be stride_in or stride_out, they are vector<size_t> type
        output
            if true, offset_name2, stride_name2 are enabled
            else not enabled
        */

        inline std::string OffsetCalc(const std::string& offset_name1,
                                      const std::string& stride_name1,
                                      const std::string& offset_name2,
                                      const std::string& stride_name2,
                                      bool               output,
                                      bool               rc_second_index = false)
        {
            std::string str;

            /*===========the comments assume a 16-point
             * FFT============================================*/

            // generate statement like "size_t counter_mod = batch*16 + (me/4);"
            std::string counter_mod;
            if(r2c2r && !rcSimple)
            {
                counter_mod += "(batch*";
                counter_mod += std::to_string(2 * numTrans);
                if(rc_second_index)
                    counter_mod += " + 1";
                else
                    counter_mod += " + 0";

                if(numTrans != 1)
                {
                    counter_mod += " + 2*(me/";
                    counter_mod += std::to_string(workGroupSizePerTrans);
                    counter_mod += "))";
                }
                else
                {
                    counter_mod += ")";
                }
            }
            else
            {
                if(numTrans == 1)
                {
                    counter_mod += "batch";
                }
                else
                {
                    counter_mod += "(batch*";
                    counter_mod += std::to_string(numTrans);
                    counter_mod += " + (me/";
                    counter_mod += std::to_string(workGroupSizePerTrans);
                    counter_mod += "))";
                }
            }

            str += "\t";
            str += "size_t counter_mod = ";
            str += counter_mod;
            str += ";\n";

            /*=======================================================*/
            /*  generate a loop like
            if(dim == 1){
                iOffset += counter_mod*strides[1];
            }
            else if(dim == 2){
                int counter_1 = counter_mod / lengths[1];
                int counter_mod_1 = counter_mod % lengths[1];

                iOffset += counter_1*strides[2] + counter_mod_1*strides[1];
            }
            else if(dim == 3){
                int counter_2 = counter_mod / (lengths[1] * lengths[2]);
                int counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);

                int counter_1 = counter_mod_2 / lengths[1];
                int counter_mod_1 = counter_mod_2 % lengths[1];

                iOffset += counter_2*strides[3] + counter_1*strides[2] +
            counter_mod_1*strides[1];
            }
            else{
                for(int i = dim; i>1; i--){
                    int currentLength = 1;
                    for(int j=1; j<i; j++){
                        currentLength *= lengths[j];
                    }

                    iOffset += (counter_mod / currentLength)*stride[i];
                    counter_mod = counter_mod % currentLength;
                }
                ioffset += counter_mod*strides[1];
            }
            */

            /*=======================================================*/
            std::string loop;
            loop += "\tif(dim == 1){\n";
            loop += "\t\t" + offset_name1 + " += counter_mod*" + stride_name1 + "[1];\n";
            if(output == true)
                loop += "\t\t" + offset_name2 + " += counter_mod*" + stride_name2 + "[1];\n";
            loop += "\t}\n";

            loop += "\telse if(dim == 2){\n";
            loop += "\t\tint counter_1 = counter_mod / lengths[1];\n";
            loop += "\t\tint counter_mod_1 = counter_mod % lengths[1];\n";
            loop += "\t\t" + offset_name1 + " += counter_1*" + stride_name1 + "[2] + counter_mod_1*"
                    + stride_name1 + "[1];\n";
            if(output == true)
                loop += "\t\t" + offset_name2 + " += counter_1*" + stride_name2
                        + "[2] + counter_mod_1*" + stride_name2 + "[1];\n";
            loop += "\t}\n";

            loop += "\telse if(dim == 3){\n";
            loop += "\t\tint counter_2 = counter_mod / (lengths[1] * lengths[2]);\n";
            loop += "\t\tint counter_mod_2 = counter_mod % (lengths[1] * lengths[2]);\n";
            loop += "\t\tint counter_1 = counter_mod_2 / lengths[1];\n";
            loop += "\t\tint counter_mod_1 = counter_mod_2 % lengths[1];\n";
            loop += "\t\t" + offset_name1 + " += counter_2*" + stride_name1 + "[3] + counter_1*"
                    + stride_name1 + "[2] + counter_mod_1*" + stride_name1 + "[1];\n";
            if(output == true)
                loop += "\t\t" + offset_name2 + " += counter_2*" + stride_name2 + "[3] + counter_1*"
                        + stride_name2 + "[2] + counter_mod_1*" + stride_name2 + "[1];\n";
            loop += "\t}\n";

            loop += "\telse{\n";
            loop += "\t\tfor(int i = dim; i>1; i--){\n"; // dim is a runtime variable
            loop += "\t\t\tint currentLength = 1;\n";
            loop += "\t\t\tfor(int j=1; j<i; j++){\n";
            loop += "\t\t\t\tcurrentLength *= lengths[j];\n"; // lengths is a runtime
            // variable
            loop += "\t\t\t}\n";
            loop += "\n";
            loop += "\t\t\t" + offset_name1 + " += (counter_mod / currentLength)*" + stride_name1
                    + "[i];\n";
            if(output == true)
                loop += "\t\t\t" + offset_name2 + " += (counter_mod / currentLength)*"
                        + stride_name2 + "[i];\n";
            loop += "\t\t\tcounter_mod = counter_mod % currentLength;\n"; // counter_mod
            // is
            // calculated
            // at runtime
            loop += "\t\t}\n";
            loop += "\t\t" + offset_name1 + "+= counter_mod * " + stride_name1 + "[1];\n";
            if(output == true)
                loop += "\t\t" + offset_name2 + "+= counter_mod * " + stride_name2 + "[1];\n";
            loop += "\t}\n";

            str += loop;

            return str;
        }

        // A wraper function to parse config parameters before to generate
        // kernel for a single pass in Stockham.
        void GenerateSinglePassKernel(std::string& str,
                                      bool         fwd,
                                      double       scale,
                                      bool         inReal,
                                      bool         outReal,
                                      bool         inInterleaved,
                                      bool         outInterleaved,
                                      typename std::vector<Pass<PR>>::const_iterator& p)
        {
            //TODO: this function is not encapsulated good enough...

            bool ldsInterleaved = inInterleaved || outInterleaved;
            ldsInterleaved      = halfLds ? false : ldsInterleaved;
            ldsInterleaved      = blockCompute ? true : ldsInterleaved;

            double s   = 1.0;
            size_t ins = 1, outs = 1; // default unit_stride
            bool   gIn = false, gOut = false;
            bool   inIlvd = false, outIlvd = false;
            bool   inRl = false, outRl = false;
            bool   tw3Step = false;

            if(p == passes.cbegin() && params.fft_twiddleFront)
            {
                tw3Step = params.fft_3StepTwiddle;
            }
            if((p + 1) == passes.cend())
            {
                s = scale;
                if(!params.fft_twiddleFront)
                    tw3Step = params.fft_3StepTwiddle;
            }

            if(blockCompute && !r2c2r)
            {
                inIlvd  = ldsInterleaved;
                outIlvd = ldsInterleaved;
            }
            else
            {
                if(p == passes.cbegin())
                {
                    inIlvd = inInterleaved;
                    inRl   = inReal;
                    gIn    = true;
                    ins    = 0x7fff;
                } // 0x7fff = 32767 in decimal, indicating non-unit stride
                // ins = -1 is for non-unit stride, the 1st pass may read strided
                // memory, while the middle pass read/write LDS which guarantees
                // unit-stride
                if((p + 1) == passes.cend())
                {
                    outIlvd = outInterleaved;
                    outRl   = outReal;
                    gOut    = true;
                    outs    = 0x7fff;
                } // 0x7fff is non-unit stride
                // ins = -1 is for non-unit stride, the last pass may write strided
                // memory
                if(p != passes.cbegin())
                {
                    inIlvd = ldsInterleaved;
                }
                if((p + 1) != passes.cend())
                {
                    outIlvd = ldsInterleaved;
                }
            }
            p->GeneratePass(fwd,
                            name_suffix,
                            str,
                            tw3Step,
                            params.fft_twiddleFront,
                            inIlvd,
                            outIlvd,
                            inRl,
                            outRl,
                            ins,
                            outs,
                            s,
                            gIn,
                            gOut);
        }

        /* =====================================================================
                write pass functions
                passes call butterfly device functions
                passes use twiddles
                inplace outof place shared the same pass functions
            =================================================================== */
        void GeneratePassesKernel(std::string& str)
        {
            str += "\n////////////////////////////////////////Passes kernels\n";
            // Input is real format
            bool inReal = params.fft_inputLayout == rocfft_array_type_real;
            // Output is real format
            bool outReal = params.fft_outputLayout == rocfft_array_type_real;

            for(size_t d = 0; d < 2; d++)
            {
                bool   fwd   = d ? false : true;
                double scale = fwd ? params.fft_fwdScale : params.fft_backScale;
                for(auto p = passes.cbegin(); p != passes.cend(); ++p)
                {
                    GenerateSinglePassKernel(str, fwd, scale, inReal, outReal, true, true, p);

                    // TODO: double check the special cases sbrc and sbcc
                    if(!(name_suffix == "_sbrc" || name_suffix == "_sbcc"))
                    {
                        if(numPasses == 1)
                        {
                            GenerateSinglePassKernel(
                                str, fwd, scale, inReal, outReal, false, true, p);
                            GenerateSinglePassKernel(
                                str, fwd, scale, inReal, outReal, true, false, p);
                            GenerateSinglePassKernel(
                                str, fwd, scale, inReal, outReal, false, false, p);
                        }
                        else if(p == passes.cbegin())
                        {
                            GenerateSinglePassKernel(
                                str, fwd, scale, inReal, outReal, false, true, p);
                        }
                        else if((p + 1) == passes.cend())
                        {
                            GenerateSinglePassKernel(
                                str, fwd, scale, inReal, outReal, true, false, p);
                        }
                    }
                }
            }
        }

        /* =====================================================================
                generate fwd or back ward length-point FFT device functions :
                encapsulate passes
                called by kernels which set up shared memory (LDS), offset, etc
            =================================================================== */
        void GenerateEncapsulatedPassesKernel(std::string& str)
        {
            str += "\n////////////////////////////////////////Encapsulated passes kernels\n";
            std::string rType  = RegBaseType<PR>(1);
            std::string r2Type = RegBaseType<PR>(2);
            for(int in = 1; in >= 0; in--)
                for(int out = 1; out >= 0; out--)
                {
                    bool inInterleaved  = in;
                    bool outInterleaved = out;

                    // use interleaved LDS when halfLds constraint absent
                    bool ldsInterleaved = inInterleaved || outInterleaved;
                    ldsInterleaved      = halfLds ? false : ldsInterleaved;
                    ldsInterleaved      = blockCompute ? true : ldsInterleaved;

                    for(size_t d = 0; d < 2; d++)
                    {
                        bool fwd;
                        fwd = d ? false : true;

                        if(NeedsLargeTwiddles())
                        {
                            str += "template <typename T, StrideBin sb, bool TwdLarge, "
                                   "CallbackType cbtype, size_t "
                                   "LTBase>\n";
                        }
                        else
                        {
                            str += "template <typename T, StrideBin sb,"
                                   "CallbackType cbtype>\n";
                        }

                        str += "__device__ void \n";

                        if(fwd)
                            str += "fwd_len";
                        else
                            str += "back_len";
                        str += std::to_string(length) + name_suffix;
                        str += "_device";

                        str += "(const T * const twiddles, ";
                        if(NeedsLargeTwiddles())
                            str += "const T * const twiddles_large, "; // the blockCompute BCT_C2C
                        // algorithm use one more twiddle parameter
                        str += "const size_t stride_in, const size_t stride_out, unsigned int "
                               "rw, unsigned int b, ";
                        str += "unsigned int me, unsigned int ldsOffset, ";

                        if(inInterleaved)
                            str += r2Type + " *gbIn, ";
                        else
                            str += rType + " *bufInRe, " + rType + " *bufInIm, ";
                        if(!blockCompute)
                            str += "unsigned int iOffset, ";

                        if(outInterleaved)
                            str += r2Type + " *gbOut";
                        else
                            str += rType + " *bufOutRe, " + rType + " *bufOutIm";
                        if(!blockCompute)
                            str += ", unsigned int oOffset";

                        if(blockCompute) // blockCompute' lds type is T
                        {
                            str += ", " + r2Type + " *lds";
                        }
                        else
                        {
                            if(numPasses > 1)
                                str += ", " + rType + " *lds"; // only multiple pass use lds
                        }

                        // declare callback params for "small"
                        // kernels ("large" kernels read/write to
                        // global mem outside of the passes and would
                        // run callbacks there)
                        if(!blockCompute)
                            str += DeclareCallbackParams();

                        str += ")\n";
                        str += "{\n";

                        // Setup registers if needed
                        if(linearRegs)
                        {
                            str += "\t";
                            str += r2Type;
                            str += " ";
                            str += IterRegs("", false);
                            str += ";\n";
                        }

                        if(numPasses == 1)
                        {
                            str += "\t";
                            str += PassName(0, fwd, length, name_suffix);
                            if(NeedsLargeTwiddles())
                            {
                                str += "<T, sb, TwdLarge, cbtype, LTBase>(twiddles, "
                                       "twiddles_large, "; // the blockCompute BCT_C2C algorithm use
                            }
                            else
                            {
                                str += "<T, sb, cbtype>(twiddles, ";
                            }

                            // one more twiddle parameter
                            str += "stride_in, stride_out, rw, b, me, iOffset, oOffset,";

                            if(inInterleaved)
                                str += " gbIn,";
                            else
                                str += " bufInRe, bufInIm,";

                            if(outInterleaved)
                                str += " gbOut";
                            else
                                str += " bufOutRe, bufOutIm";

                            str += IterRegs("&");
                            // blockCompute reads/writes to global outside of the FFT passes
                            if(!blockCompute)
                                str += PassCallbackParams();
                            str += ");\n";
                        }
                        else
                        {
                            for(auto p = passes.begin(); p != passes.end(); ++p)
                            {
                                bool        firstPass = p == passes.begin();
                                bool        lastPass  = (p + 1) == passes.end();
                                std::string exTab     = "";

                                str += exTab;
                                str += "\t";
                                str += PassName(p->GetPosition(), fwd, length, name_suffix);
                                // the blockCompute BCT_C2C algorithm use one more twiddle parameter
                                if(NeedsLargeTwiddles())
                                {
                                    str += "<T, sb, TwdLarge, cbtype, LTBase>(twiddles, "
                                           "twiddles_large, ";
                                }
                                else
                                {
                                    str += "<T, sb, cbtype>(twiddles, ";
                                }

                                str += "stride_in, stride_out, rw, b, me, ";

                                std::string ldsArgs;
                                if(halfLds)
                                {
                                    ldsArgs += "lds, lds";
                                }
                                else
                                {
                                    if(ldsInterleaved)
                                    {
                                        ldsArgs += "lds";
                                    }
                                    else
                                    {
                                        ldsArgs += "lds, lds + ";
                                        ldsArgs += std::to_string(length * numTrans);
                                    }
                                }

                                // about offset
                                if(firstPass) // beginning pass
                                {
                                    if(blockCompute) // blockCompute use shared memory (lds), so if
                                    // true, use ldsOffset
                                    {
                                        str += "ldsOffset, ";
                                    }
                                    else
                                    {
                                        str += "iOffset, ";
                                    }

                                    str += "ldsOffset, ";
                                    if(inInterleaved)
                                        str += " gbIn, ";
                                    else
                                        str += " bufInRe, bufInIm, ";

                                    str += ldsArgs;
                                }
                                else if(lastPass) // ending pass
                                {
                                    str += "ldsOffset, ";
                                    if(blockCompute) // blockCompute use shared memory (lds), so if
                                    // true, use ldsOffset
                                    {
                                        str += "ldsOffset, ";
                                    }
                                    else
                                    {
                                        str += "oOffset, ";
                                    }
                                    str += ldsArgs;

                                    if(outInterleaved)
                                        str += ",  gbOut";
                                    else
                                        str += ", bufOutRe, bufOutIm";
                                }
                                else // intermediate pass
                                {
                                    str += "ldsOffset, ldsOffset, ";
                                    str += ldsArgs;
                                    str += ", ";
                                    str += ldsArgs;
                                }

                                str += IterRegs("&");
                                // blockCompute reads/writes to global outside of the FFT passes
                                if(!blockCompute)
                                    str += PassCallbackParams(firstPass ? PassNormal : PassNothing,
                                                              lastPass ? PassNormal : PassNothing);
                                str += ");\n";
                                if(!halfLds)
                                {
                                    str += exTab;
                                    str += "\t__syncthreads();\n";
                                }
                            }
                        } // if (numPasses == 1)
                        str += "}\n\n";
                    }
                }
        }

    public:
        virtual size_t SharedMemSize(bool ldsInterleaved)
        {
            if(blockCompute)
                return blockLDS;
            else
            {
                size_t ldsSize = halfLds ? length * numTrans : 2 * length * numTrans;
                return ldsInterleaved ? ldsSize / 2 : ldsSize;
            }
        }
        virtual void GenerateSingleGlobalKernelSharedMem(std::string&            str,
                                                         bool                    ldsInterleaved,
                                                         rocfft_result_placement placeness,
                                                         const std::string&      rType,
                                                         const std::string&      r2Type)
        {
            size_t ldsSize = SharedMemSize(ldsInterleaved);
            str += "\n\t";
            if(blockCompute && blockComputeType == BCT_R2C)
                str += "extern ";

            str += "__shared__ ";
            if(blockCompute)
                str += r2Type;
            else
                str += ldsInterleaved ? r2Type : rType;
            str += " lds[";
            if(!(blockCompute && blockComputeType == BCT_R2C))
                str += std::to_string(ldsSize);
            str += "];\n";

            if(NeedsLargeTwiddles())
            {
                str += "\t// when TwdLarge=false, host even doesn't allocate large twd memory. "
                       "Declare a 0-size array to pass compiler";
                str += "\n\t__shared__ ";
                str += r2Type;
                str += " large_twd_lds[";
                str += "(TwdLarge && LTBase < " + std::to_string(LTWD_BASE_DEFAULT)
                       + ")? (1 << LTBase) * 3 : 0"; // for 3-steps ltwd
                str += "];\n";
            }
        }

        virtual std::string LaunchBounds()
        {
            std::string str = "__launch_bounds__(";
            if(blockCompute)
                str += std::to_string(blockWGS);
            else
                str += std::to_string(workGroupSize);
            str += ")\n";
            return str;
        }

        virtual std::string GlobalKernelFunctionSuffix()
        {
            return "_len" + std::to_string(length) + name_suffix;
        }

        virtual bool StrideParamUnderscore()
        {
            return false;
        }
        virtual bool LengthParamUnderscore()
        {
            return false;
        }
        virtual bool IOParamUnderscore()
        {
            return false;
        }

        void GenerateSingleGlobalKernelPrototype(std::string&            str,
                                                 bool                    fwd,
                                                 rocfft_result_placement placeness,
                                                 bool                    inInterleaved,
                                                 bool                    outInterleaved,
                                                 bool                    ldsInterleaved,
                                                 const std::string&      rType,
                                                 const std::string&      r2Type)
        {
            str += "//Kernel configuration: number of threads per thread block: ";
            if(blockCompute)
                str += std::to_string(blockWGS);
            else
                str += std::to_string(workGroupSize);
            str += ", ";
            if(!blockCompute)
                str += "maximum ";
            str += "transforms: " + std::to_string(numTrans)
                   + ", Passes: " + std::to_string(numPasses) + "\n";
            // FFT kernel begin
            // Function signature
            if(NeedsLargeTwiddles())
            {
                str += "template <typename T, StrideBin sb, bool TwdLarge, CallbackType cbtype, "
                       "size_t LTBase="
                       + std::to_string(LTWD_BASE_DEFAULT) + ">\n";
            }
            // SBRC has additional parameters for fused transpose varieties
            else if(blockComputeType == BCT_R2C)
            {
                str += "template <typename T, StrideBin sb, SBRC_TYPE Tsbrc, SBRC_TRANSPOSE_TYPE "
                       "Ttranspose, EmbeddedType ebtype, CallbackType cbtype>\n";
            }
            else
            {
                str += "template <typename T, StrideBin sb, EmbeddedType ebtype, CallbackType "
                       "cbtype>\n";
            }

            str += "__global__ void\n";
            str += LaunchBounds();

            // kernel name
            if(fwd)
                str += "fft_fwd_";
            else
                str += "fft_back_";
            if(placeness == rocfft_placement_notinplace)
                str += "op"; // outof place
            else
                str += "ip"; // inplace
            str += GlobalKernelFunctionSuffix();
            /* kernel arguments,
                    lengths, strides are transferred to kernel as a run-time parameter.
                    lengths, strides may be high dimension arrays
                */
            str += "( ";
            str += "const " + r2Type + " * __restrict__ twiddles, ";
            if(NeedsLargeTwiddles())
            {
                str += "const " + r2Type
                       + " * __restrict__ twiddles_large, "; // blockCompute introduce
                // one more twiddle parameter
            }
            str += "const size_t dim, const size_t *";
            if(LengthParamUnderscore())
                str += "_";
            str += "lengths, ";
            str += "const size_t *";
            if(StrideParamUnderscore())
                str += "_";
            str += "stride_in, ";
            if(placeness == rocfft_placement_notinplace)
            {
                str += "const size_t *";
                if(StrideParamUnderscore())
                    str += "_";
                str += "stride_out, ";
            }
            str += "const size_t batch_count";
            str += DeclareCallbackParams();
            str += ", ";

            // Function attributes
            if(placeness == rocfft_placement_inplace)
            {

                assert(inInterleaved == outInterleaved);

                if(inInterleaved)
                {
                    str += r2Type;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gb";
                    str += ")\n";
                }
                else
                {
                    str += rType;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbRe, ";
                    str += rType;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbIm";

                    str += ")\n";
                }
            }
            else
            {
                if(inInterleaved)
                {
                    // str += "const ";
                    str += r2Type;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbIn, "; // has to remove const qualifier
                        // due to HIP on ROCM 1.4
                }
                else
                {
                    str += rType;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbInRe, ";
                    // str += "const ";
                    str += rType;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbInIm, ";
                }

                if(outInterleaved)
                {
                    str += r2Type;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbOut";
                }
                else
                {
                    str += rType;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbOutRe, ";
                    str += rType;
                    str += " * __restrict__ ";
                    if(IOParamUnderscore())
                        str += "_";
                    str += "gbOutIm";
                }

                str += ")\n";
            }
        }

        virtual void GenerateSingleGlobalKernelRWFlag(std::string& str)
        {
            // Conditional read-write ('rw') controls each thread behavior when it
            // is not divisible
            // for 2D, 3D layout, the "upper_count" viewed by kernels is
            // batch_count*length[1]*length[2]*...*length[dim-1]
            // because we flatten other dimensions to 1D dimension when
            // configurating the thread blocks
            if((numTrans > 1) && !blockCompute)
            {
                str += "\tunsigned int upper_count = batch_count;\n";
                str += "\tfor(int i=1; i<dim; i++){\n";
                str += "\t\tupper_count *= lengths[i];\n";
                str += "\t}\n";
                str += "\t// do signed math to guard against underflow\n";
                str += "\tunsigned int rw = (static_cast<int>(me) < "
                       "(static_cast<int>(upper_count) ";
                str += " - static_cast<int>(batch)*";
                str += std::to_string(numTrans);
                str += ")*";
                str += std::to_string(workGroupSizePerTrans);
                str += ") ? 1 : 0;\n\n";
            }
            else
            {
                str += "\tunsigned int rw = 1;\n\n";
            }

            // The following lines suppress warning; when rw=1, generator directly
            // puts 1 as the pass device function
            str += "\t//suppress warning\n";
            str += "\t#ifdef __NVCC__\n";
            str += "\t\t(void)(rw == rw);\n";
            str += "\t#else\n";
            str += "\t\t(void)rw;\n";
            str += "\t#endif\n";
        }

        virtual void GenerateSingleGlobalKernelIOOffsets(std::string&            str,
                                                         rocfft_result_placement placeness)

        {
            /* =====================================================================
                    Setup memory pointers with offset
                    =================================================================== */

            if(placeness == rocfft_placement_inplace)
            {

                if(blockCompute)
                    str += OffsetCalcBlockCompute("ioOffset", "stride_in", "", "", true, false);
                else
                    str += OffsetCalc("ioOffset", "stride_in", "", "", false);
            }
            else
            {
                if(blockCompute)
                {
                    str += OffsetCalcBlockCompute(
                        "iOffset", "stride_in", "oOffset", "stride_out", true, true);
                }
                else
                {
                    str += OffsetCalc("iOffset", "stride_in", "oOffset", "stride_out", true);
                }
            }
        }

        virtual void GenerateSingleGlobalKernelBody(std::string&            str,
                                                    bool                    fwd,
                                                    rocfft_result_placement placeness,
                                                    bool                    inInterleaved,
                                                    bool                    outInterleaved,
                                                    const std::string&      rType,
                                                    const std::string&      r2Type)
        {
            // Initialize
            str += "\t";
            str += "unsigned int me = (unsigned int)hipThreadIdx_x;\n\t";
            str += "unsigned int batch = (unsigned int)hipBlockIdx_x;";
            str += "\n";

            std::string ltwdLDS_cond = "TwdLarge && LTBase < " + std::to_string(LTWD_BASE_DEFAULT);

            if(NeedsLargeTwiddles())
            {
                str += "\n\t// when TwdLarge=false, host even doesn't allocate large twd memory";
                str += "\n\tif(" + ltwdLDS_cond + ")";
                str += "\n\t{";
                str += "\n\t\tsize_t ltwd_id = me;";
                str += "\n\t\twhile(ltwd_id < (1 << LTBase) * 3 )";
                str += "\n\t\t{";
                str += "\n\t\t\tlarge_twd_lds[ltwd_id] = twiddles_large[ltwd_id];";
                str += "\n\t\t\tltwd_id += " + std::to_string(blockWGS) + ";";
                str += "\n\t\t}";
                str += "\n\t\t//__syncthreads(); don't have to sync because pass0 will sync "
                       "anyway\n";
                str += "\n\t}";
            }

            // Declare memory pointers
            str += "\n\t";

            if(placeness == rocfft_placement_inplace)
            {
                str += "unsigned int ioOffset = 0;\n\t";
                str += "\n";
            }
            else
            {
                str += "unsigned int iOffset = 0;\n\t";
                str += "unsigned int oOffset = 0;\n\t";
                str += "\n";
            }

            GenerateSingleGlobalKernelRWFlag(str);

            // printf("fft_3StepTwiddle = %d, lengths = %zu\n",
            // params.fft_3StepTwiddle, length);

            // Transform index for 3-step twiddles
            if(params.fft_3StepTwiddle && !blockCompute)
            {
                if(numTrans == 1)
                {
                    str += "\tunsigned int b = batch%";
                }
                else
                {
                    str += "\tunsigned int b = (batch*";
                    str += std::to_string(numTrans);
                    str += " + (me/";
                    str += std::to_string(workGroupSizePerTrans);
                    str += "))%";
                }

                // str += std::to_string(params.fft_N[1]);
                str += "lengths[1]";
                str += ";\n\n";
            }
            else
            {
                str += "\tunsigned int b = 0;\n\n";
            }

            str += "   ";
            str += GEN_REF_LINE();
            GenerateSingleGlobalKernelIOOffsets(str, placeness);

            /* =====================================================================
                    blockCompute only: Read data into shared memory (LDS) for blocked
                    access
                    =================================================================== */

            if(blockCompute)
            {
                str += "\n\tunsigned int lds_row_padding = 0;";
                str += DeclareLoadCBPointer();

                if(blockComputeType == BCT_R2C)
                {
                    str += "\n\tif(Tsbrc == SBRC_3D_FFT_ERC_TRANS_Z_XY)";
                    str += "\n\t\tlds_row_padding = " + std::to_string(blockLdsRowPadding) + ";";
                }

                size_t loopCount = (length * blockWidth) / blockWGS;

                str += "\n\tfor(unsigned int t=0; t<";
                str += std::to_string(loopCount);
                str += "; t++)";
                str += GEN_REF_LINE();
                str += "\t{\n";

                if(blockComputeType == BCT_R2C)
                {
                    str += "\t\tif(Ttranspose == TILE_UNALIGNED && Tsbrc == "
                           "SBRC_3D_FFT_TRANS_XY_Z)\n";
                    str += "\t\t{\n";
                    str += "\t\t\t// for rectangular cases, make sure we don't read more "
                           "rows than are there\n";
                    str += "\t\t\tif(block_in_batch / lengths[1] * " + std::to_string(blockWidth)
                           + " + t % " + std::to_string(blockWidth) + " + me / "
                           + std::to_string(length) + " * " + std::to_string(blockWidth) + " * "
                           + std::to_string(length) + " / " + std::to_string(blockWGS)
                           + " >= lengths[2])\n";
                    str += "\t\t\t\tcontinue;\n";
                    str += "\t\t}\n";
                }

                str += "\t\tT R0;\n";
                str += "\t\t// Calc global offset within a tile and read\n";

                std::string input_offset
                    = placeness == rocfft_placement_inplace ? "ioOffset" : "iOffset";

                if((blockComputeType == BCT_C2C) || (blockComputeType == BCT_C2R))
                {
                    std::string global_offset = input_offset + " + (me%";
                    global_offset += std::to_string(blockWidth);

                    if(blockComputeType
                       == BCT_C2C) // the most inner part of offset calc needs to count stride[1] for SBCC
                        global_offset += ") * stride_in[1] + ";
                    else
                        global_offset += ") + ";

                    global_offset += "(me/";
                    global_offset += std::to_string(blockWidth);
                    global_offset += ")*stride_in[0] + t*stride_in[0]*";
                    global_offset += std::to_string(blockWGS / blockWidth);

                    str += GlobalReadStrCat(global_offset, placeness, inInterleaved);
                }
                else // SBRC
                {
                    str += "\t\tif(Tsbrc == SBRC_2D || Tsbrc == SBRC_3D_FFT_TRANS_Z_XY || "
                           "Tsbrc == SBRC_3D_FFT_ERC_TRANS_Z_XY)\n";
                    str += "\t\t{\n";

                    std::string global_offset;
                    global_offset = input_offset + " + me + t*";
                    global_offset += std::to_string(blockWGS);

                    str += GlobalReadStrCat(global_offset, placeness, inInterleaved);

                    str += "\t\t}\n";
                    str += "\t\telse if(Tsbrc == SBRC_3D_FFT_TRANS_XY_Z)\n";
                    str += "\t\t{\n";

                    // recalculate offset for case SBRC_3D_FFT_TRANS_XY_Z
                    global_offset = input_offset + " + me % " + std::to_string(length)
                                    + " * stride_in[0] + ((me /" + std::to_string(length) + " * "
                                    + std::to_string(blockWGS / blockWidth) + ") + t % "
                                    + std::to_string(blockWidth) + ")*stride_in[2] + t / "
                                    + std::to_string(blockWidth) + " * " + std::to_string(blockWGS)
                                    + " * stride_in[0]";

                    str += GlobalReadStrCat(global_offset, placeness, inInterleaved);

                    str += "\t\t}\n";
                }

                if((blockComputeType == BCT_C2C) || (blockComputeType == BCT_C2R))
                {
                    str += "\t\t// Write into lds in column-major\n";
                    str += "\t\t// In lds, the offset = blockIdx * blockOffset + threadIdx_x * "
                           "threadOffset_x + threadIdx_y * 1\n";
                    str += "\t\t// which is    R0 = lds[   t     *  (wgs/bwd)  +  (me%bwd)   *  "
                           "length[0]     + (me/bwd)    * 1]\n";
                    str += "\t\tlds[t*";
                    str += std::to_string(blockWGS / blockWidth);
                    str += " + ";
                    str += "(me%";
                    str += std::to_string(blockWidth);
                    str += ")*";
                    str += std::to_string(length);
                    str += " + ";
                    str += "(me/";
                    str += std::to_string(blockWidth);
                    str += ")] = R0;";
                    str += "\n";
                }
                else
                {
                    str += "\n\t\t// Write into lds in row-major\n";
                    str += "\t\tif(Tsbrc == SBRC_2D || Tsbrc == SBRC_3D_FFT_TRANS_Z_XY || "
                           "Tsbrc == SBRC_3D_FFT_ERC_TRANS_Z_XY)\n";
                    str += "\t\t\tlds[t * ";

                    if(length / blockWGS)
                    {
                        str += std::to_string(blockWGS);
                        str += " + t / " + std::to_string(length / blockWGS) + " * lds_row_padding";
                    }
                    else
                    {
                        str += "( " + std::to_string(blockWGS) + " + "
                               + std::to_string(blockWGS / length) + " * lds_row_padding)";
                        str += " + (me / " + std::to_string(length) + " ) * lds_row_padding";
                    }

                    str += " + me] = R0;\n";
                    str += "\t\telse\n";
                    str += "\t\t\tlds[t % " + std::to_string(blockWidth) + " *"
                           + std::to_string(length) + " + t / " + std::to_string(blockWidth) + " * "
                           + std::to_string(blockWGS) + " + me % " + std::to_string(length)
                           + " + me / " + std::to_string(length) + " * "
                           + std::to_string(loopCount * length) + "] = R0;\n";
                }

                str += "\t}\n\n";
                str += "\t__syncthreads();\n\n";
            }

            /* =====================================================================
                    Set rw and 'me'
                    rw string also contains 'b'
                    =================================================================== */

            std::string rw, me;

            if(r2c2r && !rcSimple)
                rw = " rw, b, ";
            else
                rw = ((numTrans > 1) || realSpecial) ? " rw, b, " : " 1, b, ";

            if(numTrans > 1)
            {
                me += "me%";
                me += std::to_string(workGroupSizePerTrans);
                me += ", ";
            }
            else
            {
                me += "me, ";
            }

            if(blockCompute)
            {
                me = "me%";
                me += std::to_string(workGroupSizePerTrans);
                me += ", ";
            } // me is overwritten if blockCompute true

            // Buffer strings
            std::string inBuf, outBuf;

            if(placeness == rocfft_placement_inplace)
            {
                if(inInterleaved)
                {
                    inBuf  = "gb, ioOffset, ";
                    outBuf = "gb, ioOffset";
                }
                else
                {
                    inBuf  = "gbRe, gbIm, ioOffset, ";
                    outBuf = "gbRe, gbIm, ioOffset";
                }
            }
            else
            {
                if(inInterleaved)
                    inBuf = "gbIn, iOffset, ";
                else
                    inBuf = "gbInRe, gbInIm, iOffset, ";
                if(outInterleaved)
                    outBuf = "gbOut, oOffset";
                else
                    outBuf = "gbOutRe, gbOutIm, oOffset";
            }

            /* =====================================================================
                    call FFT devices functions in the generated kernel
                    ===================================================================*/

            if(blockCompute) // for blockCompute, a loop is required, inBuf, outBuf
            // would be overwritten
            {
                str += "\n\tfor(unsigned int t=0; t<";
                str += std::to_string(blockWidth / (blockWGS / workGroupSizePerTrans));
                str += "; t++)";
                str += GEN_REF_LINE();
                str += "\t{\n\n";

                inBuf  = "lds, ";
                outBuf = "lds";

                if(params.fft_3StepTwiddle)
                {
                    str += "\t\tb = (batch % (lengths[1]/";
                    str += std::to_string(blockWidth);
                    // str += std::to_string(params.fft_N[1] / blockWidth);
                    str += "))*";
                    str += std::to_string(blockWidth);
                    str += " + t*";
                    str += std::to_string(blockWGS / workGroupSizePerTrans);
                    str += " + (me/";
                    str += std::to_string(workGroupSizePerTrans);
                    str += ");\n\n";
                }
                str += "\t";
            }

            str += "\t// Perform FFT input: gb(In) ; output: gb(Out); working "
                   "space: lds \n";

            if(blockCompute)
                str += "\t";
            str += "\t// rw, b, me% control read/write; then ldsOffset, gb, lds\n";

            std::string ldsOff;

            if(blockCompute) // blockCompute changes the ldsOff
            {
                ldsOff += "t * (";
                ldsOff += std::to_string(length) + " + lds_row_padding) * "
                          + std::to_string(blockWGS / workGroupSizePerTrans);
                ldsOff += " + (me/";
                ldsOff += std::to_string(workGroupSizePerTrans);
                ldsOff += ")*(";
                ldsOff += std::to_string(length);
                ldsOff += "+ lds_row_padding)";
                str += "\t";
            }
            else
            {
                if(numTrans > 1)
                {
                    ldsOff += "(me/";
                    ldsOff += std::to_string(workGroupSizePerTrans);
                    ldsOff += ")*";
                    ldsOff += std::to_string(length);
                }
                else
                {
                    ldsOff += "0";
                }
            }
            str += "\t";
            if(fwd)
                str += "fwd_len";
            else
                str += "back_len";
            str += std::to_string(length) + name_suffix;
            std::string sb = params.forceNonUnitStride ? "SB_NONUNIT" : "sb";
            if(NeedsLargeTwiddles())
            {
                str += "_device<T, " + sb + ", TwdLarge, cbtype, LTBase>(twiddles, (" + ltwdLDS_cond
                       + ")? large_twd_lds : twiddles_large, ";
            }
            else
            {
                str += "_device<T, " + sb + ", cbtype>(twiddles, ";
            }

            str += "stride_in[0], ";
            str += ((placeness == rocfft_placement_inplace) ? "stride_in[0], " : "stride_out[0], ");

            str += rw;
            str += me;
            str += ldsOff + ", ";

            str += inBuf + outBuf;

            if(numPasses > 1)
            {
                str += ", lds"; // only multiple pass use lds
            }
            // blockCompute reads/writes to global outside of the FFT passes
            if(!blockCompute)
                str += PassCallbackParams(DeviceFuncLoadCBPassType(), DeviceFuncStoreCBPassType());
            str += ");\n";

            if(blockCompute || realSpecial) // the "}" enclose the loop introduced by blockCompute
            {
                str += "\n\t}\n\n";
            }

            // Write data from shared memory (LDS) for blocked access
            if(blockCompute)
            {
                std::string strLdsRead;
                std::string strGlobalWrite;

                size_t loopCount = (length * blockWidth) / blockWGS;

                str += "\t__syncthreads();\n\n";

                str += DeclareStoreCBPointer();

                // The code block for even-length real2complex post processing
                if(blockComputeType == BCT_R2C)
                {
                    str += "\tif (Tsbrc == SBRC_3D_FFT_ERC_TRANS_Z_XY)\n\t{";
                    str += "\n\t\tfor(unsigned int r = 0; r < " + std::to_string(blockWidth)
                           + "; r++)\n\t\t{";
                    str += "\n\t\t\tpost_process_interleaved_inplace<T, ";
                    str += (length % 2) ? "false" : "true";
                    str += ",CallbackType::NONE>(me, " + std::to_string(length) + " - me, "
                           + std::to_string(length) + ", " + std::to_string(length / 2);
                    str += ", &lds[r * (" + std::to_string(length) + " + lds_row_padding)], 0";
                    str += ", &twiddles[" + std::to_string(length) + "]";
                    str += PassCallbackParams(PassNull, PassNull);
                    str += ");";
                    str += "\n\t\t}";

                    str += "\n\t\t__syncthreads();\n\t}\n";
                }

                str += "\n\tfor(unsigned int t=0; t<";
                str += std::to_string(loopCount);
                str += "; t++)";
                str += GEN_REF_LINE();
                str += "\t{\n";

                if(blockComputeType == BCT_R2C)
                {
                    str += "\t\tif(Ttranspose == TILE_UNALIGNED && Tsbrc == "
                           "SBRC_3D_FFT_TRANS_XY_Z)\n";
                    str += "\t\t{\n";
                    str += "\t\t\t// for rectangular cases, make sure we don't write more "
                           "rows than are there\n";
                    str += "\t\t\tif(block_in_batch / lengths[1] * " + std::to_string(blockWidth)
                           + " + me % " + std::to_string(blockWidth) + " >= lengths[2])\n";
                    str += "\t\t\t\tcontinue;\n";
                    str += "\t\t}\n";
                }
                if((blockComputeType == BCT_C2C) || (blockComputeType == BCT_R2C))
                {
                    str += "\t\t// Read from lds and write to global mem in column-major\n";
                    str += "\t\t// In lds, the offset = blockIdx * blockOffset + threadIdx_x * "
                           "threadOffset_x + threadIdx_y * 1\n";
                    str += "\t\t// which is    R0 = lds[   t     *  (wgs/bwd)  +  (me%bwd)   *  "
                           "length[0]     + (me/bwd)    * 1]\n";

                    strLdsRead += "\t\tT R0 = lds[t*";
                    strLdsRead += std::to_string(blockWGS / blockWidth);
                    strLdsRead += " + ";
                    strLdsRead += "(me%";
                    strLdsRead += std::to_string(blockWidth);
                    strLdsRead += ")*(";
                    strLdsRead += std::to_string(length);
                    strLdsRead += " + lds_row_padding)";
                    strLdsRead += " + ";
                    strLdsRead += "(me/";
                    strLdsRead += std::to_string(blockWidth);
                    strLdsRead += ")];";
                    strLdsRead += "\n";

                    str += strLdsRead;
                }
                else
                {
                    str += "\t\t// Read from lds and write to global mem in row-major\n";
                    str += "\t\t// Mapping threads to lds: R0 = lds[t*wgs + me]\n";
                    str += "\t\tT R0 = lds[t*";
                    str += std::to_string(blockWGS);
                    str += " + me];";
                    str += "\n";
                }

                str += "\n\t\t// Calc global offset within a tile and write\n";
                std::string output_offset
                    = placeness == rocfft_placement_inplace ? "ioOffset" : "oOffset";
                if((blockComputeType == BCT_C2C) || (blockComputeType == BCT_R2C))
                {
                    // start to calc the global write offset

                    if(blockComputeType == BCT_R2C)
                    {
                        str += "\t\tif(Tsbrc == SBRC_2D)\n";
                        str += "\t\t{\n";
                    }

                    {
                        std::string global_offset;
                        global_offset = output_offset + " + (me%" + std::to_string(blockWidth);

                        // the most inner part of offset calc needs to count stride[1] for block compute
                        global_offset
                            += ((placeness == rocfft_placement_inplace) ? ") * stride_in[1] + "
                                                                        : ") * stride_out[1] + ");

                        global_offset += "(me/";
                        global_offset += std::to_string(blockWidth);
                        global_offset += ((placeness == rocfft_placement_inplace)
                                              ? ")*stride_in[0] + t*stride_in[0]*"
                                              : ")*stride_out[0] + t*stride_out[0]*");
                        global_offset += std::to_string(blockWGS / blockWidth);
                        str += GlobalWriteStrCat(global_offset, placeness, outInterleaved);
                    }

                    if(blockComputeType == BCT_R2C)
                    {
                        str += "\t\t}\n";
                        str += "\t\telse if(Tsbrc == SBRC_3D_FFT_TRANS_XY_Z)\n";
                        str += "\t\t{\n";

                        std::string global_offset;

                        global_offset = output_offset + " + (me%";
                        global_offset += std::to_string(blockWidth);
                        global_offset += ") * stride_";
                        global_offset += placeness == rocfft_placement_inplace ? "in" : "out";
                        global_offset += "[0] + (me/" + std::to_string(blockWidth);
                        global_offset += (placeness == rocfft_placement_inplace)
                                             ? ")*stride_in[1] + t*stride_in[1]*"
                                             : ")*stride_out[1] + t*stride_out[1]*";
                        global_offset += std::to_string(blockWGS / blockWidth);
                        str += GlobalWriteStrCat(global_offset, placeness, outInterleaved);

                        str += "\t\t}\n";
                        str += "\t\telse if(Tsbrc == SBRC_3D_FFT_TRANS_Z_XY || Tsbrc == "
                               "SBRC_3D_FFT_ERC_TRANS_Z_XY)\n";
                        str += "\t\t{\n";

                        global_offset = output_offset + " + (me%";
                        global_offset += std::to_string(blockWidth);
                        global_offset += ") * stride_";
                        global_offset += placeness == rocfft_placement_inplace ? "in" : "out";
                        global_offset += "[0] + (me/" + std::to_string(blockWidth);
                        global_offset += (placeness == rocfft_placement_inplace)
                                             ? ")*stride_in[2] + t*stride_in[2]*"
                                             : ")*stride_out[2] + t*stride_out[2]*";
                        global_offset += std::to_string(blockWGS / blockWidth);

                        str += GlobalWriteStrCat(global_offset, placeness, outInterleaved);
                        strGlobalWrite = global_offset;
                        str += "\t\t}\n";
                    }
                }
                else // SBCR
                {
                    std::string global_offset
                        = output_offset + " + me + t*" + std::to_string(blockWGS);
                    str += GlobalWriteStrCat(global_offset, placeness, outInterleaved);
                }

                str += "\t}\n\n"; // "}" enclose the loop intrduced

                if(blockComputeType == BCT_R2C)
                {
                    str += "\tif (Tsbrc == SBRC_3D_FFT_ERC_TRANS_Z_XY)\n\t{";
                    str += "\n\t\tif(me < " + std::to_string(blockWidth) + ")\n\t\t{";
                    str += "\n\t\t\tunsigned int t = " + std::to_string(loopCount) + ";";
                    str += "\n\t" + strLdsRead + "\n";
                    str += GlobalWriteStrCat(strGlobalWrite, placeness, outInterleaved);
                    str += "\t\t}\n\t}\n";
                }

            } // end if blockCompute
        }

        void GenerateSingleGlobalKernel(std::string&            str,
                                        rocfft_result_placement placeness,
                                        bool                    inInterleaved,
                                        bool                    outInterleaved)
        {
            // use interleaved LDS when halfLds constraint absent
            bool ldsInterleaved = inInterleaved || outInterleaved;
            ldsInterleaved      = halfLds ? false : ldsInterleaved;
            ldsInterleaved      = blockCompute ? true : ldsInterleaved;

            // Base type
            std::string rType = RegBaseType<PR>(1);
            // Vector type
            std::string r2Type = RegBaseType<PR>(2);

            for(size_t d = 0; d < 2; d++)
            {
                bool fwd = d ? false : true;

                GenerateSingleGlobalKernelPrototype(str,
                                                    fwd,
                                                    placeness,
                                                    inInterleaved,
                                                    outInterleaved,
                                                    ldsInterleaved,
                                                    rType,
                                                    r2Type);
                str += "{\n";
                // Allocate LDS
                GenerateSingleGlobalKernelSharedMem(str, ldsInterleaved, placeness, rType, r2Type);

                GenerateSingleGlobalKernelBody(
                    str, fwd, placeness, inInterleaved, outInterleaved, rType, r2Type);

                str += "}\n\n"; // end the kernel

            } // end fwd, backward
        }

        /* =====================================================================
                Generate Main kernels: call passes
                Generate forward (fwd) cases and backward kernels
                Generate inplace and outof place kernels
            =================================================================== */
        void GenerateGlobalKernel(std::string& str)
        {
            str += "\n////////////////////////////////////////Global kernels\n";

            // inplace, support only: interleaved to interleaved, planar to planar
            if((!blockCompute) || (blockCompute && (blockComputeType == BCT_C2C)))
            {
                GenerateSingleGlobalKernel(str, rocfft_placement_inplace, true, true);
                GenerateSingleGlobalKernel(str, rocfft_placement_inplace, false, false);
            }

            // out of place, support all 4 combinations
            GenerateSingleGlobalKernel(str, rocfft_placement_notinplace, true, true);
            GenerateSingleGlobalKernel(str, rocfft_placement_notinplace, true, false);
            GenerateSingleGlobalKernel(str, rocfft_placement_notinplace, false, true);
            GenerateSingleGlobalKernel(str, rocfft_placement_notinplace, false, false);
        }

        Kernel(const FFTKernelGenKeyParams& paramsVal)
            : params(paramsVal)
        {

            /* in principle, the fft_N should be passed as a run-time parameter to
               kernel (with the name lengths)
               However, we have to take out the fft_N[0] (length) to calculate the pass,
               blockCompute related parameter at kernel generation stage
            */
            length           = params.fft_N[0];
            workGroupSize    = params.fft_workGroupSize;
            numTrans         = params.fft_numTrans;
            blockComputeType = params.blockComputeType;
            name_suffix      = params.name_suffix;

            // Check if it is R2C or C2R transform
            if(params.fft_inputLayout == rocfft_array_type_real)
                r2c = true;
            if(params.fft_outputLayout == rocfft_array_type_real)
                c2r = true;
            r2c2r = (r2c || c2r);

            if(r2c)
            {
                rcFull = ((params.fft_outputLayout == rocfft_array_type_complex_interleaved)
                          || (params.fft_outputLayout == rocfft_array_type_complex_planar))
                             ? true
                             : false;
            }
            if(c2r)
            {
                rcFull = ((params.fft_inputLayout == rocfft_array_type_complex_interleaved)
                          || (params.fft_inputLayout == rocfft_array_type_complex_planar))
                             ? true
                             : false;
            }

            rcSimple = params.fft_RCsimple;

            halfLds    = true;
            linearRegs = true;

            realSpecial = params.fft_realSpecial;

            blockCompute = params.blockCompute;

            // Make sure we can utilize all Lds if we are going to
            // use blocked columns to compute FFTs
            if(blockCompute)
            {
                assert(length <= 256); // 256 parameter comes from prototype experiments
                // largest length at which block column possible given 32KB LDS limit
                // if LDS limit is different this number need to be changed appropriately
                halfLds    = false;
                linearRegs = true;
            }

            assert(((length * numTrans) % workGroupSize) == 0);
            cnPerWI               = (numTrans * length) / workGroupSize;
            workGroupSizePerTrans = workGroupSize / numTrans;

            // !!!! IMPORTANT !!!! Keep these assertions unchanged, algorithm depend on
            // these to be true
            assert((cnPerWI * workGroupSize) == (numTrans * length));
            assert(cnPerWI <= length); // Don't do more than 1 fft per work-item

            // Breakdown into passes

            size_t LS = 1;
            size_t L;
            size_t R   = length;
            size_t pid = 0;

            // See if we can get radices from the lookup table, only part of pow2 is in
            // the table
            KernelCoreSpecs     kcs;
            std::vector<size_t> radices = kcs.GetRadices(length);
            size_t              nPasses = radices.size();

            if((params.fft_MaxWorkGroupSize >= 256) && (nPasses != 0))
            {
                for(size_t i = 0; i < nPasses; i++)
                {
                    size_t rad = radices[i];
                    // printf("length: %d, rad = %d, linearRegs=%d ", (int)length, (int)rad,
                    // linearRegs);
                    L = LS * rad;
                    R /= rad;

                    passes.push_back(Pass<PR>(i,
                                              length,
                                              rad,
                                              cnPerWI,
                                              L,
                                              LS,
                                              R,
                                              linearRegs,
                                              halfLds,
                                              r2c,
                                              c2r,
                                              rcFull,
                                              rcSimple,
                                              realSpecial));

                    LS *= rad;
                }
                assert(R == 1); // this has to be true for correct radix composition of the length
                numPasses = nPasses;
            }
            else
            {
                // printf("generating radix sequences\n");

                // Possible radices
                size_t cRad[] = {13, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1}; // Must be in descending order
                size_t cRadSize = (sizeof(cRad) / sizeof(cRad[0]));

                // Generate the radix and pass objects
                while(true)
                {
                    size_t rad;

                    assert(cRadSize >= 1);

                    // Picks the radices in descending order (biggest radix first)
                    for(size_t r = 0; r < cRadSize; r++)
                    {
                        rad = cRad[r];

                        if((rad > cnPerWI) || (cnPerWI % rad))
                            continue;

                        if(!(R % rad))
                            break;
                    }

                    assert((cnPerWI % rad) == 0);

                    L = LS * rad;
                    R /= rad;

                    radices.push_back(rad);
                    passes.push_back(Pass<PR>(pid,
                                              length,
                                              rad,
                                              cnPerWI,
                                              L,
                                              LS,
                                              R,
                                              linearRegs,
                                              halfLds,
                                              r2c,
                                              c2r,
                                              rcFull,
                                              rcSimple,
                                              realSpecial));

                    pid++;
                    LS *= rad;

                    assert(R >= 1);
                    if(R == 1)
                        break;
                } // end while
                numPasses = pid;
            }

            assert(numPasses == passes.size());
            assert(numPasses == radices.size());

#ifdef PARMETERS_TO_BE_READ

            ParamRead pr;
            ReadParameterFile(pr);

            radices.clear();
            passes.clear();

            radices   = pr.radices;
            numPasses = radices.size();

            LS = 1;
            R  = length;
            for(size_t i = 0; i < numPasses; i++)
            {
                size_t rad = radices[i];
                L          = LS * rad;
                R /= rad;

                passes.push_back(Pass<PR>(i, length, rad, cnPerWI, L, LS, R, linearRegs));

                LS *= rad;
            }
            assert(R == 1);
#endif

            // Grouping read/writes ok?
            bool grp = IsGroupedReadWritePossible();
            // printf("len=%d, grp = %d\n", length, grp);
            for(size_t i = 0; i < numPasses; i++)
                passes[i].SetGrouping(grp);

            // Store the next pass-object pointers
            if(numPasses > 1)
                for(size_t i = 0; i < (numPasses - 1); i++)
                    passes[i].SetNextPass(&passes[i + 1]);

            if(blockCompute)
            {
                blockLdsRowPadding
                    = (blockComputeType == BCT_R2C) ? 1 : 0; //TODO: move it some where else?
                BlockSizes::GetValue(length, blockWidth, blockWGS, blockLDS);
                blockLDS += blockLdsRowPadding * blockWidth;
            }
            else
            {
                blockWidth = blockWGS = blockLDS = blockLdsRowPadding = 0;
            }
        } // end of if ((params.fft_MaxWorkGroupSize >= 256) && (nPasses != 0))

        class BlockSizes
        {
        public:
            static void GetValue(size_t N, size_t& bwd, size_t& wgs, size_t& lds)
            {
                // wgs preferred work group size
                // bwd block width to be used
                // lds LDS size to be used for the block

                GetBlockComputeTable(N, bwd, wgs, lds);

                /*
                // bwd > t_nt is always ture, TODO: remove
                KernelCoreSpecs kcs;
                size_t t_wgs, t_nt;
                kcs.GetWGSAndNT(N, t_wgs, t_nt);
                wgs =  (bwd > t_nt) ? wgs : t_wgs;
                */

                // printf("N=%d, bwd=%d, wgs=%d, lds=%d\n", N, bwd, wgs, lds);
                // block width cannot be less than numTrans, math in other parts of code
                // depend on this assumption
                // assert(bwd >= t_nt);
            }
        };

        /* =====================================================================
            This is the main entrance to generate all device code.
            Notes:
                In this GenerateKernel function
                Real2Complex Complex2Real features are not available
                Callback features are not available
            =================================================================== */
        void GenerateKernel(std::string& str)
        {
            // str += "#include \"common.h\"\n";
            str += "#include \"rocfft_butterfly_template.h\"\n";
            if(blockCompute && blockComputeType == BCT_R2C)
            {
                str += "#include \"real2complex.h\"\n\n";
            }

            GeneratePassesKernel(str);

            GenerateEncapsulatedPassesKernel(str);

            GenerateGlobalKernel(str);
        }

        virtual CBPassType DeviceFuncLoadCBPassType()
        {
            return PassNormal;
        }
        virtual CBPassType DeviceFuncStoreCBPassType()
        {
            return PassNormal;
        }
    };

    // Single pass of a 2D_SINGLE kernel, either to do row transform or
    // column transform.  This generates templated kernels that don't
    // care about precision, so just hardcode Kernel's precision arg.
    class Kernel2D_SINGLE_pass : public Kernel<rocfft_precision_single>
    {
    public:
        Kernel2D_SINGLE_pass(const FFTKernelGenKeyParams& paramsVal, bool _isRowTransform)
            : Kernel(paramsVal)
            , isRowTransform(_isRowTransform)
        {
        }

        void GenerateSingleGlobalKernelRWFlag(std::string& str) override
        {
            str += "\t// set rw for enough threads to cover total number of 2D elements\n";
            str += "\tunsigned int rw = me < (lengths[0] * lengths[1] / " + std::to_string(cnPerWI)
                   + ");\n";
        }

        void GenerateSingleGlobalKernelIOOffsets(std::string&            str,
                                                 rocfft_result_placement placeness) override
        {
            if(isRowTransform)
            {
                str += "\t// row transform writes to LDS, so respect non-unit strides for input\n";
                str += "\t// and assume unit stride for output\n";
                str += "\tiOffset = dim == 2 ?\n";
                str += "\t\tbatch * _stride_in[2] :\n";
                str += "\t\tbatch / lengths[2] * _stride_in[3] + batch % lengths[2] * "
                       "_stride_in[2];\n";
            }
            else
            {
                str += "\t// col transform reads from LDS, so respect non-unit strides for "
                       "output\n";
                str += "\t// and assume unit stride for input\n";
                std::string outstride
                    = placeness == rocfft_placement_inplace ? "_stride_in" : "stride_out";
                str += "\toOffset = dim == 2 ?\n";
                str += "\t\tbatch * " + outstride + "[2] :\n";
                str += "\t\tbatch / lengths[2] * " + outstride + "[3] + batch % lengths[2] * "
                       + outstride + "[2];\n";
            }
            // HACK: we're doing a single 2D transform per
            // threadblock to/from LDS.  Convince the IO offset
            // generating code to assume everything is batch zero,
            // and use the code above to compensate for actual batch
            // location only on input or output
            size_t temp = 0;
            std::swap(temp, numTrans);
            Kernel<rocfft_precision_single>::GenerateSingleGlobalKernelIOOffsets(str, placeness);
            std::swap(temp, numTrans);
        }
        bool isRowTransform;

        CBPassType DeviceFuncLoadCBPassType() override
        {
            return isRowTransform ? PassNormal : PassNull;
        }
        CBPassType DeviceFuncStoreCBPassType() override
        {
            return isRowTransform ? PassNull : PassNormal;
        }
    };
    // Generate 2D kernels.  Thus far, we're only generating templated
    // kernels that don't need to care about precision
    class Kernel2D : public Kernel<rocfft_precision_single>
    {
    public:
        // size of first dimension is given in paramsVal, second
        // dimension needs to be specified separately.
        Kernel2D(const FFTKernelGenKeyParams& paramsVal1, const FFTKernelGenKeyParams& paramsVal2)
            : Kernel(paramsVal1)
            , transform_row(paramsVal1, true)
            , transform_col(paramsVal2, false)
        {
            // ensure the row transform knows it's being done for each
            // column, and vice-versa
            transform_row.numTrans = transform_col.length;
            transform_col.numTrans = transform_row.length;

            // rewrite the correct work group size for parent and child kernels
            size_t wgs = Get2DSingleThreadCount(
                paramsVal1.fft_N[0],
                paramsVal2.fft_N[0],
                [](size_t length, size_t& workGroupSize, size_t& numTransforms) {
                    return KernelCoreSpecs().GetWGSAndNT(length, workGroupSize, numTransforms);
                });
            this->workGroupSize         = wgs;
            transform_row.workGroupSize = wgs;
            transform_col.workGroupSize = wgs;
        }

    private:
        // give parameters underscore prefixes, since we define mutable
        // local variables with the normally-expected names
        bool StrideParamUnderscore() override
        {
            return true;
        }
        bool LengthParamUnderscore() override
        {
            return true;
        }
        bool IOParamUnderscore() override
        {
            return true;
        }

        std::string LaunchBounds() override
        {
            return "__launch_bounds__(" + std::to_string(this->workGroupSize) + ")\n";
        }

        std::string GlobalKernelFunctionSuffix() override
        {
            return "_2D_" + std::to_string(transform_row.length) + "_"
                   + std::to_string(transform_col.length);
        }

        void GenerateSingleGlobalKernelBody(std::string&            str,
                                            bool                    fwd,
                                            rocfft_result_placement placeness,
                                            bool                    inInterleaved,
                                            bool                    outInterleaved,
                                            const std::string&      rType,
                                            const std::string&      r2Type) override
        {
            str += "\t// use supplied input stride for row transform\n";
            str += "\tsize_t stride_in[4];\n";
            str += "\tstride_in[0] = _stride_in[0];\n";
            str += "\tstride_in[1] = _stride_in[1];\n";
            str += "\tstride_in[2] = _stride_in[2];\n";
            str += "\tstride_in[3] = _stride_in[3];\n";

            str += "\t// set unit output stride, since we're writing to LDS\n";
            str += "\tsize_t stride_out[4];\n";
            str += "\tstride_out[0] = 1;\n";
            str += "\tstride_out[1] = _lengths[0];\n";
            str += "\tstride_out[2] = _lengths[1];\n";
            str += "\tstride_out[3] = _lengths[2];\n";

            str += "\t// use supplied lengths for row transform\n";
            str += "\tsize_t lengths[3];\n";
            str += "\tlengths[0] = _lengths[0];\n";
            str += "\tlengths[1] = _lengths[1];\n";
            str += "\tlengths[2] = _lengths[2];\n";

            str += "\t// declare input/output pointers\n";
            if(placeness == rocfft_placement_inplace)
            {
                if(inInterleaved)
                {
                    str += "\tT* gbIn = _gb;\n";
                }
                else
                {
                    str += "\treal_type_t<T>* gbInRe = _gbRe;\n";
                    str += "\treal_type_t<T>* gbInIm = _gbIm;\n";
                }
            }
            else
            {
                if(inInterleaved)
                {
                    str += "\tT* gbIn = _gbIn;\n";
                }
                else
                {
                    str += "\treal_type_t<T>* gbInRe = _gbInRe;\n";
                    str += "\treal_type_t<T>* gbInIm = _gbInIm;\n";
                }
            }
            str += "\t// write to LDS\n";
            str += "\tT* gbOut = lds_data;\n";
            str += "\t// transform each row\n";
            str += "\t{\n";
            // force row transform to be out-of-place, interleaved output
            transform_row.GenerateSingleGlobalKernelBody(
                str, fwd, rocfft_placement_notinplace, inInterleaved, true, rType, r2Type);
            str += "\t}\n";

            // row transform is now done, set up column transform
            if(transform_row.length != transform_col.length)
            {
                str += "\t// we have two twiddle tables back to back in device\n";
                str += "\t// memory - move to the second table (if nonsquare)\n";
                str += "\ttwiddles = twiddles + lengths[0];\n";
            }

            if(placeness == rocfft_placement_notinplace)
            {
                str += "\t// write output to original out-of-place destination\n";
                str += "\tstride_out[0] = _stride_out[1];\n";
                str += "\tstride_out[1] = _stride_out[0];\n";
                str += "\tstride_out[2] = _stride_out[2];\n";
                str += "\tstride_out[3] = _stride_out[3];\n";
            }
            else
            {
                str += "\t// write output to original in-place destination\n";
                str += "\tstride_out[0] = _stride_in[1];\n";
                str += "\tstride_out[1] = _stride_in[0];\n";
                str += "\tstride_out[2] = _stride_in[2];\n";
                str += "\tstride_out[3] = _stride_in[3];\n";
            }
            str += "\t// get unit stride input from LDS\n";
            str += "\tstride_in[0] = _lengths[0];\n";
            str += "\tstride_in[1] = 1;\n";
            str += "\tstride_in[2] = _lengths[2];\n";
            str += "\tstride_in[3] = _lengths[3];\n";
            str += "\t\n";
            str += "\t// flip dimensions and transform each column\n\n";
            str += "\tauto temp = lengths[0];\n";
            str += "\tlengths[0] = lengths[1];\n";
            str += "\tlengths[1] = temp;\n";

            str += "\t// Let the row transform finish before starting column transform\n";
            str += "\t__syncthreads();\n";

            str += "\t// declare input/output pointers for column transform\n";
            if(!inInterleaved)
                str += "\tT* gbIn = lds_data;\n";
            else
                str += "\tgbIn = lds_data;\n";
            if(placeness == rocfft_placement_inplace)
            {
                if(outInterleaved)
                {
                    str += "\tgbOut = _gb;\n";
                }
                else
                {
                    str += "\treal_type_t<T>* gbOutRe = _gbRe;\n";
                    str += "\treal_type_t<T>* gbOutIm = _gbIm;\n";
                }
            }
            else
            {
                if(outInterleaved)
                {
                    str += "\tgbOut = _gbOut;\n";
                }
                else
                {
                    str += "\treal_type_t<T>* gbOutRe = _gbOutRe;\n";
                    str += "\treal_type_t<T>* gbOutIm = _gbOutIm;\n";
                }
            }
            str += "\t{\n";
            // for column transform, it's also out-of-place (since
            // the row transform results were written to LDS, and the
            // input is always interleaved
            transform_col.GenerateSingleGlobalKernelBody(
                str, fwd, rocfft_placement_notinplace, true, outInterleaved, rType, r2Type);
            str += "\t}\n";
        }

        size_t SharedMemSize(bool ldsInterleaved) override
        {
            // We're trying to do an entire 2D transform in a single
            // threadblock.  Each thread needs enough LDS space to do its
            // butterfly operations.
            //
            // LDS space counts in reals, but needs the same number
            // of elements as the complex transform has, since we
            // only store one of real/imag data at a time.
            return transform_row.length * transform_col.length;
        }

        void GenerateSingleGlobalKernelSharedMem(std::string&            str,
                                                 bool                    ldsInterleaved,
                                                 rocfft_result_placement placeness,
                                                 const std::string&      rType,
                                                 const std::string&      r2Type) override
        {
            Kernel<rocfft_precision_single>::GenerateSingleGlobalKernelSharedMem(
                str, ldsInterleaved, placeness, rType, r2Type);
            // also need to allocate LDS to store semi-transformed user data
            //
            // TODO: Technically this extra space is not necessary -
            // it's reasonable to put temporary butterfly data in LDS,
            // and write the semi-transformed data back to that same
            // LDS buffer.  But currently we have no easy way to make
            // the LDS usage follow the same stride pattern as the
            // strided column transform .
            str += "\t__shared__ T lds_data[" + std::to_string(transform_row.length) + "*"
                   + std::to_string(transform_col.length) + "];\n";
        }

        // details of the row and column transforms
        Kernel2D_SINGLE_pass transform_row;
        Kernel2D_SINGLE_pass transform_col;
    };
};

#endif
