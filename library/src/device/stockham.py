"""Stockham kernel generator."""

import functools
import math
import sys

from collections import namedtuple
from math import ceil
from pathlib import Path
from types import SimpleNamespace as NS
from enum import Enum

from generator import *


#
# Helpers
#

LaunchParams = namedtuple('LaunchParams', ['transforms_per_block',
                                           'threads_per_block',
                                           'threads_per_transform'])


def kernel_launch_name(length, precision):
    """Return kernel name."""
    return f'rocfft_internal_dfn_{precision}_ci_ci_stoc_{length}'


def product(factors):
    """Return the product of the factors."""
    if factors:
        return functools.reduce(lambda a, b: a * b, factors)
    return 1


def quantize(n, granularity):
    """Round up a number 'n' to the next integer multiple of 'granularity'"""
    return granularity * ((n - 1) // granularity + 1)


def get_launch_params(factors,
                      flavour='uwide',
                      bytes_per_element=16,
                      lds_byte_limit=32 * 1024,
                      threads_per_block=256,
                      threads_per_transform=1,
                      **kwargs):
    """Return kernel launch parameters.

    Computes the maximum number of batches-per-block without:
    - going over 'lds_byte_limit' (32KiB by default) per block
    - going beyond 'threads_per_block' threads per block.
    """
    thread_granularity = 1

    length = product(factors)
    bytes_per_batch = length * bytes_per_element

    if flavour == 'uwide':
        threads_per_transform = length // min(factors)
    elif flavour == 'wide':
        threads_per_transform = length // max(factors)
    elif flavour == 'tall':
        pass

    batches_per_block = lds_byte_limit // bytes_per_batch
    while threads_per_transform * batches_per_block > threads_per_block:
        batches_per_block -= 1
    return LaunchParams(batches_per_block,
                        quantize(threads_per_transform * batches_per_block, thread_granularity),
                        threads_per_transform)


def get_callback_args():
    """Return callback argument list."""
    return ArgumentList(*[
        Variable('load_cb_fn', 'void', array=True, restrict=True),
        Variable('load_cb_data', 'void', array=True, restrict=True),
        Variable('load_cb_lds_bytes', 'uint32_t'),
        Variable('store_cb_fn', 'void', array=True, restrict=True),
        Variable('store_cb_data', 'void', array=True, restrict=True)])


def common_variables(length, params, nregisters):
    """Return namespace of common/frequent variables used in Stockham kernels."""
    kvars = NS(
        # templates
        scalar_type   = Variable('scalar_type', 'typename'),
        callback_type = Variable('cbtype', 'CallbackType'),
        stride_type   = Variable('sb', 'StrideBin'),
        # arguments
        buf         = Variable('buf', 'scalar_type', array=True, restrict=True),
        twiddles    = Variable('twiddles', 'const scalar_type', array=True, restrict=True),
        dim         = Variable('dim', 'const size_t'),
        lengths     = Variable('lengths', 'const size_t', array=True, restrict=True),
        stride      = Variable('stride', 'const size_t', array=True, restrict=True),
        nbatch      = Variable('nbatch', 'const size_t'),
        lds_padding = Variable('lds_padding', 'const unsigned int'),
        # locals
        lds        = Variable('lds', 'scalar_type',
                              size=length * params.transforms_per_block,
                              array=True, restrict=True, shared=True),
        block_id   = Variable('blockIdx.x'),
        thread_id  = Variable('threadIdx.x'),
        thread     = Variable('thread', 'size_t'),
        offset     = Variable('offset', 'size_t', value=0),
        offset_lds = Variable('offset_lds', 'unsigned int'),
        batch      = Variable('batch', 'size_t'),
        transform  = Variable('transform', 'size_t'),
        stride0    = Variable('stride0', 'const size_t'),
        # device
        W      = Variable('W', 'scalar_type'),
        t      = Variable('t', 'scalar_type'),
        R      = Variable('R', 'scalar_type', size=nregisters),
    )
    return kvars, kvars.__dict__


class CallbackDeclaration(BaseNode):
    def __str__(self):
        ret = 'auto load_cb = get_load_cb<scalar_type,cbtype>(load_cb_fn);'
        ret += 'auto store_cb = get_store_cb<scalar_type,cbtype>(store_cb_fn);'
        return ret


#
# Base helpers
#

class AdditionalArgumentMixin:
    """Add default template/argument list manipulation methods."""

    def add_templates(self, tlist, **kwargs):
        """Return list of extra template arguments."""
        return tlist

    def add_global_arguments(self, alist, **kwargs):
        """Return new function arguments to the global kernel function."""
        return alist

    def add_device_arguments(self, alist, **kwargs):
        """Return new function arguments to the device kernel function."""
        return alist

    def add_device_call_arguments(self, alist, **kwargs):
        """Return new function arguments for calling the device kernel function."""
        return alist


#
# Large twiddle table support
#

class StockhamLargeTwiddles(AdditionalArgumentMixin):
    """Base large twiddle table."""

    def load(self, length, params, **kwargs):
        return StatementList()

    def multiply(self, width, cumheight, **kwargs):
        return StatementList()


class StockhamLargeTwiddles2Step():
    """Two stage large twiddle table."""

    def __init__(self):
        self.apply_large_twiddle = Variable('apply_large_twiddle', 'bool')
        self.large_twiddle_base  = Variable('large_twiddle_base', 'size_t', value=8)
        self.large_twiddles      = Variable('large_twiddles', 'const scalar_type', array=True)
        self.trans_local         = Variable('trans_local', 'size_t')

    def add_templates(self, tlist, **kwargs):
        return tlist + TemplateList(self.apply_large_twiddle, self.large_twiddle_base)

    def add_global_arguments(self, alist, **kwargs):
        nargs = list(alist.args)
        nargs.insert(1, self.large_twiddles)
        return ArgumentList(*nargs)

    def add_device_arguments(self, alist, **kwargs):
        return alist + ArgumentList(self.large_twiddles, self.trans_local)

    def add_device_call_arguments(self, alist, transform=None, **kwargs):
        which = Ternary(And(self.apply_large_twiddle, self.large_twiddle_base < 8),
                        self.large_twd_lds, self.large_twiddles)
        return alist + ArgumentList(which, transform)

    def load(self, length, params,
             transform=None, dim=None,
             block_id=None, thread_id=None, lengths=None, stride=None, offset=None, batch=None,
             offset_lds=None,
             **kwargs):

        ltwd_id = Variable('ltwd_id', 'size_t', value=thread_id)
        ltwd_entries = Multiply(B(ShiftLeft(1, self.large_twiddle_base)), 3)
        ltwd_in_lds = And(self.apply_large_twiddle, Less(self.large_twiddle_base, 8))

        self.large_twd_lds = Variable('large_twd_lds', '__shared__ scalar_type',
                                      size=Ternary(ltwd_in_lds, ltwd_entries, 0))

        stmts = StatementList()
        stmts += Declarations(self.large_twd_lds)
        stmts += If(ltwd_in_lds,
                    StatementList(
                        Declaration(ltwd_id.name, ltwd_id.type, value=ltwd_id.value),
                        While(Less(ltwd_id, ltwd_entries),
                              StatementList(
                                  Assign(self.large_twd_lds[ltwd_id], self.large_twiddles[ltwd_id]),
                                  AddAssign(ltwd_id, params.threads_per_block)))))

        return stmts

    def multiply(self, width, cumheight,
                 W=None, t=None, R=None,
                 thread=None, scalar_type=None, **kwargs):
        stmts = StatementList()
        stmts += CommentLines('large twiddle multiplication')
        for w in range(width):
            idx = B(B(thread % cumheight) + w * cumheight) * self.trans_local
            stmts += Assign(W, InlineCall('TW2step',
                                          arguments=ArgumentList(self.large_twiddles, idx),
                                          templates=TemplateList(scalar_type, self.large_twiddle_base)))
            stmts += Assign(t.x, W.x * R[w].x - W.y * R[w].y)
            stmts += Assign(t.y, W.y * R[w].x + W.x * R[w].y)
            stmts += Assign(R[w], t)
        return If(self.apply_large_twiddle, stmts)


#
# Tilings
#

class StockhamTiling(AdditionalArgumentMixin):
    """Base tiling."""

    def calculate_offsets(self, *args, **kwargs):
        """Return code to calculate batch and buffer offsets."""
        return StatementList()

    def load_from_global(self, *args, **kwargs):
        """Return code to load from global buffer to LDS."""
        return StatementList()

    def store_to_global(self, *args, **kwargs):
        """Return code to store LDS to global buffer."""
        return StatementList()


class StockhamTilingRR(StockhamTiling):
    """Row/row tiling."""

    name = 'SBRR'

    def calculate_offsets(self, length, params,
                          lengths=None, stride=None,
                          dim=None, transform=None, block_id=None, thread_id=None,
                          batch=None, offset=None, offset_lds=None, **kwargs):

        d             = Variable('d', 'int')
        index_along_d = Variable('index_along_d', 'size_t')
        remaining     = Variable('remaining', 'size_t')
        plength       = Variable('plength', 'size_t', value=1)

        stmts = StatementList()
        stmts += Declarations(remaining, plength, index_along_d)
        stmts += Assign(transform, block_id * params.transforms_per_block + thread_id / params.threads_per_transform)
        stmts += Assign(remaining, transform)
        stmts += For(d.inline(1), d < dim, Increment(d),
                     StatementList(
                         Assign(plength, plength * lengths[d]),
                         Assign(index_along_d, remaining % lengths[d]),
                         Assign(remaining, remaining / lengths[d]),
                         Assign(offset, offset + index_along_d * stride[d])))
        stmts += Assign(batch, transform / plength)
        stmts += Assign(offset, offset + batch * stride[dim])
        stmts += Assign(offset_lds, length * B(transform % params.transforms_per_block))

        return stmts

    def load_from_global(self, length, params,
                         thread=None, thread_id=None, stride0=None,
                         buf=None, offset=None, lds=None, offset_lds=None,
                         **kwargs):
        width  = params.threads_per_transform
        height = length // width
        stmts = StatementList()
        stmts += Assign(thread, thread_id % width)
        for w in range(height):
            idx = thread + w * width
            stmts += Assign(lds[offset_lds + idx], LoadGlobal(buf, offset + B(idx) * stride0))
        return stmts

    def store_to_global(self, length, params,
                        thread=None, thread_id=None, stride0=None,
                        buf=None, offset=None, lds=None, offset_lds=None,
                        **kwargs):
        width  = params.threads_per_transform
        height = length // width
        stmts = StatementList()
        stmts += Assign(thread, thread_id % width)
        for w in range(height):
            idx = thread + w * width
            stmts += StoreGlobal(buf, offset + B(idx) * stride0, lds[offset_lds + idx])
        return If(thread < width, stmts)


class StockhamTilingCC(StockhamTiling):
    """Column/column tiling."""

    name = 'SBCC'

    def __init__(self):
        self.tile_index          = Variable('tile_index', 'size_t')
        self.tile_length         = Variable('tile_length', 'size_t')
        self.edge                = Variable('edge', 'bool')
        self.tid1                = Variable('tid1', 'size_t')
        self.tid0                = Variable('tid0', 'size_t')

    def calculate_offsets(self, length, params,
                          transform=None, dim=None,
                          block_id=None, thread_id=None, lengths=None, stride=None, offset=None, batch=None,
                          offset_lds=None,
                          **kwargs):

        d             = Variable('d', 'int')
        index_along_d = Variable('index_along_d', 'size_t')
        remaining     = Variable('remaining', 'size_t')
        plength       = Variable('plength', 'size_t', value=1)

        stmts = StatementList()
        stmts += Declarations(self.tile_index, self.tile_length)
        stmts += LineBreak()
        stmts += CommentLines('calculate offset for each tile:',
                              '  tile_index  now means index of the tile along dim1',
                              '  tile_length now means number of tiles along dim1')
        stmts += Declarations(plength, remaining, index_along_d)
        stmts += Assign(self.tile_length, B(lengths[1] - 1) / params.transforms_per_block + 1)
        stmts += Assign(plength, self.tile_length)
        stmts += Assign(self.tile_index, block_id % self.tile_length)
        stmts += Assign(remaining, block_id / self.tile_length)
        stmts += Assign(offset, self.tile_index * params.transforms_per_block * stride[1])
        stmts += For(d.inline(2), d < dim, Increment(d),
                     StatementList(
                         Assign(plength, plength * lengths[d]),
                         Assign(index_along_d, remaining % lengths[d]),
                         Assign(remaining, remaining / lengths[d]),
                         Assign(offset, offset + index_along_d * stride[d])))
        stmts += LineBreak()
        stmts += Assign(transform, self.tile_index * params.transforms_per_block + thread_id / params.threads_per_transform)
        stmts += Assign(batch, block_id / plength)
        stmts += Assign(offset, offset + batch * stride[dim])
        stmts += Assign(offset_lds, length * B(transform % params.transforms_per_block))

        return stmts

    def load_from_global(self, length, params,
                         buf=None, offset=None, lds=None,
                         lengths=None, thread_id=None, stride=None, stride0=None, **kwargs):

        edge, tid0, tid1 = self.edge, self.tid0, self.tid1
        stripmine_w   = params.transforms_per_block
        stripmine_h   = params.threads_per_block // stripmine_w
        stride_lds    = length + kwargs.get('lds_padding', 0)  # XXX

        stmts = StatementList()
        stmts += Declarations(edge, tid0, tid1)
        stmts += ConditionalAssign(edge,
                                   Greater(B(self.tile_index + 1) * params.transforms_per_block, lengths[1]),
                                   'true', 'false')
        stmts += Assign(tid1, thread_id % stripmine_w)  # tid0 walks the columns; tid1 walks the rows
        stmts += Assign(tid0, thread_id / stripmine_w)
        offset_tile_rbuf = lambda i : tid1 * stride[1]  + B(tid0 + i * stripmine_h) * stride0
        offset_tile_wlds = lambda i : tid1 * stride_lds + B(tid0 + i * stripmine_h) * 1
        pred, tmp_stmts = StatementList(), StatementList()
        pred = self.tile_index * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length // stripmine_h):
            tmp_stmts += Assign(lds[offset_tile_wlds(i)], LoadGlobal(buf, offset + offset_tile_rbuf(i)))

        stmts += If(Not(edge), tmp_stmts)
        stmts += If(edge, If(pred, tmp_stmts))

        return stmts

    def store_to_global(self, length, params,
                        stride=None, stride0=None, lengths=None, buf=None, offset=None, lds=None,
                        **kwargs):

        edge, tid0, tid1 = self.edge, self.tid0, self.tid1
        stripmine_w   = params.transforms_per_block
        stripmine_h   = params.threads_per_block // stripmine_w
        stride_lds    = length + kwargs.get('lds_padding', 0)  # XXX

        stmts = StatementList()
        offset_tile_rbuf = lambda i : tid1 * stride[1]  + B(tid0 + i * stripmine_h) * stride0
        offset_tile_wlds = lambda i : tid1 * stride_lds + B(tid0 + i * stripmine_h) * 1
        offset_tile_wbuf = offset_tile_rbuf
        offset_tile_rlds = offset_tile_wlds
        pred, tmp_stmts = StatementList(), StatementList()
        pred = self.tile_index * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length // stripmine_h):
            tmp_stmts += StoreGlobal(buf, offset + offset_tile_wbuf(i), lds[offset_tile_rlds(i)])
        stmts += If(Not(edge), tmp_stmts)
        stmts += If(edge, If(pred, tmp_stmts))

        return stmts


class StockhamTilingRC(StockhamTiling):
    pass


class StockhamTilingCR(StockhamTiling):
    pass


#
# Kernel helpers
#

def load_lds(width=None, height=None, spass=0,
             thread=None, R=None, lds=None, offset_lds=None, **kwargs):
    """Load registers 'R' from LDS 'X'."""
    stmts = StatementList()
    for w in range(width):
        idx = offset_lds + thread + w * height
        stmts += Assign(R[spass * width + w], lds[idx])
    stmts += LineBreak()
    return stmts


def twiddle(width=None, cumheight=None, spass=0,
            thread=None, W=None, t=None, twiddles=None, R=None, **kwargs):
    """Apply twiddles from 'T' to registers 'R'."""
    stmts = StatementList()
    for w in range(1, width):
        tidx = cumheight - 1 + w - 1 + (width - 1) * B(thread % cumheight)
        ridx = spass * width + w
        stmts += Assign(W, twiddles[tidx])
        stmts += Assign(t.x, W.x * R[ridx].x - W.y * R[ridx].y)
        stmts += Assign(t.y, W.y * R[ridx].x + W.x * R[ridx].y)
        stmts += Assign(R[ridx], t)
    stmts += LineBreak()
    return stmts


def butterfly(width=None, R=None, spass=0, **kwargs):
    """Apply butterly to registers 'R'."""
    stmts = StatementList()
    stmts += Call(name=f'FwdRad{width}B1',
                  arguments=ArgumentList(*[R[spass * width + w].address() for w in range(width)]))
    stmts += LineBreak()
    return stmts


def store_lds(width=None, cumheight=None, spass=0,
              lds=None, R=None, thread=None, offset_lds=None, **kwargs):
    """Store registers 'R' to LDS 'X'."""
    stmts = StatementList()
    for w in range(width):
        idx = offset_lds + B(thread / cumheight) * (width * cumheight) + thread % cumheight + w * cumheight
        stmts += Assign(lds[idx], R[spass * width + w])
    stmts += LineBreak()
    return stmts


#
# Stockham kernels
#

class StockhamKernel:
    """Base Stockham kernel."""

    def __init__(self, factors, scheme, tiling, large_twiddles, **kwargs):
        self.length = product(factors)
        self.factors = factors
        self.scheme = scheme
        self.tiling = tiling
        self.large_twiddles = large_twiddles
        self.kwargs = kwargs

    def templates(self, kvars, **kwvars):
        templates = TemplateList(kvars.scalar_type, kvars.stride_type, kvars.callback_type)
        templates = self.large_twiddles.add_templates(templates, **kwvars)
        templates = self.tiling.add_templates(templates, **kwvars)
        return templates

    def device_arguments(self, kvars, **kwvars):
        arguments = ArgumentList(kvars.lds, kvars.twiddles, kvars.stride0, kvars.offset_lds)
        arguments = self.large_twiddles.add_device_arguments(arguments, **kwvars)
        arguments = self.tiling.add_device_arguments(arguments, **kwvars)
        return arguments

    def device_call_arguments(self, kvars, **kwvars):
        arguments = ArgumentList(kvars.lds, kvars.twiddles, kvars.stride0, kvars.offset_lds)
        arguments = self.large_twiddles.add_device_call_arguments(arguments, **kwvars)
        arguments = self.tiling.add_device_call_arguments(arguments, **kwvars)
        return arguments

    def global_arguments(self, kvars, **kwvars):
        cb_args = get_callback_args()
        arguments = ArgumentList(kvars.twiddles, kvars.dim, kvars.lengths, kvars.stride, kvars.nbatch, kvars.lds_padding) \
                  + cb_args + ArgumentList(kvars.buf)
        arguments = self.large_twiddles.add_global_arguments(arguments, **kwvars)
        arguments = self.tiling.add_global_arguments(arguments, **kwvars)
        return arguments

    def generate_device_function(self):
        """Stockham device function."""
        pass

    def generate_global_function(self, **kwargs):
        """Global Stockham function."""

        use_3steps = kwargs.get('3steps')
        params     = get_launch_params(self.factors, **kwargs)

        kvars, kwvars = common_variables(self.length, params, self.nregisters)

        body = StatementList()
        body += CommentLines(
            f'this kernel:',
            f'  uses {params.threads_per_transform} threads per transform',
            f'  does {params.transforms_per_block} transforms per thread block',
            f'therefore it should be called with {params.threads_per_block} threads per thread block')
        body += Declarations(kvars.lds, kvars.offset, kvars.offset_lds, kvars.batch, kvars.transform, kvars.thread)
        body += Declaration(kvars.stride0.name, kvars.stride0.type,
                            value=Ternary(kvars.stride_type == 'SB_UNIT', 1, kvars.stride[0]))
        body += CallbackDeclaration()

        body += LineBreak()
        body += CommentLines('large twiddles')
        body += self.large_twiddles.load(self.length, params, **kwvars)

        body += LineBreak()
        body += CommentLines('offsets')
        body += self.tiling.calculate_offsets(self.length, params, **kwvars)

        body += LineBreak()
        body += If(GreaterEqual(kvars.batch, kvars.nbatch), [ReturnStatement()])

        body += LineBreak()
        body += CommentLines('load global')
        body += self.tiling.load_from_global(self.length, params, **kwvars)

        body += LineBreak()
        body += CommentLines('transform')
        body += Call(f'forward_length{self.length}_{self.tiling.name}_device',
                     arguments=self.device_call_arguments(kvars, **kwvars),
                     templates=self.templates(kvars, **kwvars))

        body += LineBreak()
        body += CommentLines('store global')
        body += SyncThreads()
        body += self.tiling.store_to_global(self.length, params, **kwvars)

        return Function(name=f'forward_length{self.length}_{self.tiling.name}',
                        qualifier=f'__global__ __launch_bounds__({params.threads_per_block})',
                        arguments=self.global_arguments(kvars, **kwvars),
                        templates=self.templates(kvars, **kwvars),
                        meta=NS(factors=self.factors,
                                length=self.length,
                                transforms_per_block=params.transforms_per_block,
                                threads_per_block=params.threads_per_block,
                                scheme=self.scheme,
                                use_3steps_large_twd=use_3steps,
                                transpose=None),
                        body=body)


class StockhamKernelUWide(StockhamKernel):
    """Stockham ultra-wide kernel.

    Each thread does at-most one butterfly.
    """

    @property
    def nregisters(self):
        return max(self.factors)

    def generate_device_function(self, **kwargs):
        factors, length, params = self.factors, self.length, get_launch_params(self.factors, **kwargs)
        kvars, kwvars = common_variables(self.length, params, self.nregisters)

        body = StatementList()
        body += Declarations(kvars.thread, kvars.R, kvars.W, kvars.t)
        body += Assign(kvars.thread, kvars.thread_id % (length // min(factors)))

        for npass, width in enumerate(factors):
            cumheight = product(factors[:npass])

            body += LineBreak()
            body += CommentLines(f'pass {npass}')
            body += SyncThreads()

            body += load_lds(width=width, height=length // width, **kwvars)

            if npass > 0:
                body += twiddle(width=width, cumheight=cumheight, **kwvars)

            body += butterfly(width=width, **kwvars)

            if npass == len(factors) - 1:
                body += self.large_twiddles.multiply(width, cumheight, **kwvars)

            body += SyncThreads()
            if width == min(factors):
                body += store_lds(width=width, cumheight=cumheight, **kwvars)
            else:
                body += If(kvars.thread < length // width,
                           store_lds(width=width, cumheight=cumheight, **kwvars))

        return Function(f'forward_length{length}_{self.tiling.name}_device',
                        arguments=self.device_arguments(kvars, **kwvars),
                        templates=self.templates(kvars, **kwvars),
                        body=body,
                        qualifier='__device__')


class StockhamKernelWide(StockhamKernel):
    """Stockham wide kernel.

    Each thread does at-least one butterfly.
    """

    @property
    def nregisters(self):
        return 2 * max(self.factors)

    def generate_device_function(self, **kwargs):
        factors, length, params = self.factors, self.length, get_launch_params(self.factors, **kwargs)
        kvars, kwvars = common_variables(length, params, self.nregisters)

        height0 = length // max(factors)

        def add_work(codelet, assign=True, **kwargs):
            stmts = StatementList()
            if assign:
                stmts += Assign(kvars.thread, kvars.thread_id % height0 + nsubpass * height0)
            if nsubpasses == 1 or nsubpass < nsubpasses - 1:
                stmts += codelet(**kwargs)
                return stmts
            needs_work = kvars.thread_id % height0 + nsubpass * height0 < length // width
            stmts += If(needs_work, codelet(**kwargs))
            return stmts

        body = StatementList()
        body += Declarations(kvars.thread, kvars.R, kvars.W, kvars.t)
        body += LineBreak()

        body += SyncThreads()
        body += LineBreak()

        for npass, width in enumerate(factors):
            cumheight = product(factors[:npass])
            nsubpasses = ceil(max(factors) / factors[npass])

            body += CommentLines(f'pass {npass}')

            if npass > 0:
                body += SyncThreads()
                body += LineBreak()

            for nsubpass in range(nsubpasses):
                body += add_work(load_lds, width=width, height=length // width, spass=nsubpass, **kwvars)

            if npass > 0:
                for nsubpass in range(nsubpasses):
                    body += add_work(twiddle, width=width, cumheight=cumheight, spass=nsubpass, **kwvars)
                body += LineBreak()

            for nsubpass in range(nsubpasses):
                body += add_work(butterfly, width=width, spass=nsubpass, assign=False, **kwvars)

            body += SyncThreads()
            for nsubpass in range(nsubpasses):
                body += add_work(store_lds, width=width, cumheight=cumheight, spass=nsubpass, **kwvars)

            body += LineBreak()

        return Function(f'forward_length{length}_{self.tiling.name}_device',
                        arguments=self.device_arguments(kvars, **kwvars),
                        templates=self.templates(kvars, **kwvars),
                        body=body,
                        qualifier='__device__')


class StockhamKernelTall(StockhamKernel):
    """Stockham tall kernel.

    Each thread does multiple butterflies.
    """

    @property
    def nregisters(self):
        self.threads_per_transform = self.kwargs.get('threads_per_transform', 1)
        return self.length // self.threads_per_transform

    def generate_device_function(self, **kwargs):
        factors, length, params = self.factors, self.length, get_launch_params(self.factors, **kwargs)
        kvars, kwvars = common_variables(length, params, self.nregisters)

        body = StatementList()
        body += Declarations(kvars.thread, kvars.R, kvars.W, kvars.t)
        body += LineBreak()

        height0 = self.nregisters
        for width in factors:
            if height0 % width != 0:
                raise RuntimeError(f"Can't use tall kernel: threads-per-transform {self.threads_per_transform} and factor {width} are incompatible.")
        # XXX also need to catch redundant case; ie 7 7 with tpt 1

        body += Assign(kvars.thread, kvars.thread_id % (length // height0))
        body += SyncThreads()
        body += LineBreak()

        for npass, width in enumerate(factors):
            cumheight = product(factors[:npass])
            height = length // width

            body += CommentLines(f'pass {npass}')
            if npass > 0:
                body += SyncThreads()
                body += LineBreak()

            for h in range(height0 // width):
                for w in range(width):
                    idx = kvars.offset_lds + kvars.thread * (height0 // width) + (w * height + h)
                    body += Assign(kvars.R[h * width + w], kvars.lds[idx])

            if npass > 0:
                W, t, R = kvars.W, kvars.t, kvars.R
                for h in range(height0 // width):
                    for w in range(1, width):
                        tid = B((height0 // width) * kvars.thread + h)
                        tidx = cumheight - 1 + w - 1 + (width - 1) * B(tid % cumheight)
                        ridx = h * width + w
                        body += Assign(W, kvars.twiddles[tidx])
                        body += Assign(t.x, W.x * R[ridx].x - W.y * R[ridx].y)
                        body += Assign(t.y, W.y * R[ridx].x + W.x * R[ridx].y)
                        body += Assign(R[ridx], t)
                body += LineBreak()

            for h in range(height0 // width):
                body += butterfly(width=width, spass=h, assign=False, **kwvars)

            body += SyncThreads()
            for h in range(height0 // width):
                for w in range(width):
                    tid = B((height0 // width) * kvars.thread + h)
                    idx = kvars.offset_lds + B(tid / cumheight) * (width * cumheight) + tid % cumheight + w * cumheight
                    body += Assign(kvars.lds[idx], kvars.R[h * width + w])

            body += LineBreak()

        return Function(f'forward_length{length}_{self.tiling.name}_device',
                        arguments=self.device_arguments(kvars, **kwvars),
                        templates=self.templates(kvars, **kwvars),
                        body=body,
                        qualifier='__device__')


#
# AST transforms
#

def make_variants(kdevice, kglobal):
    """Given in-place complex-interleaved kernels, create all other variations.

    The ASTs in 'kglobal' and 'kdevice' are assumed to be in-place,
    complex-interleaved kernels.

    Return out-of-place and planar variations.
    """
    op_names = ['buf', 'stride', 'stride0', 'offset']

    def rename(x, pre):
        if 'forward' in x or 'inverse' in x:
            return pre + x
        return x

    def rename_ip(x):
        return rename_functions(x, lambda n: rename(n, 'ip_'))

    def rename_op(x):
        return rename_functions(x, lambda n: rename(n, 'op_'))

    if kglobal.meta.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_RC':
        # out-of-place only
        kernels = [
            rename_op(make_out_of_place(kdevice, op_names)),
            rename_op(make_out_of_place(kglobal, op_names)),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
            rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in'))
            ]
        kdevice = make_inverse(kdevice, ['twiddles', 'TW2step'])
        kglobal = make_inverse(kglobal, ['twiddles', 'TW2step'])
        kernels += [
            rename_op(make_out_of_place(kdevice, op_names)),
            rename_op(make_out_of_place(kglobal, op_names)),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
            rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
        ]
        return kernels

    # XXX: Don't need in-place/out-of-place device functions anymore
    kernels = [
        # in-place, interleaved
        rename_ip(kdevice),
        rename_ip(kglobal),
        # in-place, planar
        rename_ip(make_planar(kglobal, 'buf')),
        # out-of-place, interleaved -> interleaved
        rename_op(make_out_of_place(kdevice, op_names)),
        rename_op(make_out_of_place(kglobal, op_names)),
        # out-of-place, interleaved -> planar
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
        # out-of-place, planar -> interleaved
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
        # out-of-place, planar -> planar
        rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
    ]

    kdevice = make_inverse(kdevice, ['twiddles', 'TW2step'])
    kglobal = make_inverse(kglobal, ['twiddles', 'TW2step'])

    kernels += [
        # in-place, interleaved
        rename_ip(kdevice),
        rename_ip(kglobal),
        # in-place, planar
        rename_ip(make_planar(kglobal, 'buf')),
        # out-of-place, interleaved -> interleaved
        rename_op(make_out_of_place(kdevice, op_names)),
        rename_op(make_out_of_place(kglobal, op_names)),
        # out-of-place, interleaved -> planar
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
        # out-of-place, planar -> interleaved
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
        # out-of-place, planar -> planar
        rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
    ]

    return kernels


#
# Interface!
#

def stockham_launch(factors, **kwargs):
    """Launch helper.  Not used by rocFFT proper."""

    length = product(factors)
    params = get_launch_params(factors, **kwargs)

    # arguments
    scalar_type   = Variable('scalar_type', 'typename')
    callback_type = Variable('CallbackType::NONE', 'CallbackType')
    stride_type   = Variable('SB_UNIT', 'StrideBin')
    inout         = Variable('inout', 'scalar_type', array=True)
    twiddles      = Variable('twiddles', 'const scalar_type', array=True)
    stride_in     = Variable('stride_in', 'size_t')
    stride_out    = Variable('stride_out', 'size_t')
    nbatch        = Variable('nbatch', 'size_t')
    lds_padding = Variable('lds_padding', 'unsigned int')
    kargs         = Variable('kargs', 'size_t*')
    null          = Variable('nullptr', 'void*')

    # locals
    nblocks = Variable('nblocks', 'int')

    body = StatementList()
    body += Declarations(nblocks)
    body += Assign(nblocks, B(nbatch + (params.transforms_per_block - 1)) / params.transforms_per_block)
    body += Call(f'forward_length{length}_SBRR',
                 arguments = ArgumentList(twiddles, 1, kargs, kargs + 1, nbatch, lds_padding, null, null, 0, null, null, inout),
                 templates = TemplateList(scalar_type, stride_type, callback_type),
                 launch_params = ArgumentList(nblocks, params.threads_per_block))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type),
                    arguments = ArgumentList(inout, nbatch, lds_padding, twiddles, kargs, stride_in, stride_out),
                    body = body)


def stockham_default_factors(length):
    supported_radixes = [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16]
    supported_radixes.sort(reverse=True)

    remaining_length = length
    factors = []
    for f in supported_radixes:
        while remaining_length % f == 0:
            factors.append(f)
            remaining_length /= f

    if remaining_length != 1:
        raise RuntimeError("length {} not factorizable!".format(length))

    # default order of factors is ascending
    factors.sort()
    return factors


def stockham(length, **kwargs):
    """Generate Stockham kernels!

    Returns a list of (device, global) function pairs.  This routine
    is essentially a factory...
    """

    kwargs['factors'] = factors = kwargs.get('factors', stockham_default_factors(length))
    kwargs.pop('factors')

    # assert that factors multiply out to the length
    if functools.reduce(lambda x, y: x * y, factors) != length:
        raise RuntimeError("invalid factors {} for length {}".format(factors, length))

    defualt_3steps = {
        'sp': 'false',
        'dp': 'false'
    }
    kwargs['3steps'] = kwargs.get('use_3steps_large_twd', defualt_3steps)

    scheme = kwargs['scheme']
    kwargs.pop('scheme')

    tiling = {
        'CS_KERNEL_STOCKHAM':          StockhamTilingRR(),
        'CS_KERNEL_STOCKHAM_BLOCK_CC': StockhamTilingCC(),
        'CS_KERNEL_STOCKHAM_BLOCK_RC': StockhamTilingRC(),
        'CS_KERNEL_STOCKHAM_BLOCK_CR': StockhamTilingCR(),
    }[scheme]

    twiddles = StockhamLargeTwiddles()
    if scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CC':
        twiddles = StockhamLargeTwiddles2Step()

    kernel = {
        'uwide': StockhamKernelUWide(factors, scheme, tiling, twiddles, **kwargs),
        'wide': StockhamKernelWide(factors, scheme, tiling, twiddles, **kwargs),
        'tall': StockhamKernelTall(factors, scheme, tiling, twiddles, **kwargs),
    }[kwargs.get('flavour', 'uwide')]

    kdevice = kernel.generate_device_function(**kwargs)
    kglobal = kernel.generate_global_function(**kwargs)

    return kdevice, kglobal
    