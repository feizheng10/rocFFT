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

#include "rtc.h"
#include "device/kernel-generator-embed.h"
#include "kernel_launch.h"
#include "logging.h"
#include "plan.h"
#include "tree_node.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <exception>
#include <map>

// specifications of stockham kernel
struct stockham_specs_t
{
    stockham_specs_t(const TreeNode& node, const char* gpu_arch, bool _enable_callbacks)
        : scheme(node.scheme)
        , length(node.length.front())
        , length2D(node.scheme == CS_KERNEL_2D_SINGLE ? node.length[1] : 0)
        , placement(node.placement)
        , direction(node.direction)
        , in_type(node.inArrayType)
        , out_type(node.outArrayType)
        , precision(node.precision)
        , unit_stride(node.inStride.front() == 1 && node.outStride.front() == 1)
        , apply_large_twiddle(node.large1D > 0)
        , large_twiddle_base(node.largeTwdBase)
        , ebtype(node.ebtype)
        , enable_callbacks(_enable_callbacks)
        , arch(gpu_arch)
    {
        set_kernel_name();
    }

    ComputeScheme           scheme;
    size_t                  length;
    size_t                  length2D;
    rocfft_result_placement placement;
    int                     direction;
    rocfft_array_type       in_type;
    rocfft_array_type       out_type;
    rocfft_precision        precision;
    bool                    unit_stride;
    bool                    apply_large_twiddle;
    size_t                  large_twiddle_base;
    EmbeddedType            ebtype;
    bool                    enable_callbacks;
    std::string             arch;

    std::string kernel_name;

private:
    // generate name for the desired variant of kernel.
    // NOTE: this is the key for finding kernels in the cache, so distinct
    // kernels *MUST* have unique names.
    void set_kernel_name()
    {
        kernel_name = "fft_rtc";

        if(direction == -1)
            kernel_name += "_fwd";
        else
            kernel_name += "_back";

        kernel_name += "_len";
        kernel_name += std::to_string(length);
        if(length2D)
            kernel_name += "x" + std::to_string(length2D);

        auto array_type_name = [](rocfft_array_type type) {
            switch(type)
            {
            case rocfft_array_type_complex_interleaved:
                return "_CI";
            case rocfft_array_type_complex_planar:
                return "_CP";
            case rocfft_array_type_real:
                return "_R";
            case rocfft_array_type_hermitian_interleaved:
                return "_HI";
            case rocfft_array_type_hermitian_planar:
                return "_HP";
            default:
                return "_UN";
            }
        };

        kernel_name += precision == rocfft_precision_single ? "_sp" : "_dp";

        if(placement == rocfft_placement_inplace)
        {
            kernel_name += "_ip";
            kernel_name += array_type_name(in_type);
        }
        else
        {
            kernel_name += "_op";
            kernel_name += array_type_name(in_type);
            kernel_name += array_type_name(out_type);
        }

        if(unit_stride)
            kernel_name += "_unitstride";

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
            kernel_name += "_sbcc";
        if(apply_large_twiddle)
            kernel_name += "_twdbase" + std::to_string(large_twiddle_base);

        switch(ebtype)
        {
        case EmbeddedType::NONE:
            break;
        case EmbeddedType::C2Real_PRE:
            kernel_name += "_C2R";
            break;
        case EmbeddedType::Real2C_POST:
            kernel_name += "_R2C";
            break;
        }
        if(enable_callbacks)
            kernel_name += "_CB";
    }
};

// wrapper object to handle reference counting
class PyObjWrap
{
public:
    PyObjWrap() {}
    PyObjWrap(PyObject* ptr)
    {
        reset(ptr);
    }
    // accept naked pointer
    PyObjWrap& operator=(PyObject* ptr)
    {
        reset(ptr);
        return *this;
    }
    // allow move
    PyObjWrap(PyObjWrap&& other)
    {
        reset(other.obj);
        other.obj = nullptr;
    }

    // disallow copy
    void operator=(const PyObjWrap&) = delete;
    PyObjWrap(const PyObjWrap&)      = delete;

    ~PyObjWrap()
    {
        reset();
    }
    void reset(PyObject* ptr = nullptr)
    {
        Py_XDECREF(obj);
        obj = ptr;
    }
    // add reference to a borrowed object, and take ownership of the ref
    void add_ref(PyObject* ptr)
    {
        Py_INCREF(ptr);
        reset(ptr);
    }
    bool is_null()
    {
        return obj == nullptr;
    }

    // auto-cast to pointer, to pass to python API
    operator PyObject*()
    {
        return obj;
    }

private:
    PyObject* obj = nullptr;
};

// wrapper object to handle interpreter initialize/finalize
struct PyInit
{
    PyInit()
    {
        Py_Initialize();
    }
    ~PyInit()
    {
        Py_Finalize();
    }
};

// structure we can turn into a static object that maintains python state
struct PythonGenerator
{
    PythonGenerator()
    {
        // compile each module and turn each into a module
        generator     = Py_CompileString(generator_py, "generator.py", Py_file_input);
        generator_mod = PyImport_ExecCodeModule("generator", generator);

        stockham     = Py_CompileString(stockham_py, "stockham.py", Py_file_input);
        stockham_mod = PyImport_ExecCodeModule("stockham", stockham);

        kernel_generator
            = Py_CompileString(kernel_generator_py, "kernel-generator.py", Py_file_input);
        kernel_generator_mod = PyImport_ExecCodeModule("kernel_generator", kernel_generator);

        // returns borrowed ref, so add a ref then pass to wrapper
        glob_dict.add_ref(PyModule_GetDict(kernel_generator_mod));
        local_dict = PyDict_New();

        // give static kernel prelude to generator
        std::string kernel_prelude_str = common_h;
        kernel_prelude_str += callback_h;
        kernel_prelude_str += butterfly_constant_h;
        kernel_prelude_str += rocfft_butterfly_template_h;
        kernel_prelude_str += real2complex_h;

        kernel_prelude = PyUnicode_FromString(kernel_prelude_str.c_str());
        PyMapping_SetItemString(local_dict, "kernel_prelude", kernel_prelude);

        // execute an expression to get supported lengths, call a
        // function for each length.  function accepts length obj and
        // kernel object.
        auto get_supported_lengths = [this](const char*                                   pyexpr,
                                            std::function<void(PyObjWrap&, PyObjWrap &&)> func) {
            PyObjWrap kernel_list = PyRun_String(pyexpr, Py_eval_input, glob_dict, local_dict);

            // extract all of the lengths from that list
            auto list_size = PyList_Size(kernel_list);
            for(Py_ssize_t i = 0; i < list_size; ++i)
            {
                PyObjWrap item;
                item.add_ref(PyList_GetItem(kernel_list, i));
                PyObjWrap length_obj = PyObject_GetAttrString(item, "length");

                // make sure runtime compilation is turned on for this kernel
                PyObjWrap rtc_flag = PyObject_GetAttrString(item, "runtime_compile");
                if(!PyObject_IsTrue(rtc_flag))
                    continue;

                func(length_obj, std::move(item));
            }
        };
        auto get_supported_lengths_1D
            = [=](const char* pyexpr, std::map<size_t, PyObjWrap>& output_lengths) {
                  get_supported_lengths(pyexpr, [&](PyObjWrap& length_obj, PyObjWrap&& kernel) {
                      output_lengths.emplace(PyLong_AsSize_t(length_obj), std::move(kernel));
                  });
              };
        get_supported_lengths_1D("default_runtime_compile(list_new_kernels())", supported_lengths);
        get_supported_lengths_1D("default_runtime_compile(list_new_large_kernels())",
                                 supported_lengths_large);
        get_supported_lengths("default_runtime_compile(list_new_2d_kernels())",
                              [this](PyObjWrap& length_obj, PyObjWrap&& kernel) {
                                  std::pair<size_t, size_t> len_2D{
                                      PyLong_AsSize_t(PyList_GetItem(length_obj, 0)),
                                      PyLong_AsSize_t(PyList_GetItem(length_obj, 1))};
                                  supported_lengths_2D.emplace(len_2D, std::move(kernel));
                              });
    }
    ~PythonGenerator() {}

    // outputs the python kernel object in kernel_obj
    bool length_is_supported(const stockham_specs_t& specs, PyObjWrap*& kernel_obj)
    {
        if(specs.scheme == CS_KERNEL_2D_SINGLE)
        {
            std::pair<size_t, size_t> length2D = {specs.length, specs.length2D};
            auto                      k        = supported_lengths_2D.find(length2D);
            if(k != supported_lengths_2D.end())
            {
                kernel_obj = &(k->second);
                return true;
            }
            return false;
        }

        std::map<size_t, PyObjWrap>::iterator k;
        if(specs.scheme == CS_KERNEL_STOCKHAM)
        {
            k = supported_lengths.find(specs.length);
            if(k == supported_lengths.end())
                return false;
        }
        else if(specs.scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            k = supported_lengths_large.find(specs.length);
            if(k == supported_lengths_large.end())
                return false;
        }
        else
            return false;
        kernel_obj = &(k->second);
        return true;
    }

    bool get_source(const stockham_specs_t& specs,
                    bool                    enable_callbacks,
                    PyObjWrap&              kernel_obj,
                    std::string&            kernel_src)
    {
        kernel_src.clear();

        // call python to generate the kernel

        // set this kernel object as a local called "k", possibly
        // overwriting any previous "k"
        PyMapping_SetItemString(local_dict, "k", kernel_obj);

        // get source code for kernel
        PyObjWrap py_src
            = PyRun_String("stockham.stockham_rtc(kernel_prelude, specs, **k.__dict__)",
                           Py_eval_input,
                           glob_dict,
                           local_dict);

        if(py_src.is_null())
        {
            PyErr_PrintEx(0);
            PyErr_Clear();
            throw std::runtime_error("error generating kernel");
        }

        Py_ssize_t  src_len = 0;
        const char* src     = PyUnicode_AsUTF8AndSize(py_src, &src_len);

        // need common.h and callback.h to define all of the template parameters
        kernel_src += "// ROCFFT_RTC_BEGIN " + specs.kernel_name;
        kernel_src += "\n";
        kernel_src += common_h;
        // callbacks must be enabled
        kernel_src += "\n#define ROCFFT_CALLBACKS_ENABLED\n";
        kernel_src += callback_h;
        // and we need butterfly helpers
        kernel_src += butterfly_constant_h;
        kernel_src += rocfft_butterfly_template_h;
        kernel_src.append(src, src_len);
        kernel_src += "\n// ROCFFT_RTC_END " + specs.kernel_name;
        kernel_src += "\n";
        return true;
    }

    // produce a dict of parameters for the desired variant of kernel
    PyObjWrap get_kernel_specs_dict(const stockham_specs_t& specs, bool enable_callbacks)
    {
        PyObjWrap dict = PyDict_New();

        PyObjWrap kernel_name_val = PyUnicode_FromString(specs.kernel_name.c_str());
        PyDict_SetItemString(dict, "kernel_name", kernel_name_val);

        PyObjWrap scheme_val = PyUnicode_FromString(PrintScheme(specs.scheme).c_str());
        PyDict_SetItemString(dict, "scheme", scheme_val);

        PyObjWrap length_val = PyLong_FromLong(specs.length);
        if(specs.length2D == 0)
        {
            // 1D kernel
            PyDict_SetItemString(dict, "length", length_val);
        }
        else
        {
            // 2D kernel needs tuple for length
            PyObjWrap length_tuple = PyTuple_New(2);
            PyTuple_SetItem(length_tuple, 0, PyLong_FromLong(specs.length));
            PyTuple_SetItem(length_tuple, 1, PyLong_FromLong(specs.length2D));
            PyDict_SetItemString(dict, "length", length_tuple);
        }

        PyDict_SetItemString(
            dict, "inplace", specs.placement == rocfft_placement_inplace ? Py_True : Py_False);

        PyObjWrap direction_val = PyLong_FromLong(specs.direction);
        PyDict_SetItemString(dict, "direction", direction_val);

        PyDict_SetItemString(
            dict, "input_is_planar", array_type_is_planar(specs.in_type) ? Py_True : Py_False);
        PyDict_SetItemString(
            dict, "output_is_planar", array_type_is_planar(specs.out_type) ? Py_True : Py_False);

        PyObjWrap real_type_val
            = PyUnicode_FromString(specs.precision == rocfft_precision_single ? "float" : "double");
        PyDict_SetItemString(dict, "real_type", real_type_val);

        PyObjWrap stridebin_val
            = PyUnicode_FromString(specs.unit_stride ? "SB_UNIT" : "SB_NONUNIT");
        PyDict_SetItemString(dict, "stridebin", stridebin_val);
        PyDict_SetItemString(
            dict, "apply_large_twiddle", specs.apply_large_twiddle ? Py_True : Py_False);
        PyObjWrap large_twiddle_base_val = PyLong_FromLong(specs.large_twiddle_base);
        PyDict_SetItemString(dict, "large_twiddle_base", large_twiddle_base_val);

        PyObjWrap cbtype_val = specs.enable_callbacks
                                   ? PyUnicode_FromString("CallbackType::USER_LOAD_STORE")
                                   : PyUnicode_FromString("CallbackType::NONE");
        PyDict_SetItemString(dict, "cbtype", cbtype_val);

        PyObjWrap ebtype_val;
        switch(specs.ebtype)
        {
        case EmbeddedType::NONE:
            ebtype_val.reset(PyUnicode_FromString("EmbeddedType::NONE"));
            break;
        case EmbeddedType::Real2C_POST:
            ebtype_val.reset(PyUnicode_FromString("EmbeddedType::Real2C_POST"));
            break;
        case EmbeddedType::C2Real_PRE:
            ebtype_val.reset(PyUnicode_FromString("EmbeddedType::C2Real_PRE"));
            break;
        }
        PyDict_SetItemString(dict, "ebtype", ebtype_val);

        PyDict_SetItemString(dict, "kernel_prelude", kernel_prelude);

        return dict;
    }

    // interpreter initialization
    PyInit init;

    // handles to the embedded modules
    PyObjWrap generator;
    PyObjWrap generator_mod;

    PyObjWrap stockham;
    PyObjWrap stockham_mod;

    PyObjWrap kernel_generator;
    PyObjWrap kernel_generator_mod;

    // variables we pass whenever we generate a kernel
    PyObjWrap kernel_prelude;

    // values we keep around so we can evaluate python expressions
    PyObjWrap glob_dict;
    PyObjWrap local_dict;

    // map length to python kernel object
    std::map<size_t, PyObjWrap>                    supported_lengths;
    std::map<size_t, PyObjWrap>                    supported_lengths_large;
    std::map<std::pair<size_t, size_t>, PyObjWrap> supported_lengths_2D;
};

RTCKernel::RTCKernel(const std::string& kernel_name, const std::vector<char>& code)
{
    if(hipModuleLoadData(&module, code.data()) != hipSuccess)
        throw std::runtime_error("failed to load module");

    if(hipModuleGetFunction(&kernel, module, kernel_name.c_str()) != hipSuccess)
        throw std::runtime_error("failed to get function");
}

std::vector<char> RTCKernel::compile(const std::string& kernel_src)
{
    hiprtcProgram prog;
    // give it a .cu extension so it'll be compiled as HIP code
    if(hiprtcCreateProgram(&prog, kernel_src.c_str(), "rocfft_rtc.cu", 0, nullptr, nullptr)
       != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("unable to create program");
    }

    std::vector<const char*> options;
    options.push_back("-O3");
    options.push_back("-std=c++14");

    auto compileResult = hiprtcCompileProgram(prog, options.size(), options.data());
    if(compileResult != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS)
                throw std::runtime_error(log.data());
        }
        throw std::runtime_error("compile failed without log");
    }

    size_t codeSize;
    if(hiprtcGetCodeSize(prog, &codeSize) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code size");

    std::vector<char> code(codeSize);
    if(hiprtcGetCode(prog, code.data()) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code");
    hiprtcDestroyProgram(&prog);
    return code;
}

hipError_t RTCKernel::launch(DeviceCallIn& data)
{
    // arguments get pushed in an array of 64-bit values
    std::vector<void*> kargs;

    // twiddles
    kargs.push_back(data.node->twiddles.data());
    // large 1D twiddles
    if(data.node->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        kargs.push_back(data.node->twiddles_large.data());
    // dim
    kargs.push_back(reinterpret_cast<void*>(data.node->length.size()));
    // lengths
    kargs.push_back(data.node->devKernArg.data());
    // stride in/out
    kargs.push_back(data.node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH);
    if(data.node->placement == rocfft_placement_notinplace)
        kargs.push_back(data.node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH);
    // nbatch
    kargs.push_back(reinterpret_cast<void*>(data.node->batch));
    // lds padding
    kargs.push_back(reinterpret_cast<void*>(data.node->lds_padding));
    // callback params
    kargs.push_back(data.callbacks.load_cb_fn);
    kargs.push_back(data.callbacks.load_cb_data);
    kargs.push_back(reinterpret_cast<void*>(data.callbacks.load_cb_lds_bytes));
    kargs.push_back(data.callbacks.store_cb_fn);
    kargs.push_back(data.callbacks.store_cb_data);

    // buffer pointers
    kargs.push_back(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.push_back(data.bufIn[1]);
    if(data.node->placement == rocfft_placement_notinplace)
    {
        kargs.push_back(data.bufOut[0]);
        if(array_type_is_planar(data.node->outArrayType))
            kargs.push_back(data.bufOut[1]);
    }

    auto  size     = sizeof(kargs.size() * sizeof(void*));
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      kargs.data(),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      HIP_LAUNCH_PARAM_END};

    const auto& gp = data.gridParam;

    return hipModuleLaunchKernel(kernel,
                                 gp.b_x,
                                 gp.b_y,
                                 gp.b_z,
                                 gp.tpb_x,
                                 gp.tpb_y,
                                 gp.tpb_z,
                                 gp.lds_bytes,
                                 nullptr,
                                 nullptr,
                                 config);
}

// singleton accessor
static PythonGenerator& get_generator()
{
    static PythonGenerator generator;
    return generator;
}

std::unique_ptr<RTCKernel>
    RTCKernel::runtime_compile(TreeNode& node, const char* gpu_arch, bool enable_callbacks)
{
    stockham_specs_t specs(node, gpu_arch, enable_callbacks);

    PyObjWrap* kernel_obj = nullptr;
    auto&      generator  = get_generator();
    if(!generator.length_is_supported(specs, kernel_obj))
        return nullptr;

    // give info to generator so it can construct the right variant
    PyObjWrap dict = generator.get_kernel_specs_dict(specs, enable_callbacks);
    PyMapping_SetItemString(generator.local_dict, "specs", dict);

    std::vector<char> code;

    // get source for the kernel and build it
    std::string kernel_src;
    generator.get_source(specs, enable_callbacks, *kernel_obj, kernel_src);

    if(LOG_RTC_ENABLED())
        (*LogSingleton::GetInstance().GetRTCOS()) << kernel_src << std::flush;

    // compile to code object
    code = compile(kernel_src);

    PyObjWrap code_py = PyBytes_FromStringAndSize(code.data(), code.size());
    PyMapping_SetItemString(generator.local_dict, "code", code_py);

    return std::unique_ptr<RTCKernel>(new RTCKernel(specs.kernel_name, code));
}
