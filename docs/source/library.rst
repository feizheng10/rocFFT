
.. toctree::
   :maxdepth: 4 
   :caption: Contents:

======
rocFFT
======

Introduction
------------

The rocFFT library is an implementation of the discrete Fast Fourier Transform (FFT) written in HIP for GPU devices.
The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocFFT

The rocFFT library:

* Provides a fast and accurate platform for calculating discrete FFTs.
* Supports single and double precision floating point formats.
* Supports 1D, 2D, and 3D transforms.
* Supports computation of transforms in batches.
* Supports real and complex FFTs.
* Supports arbitrary lengths, with optimizations for combinations of
  powers of 2, 3, and 5.

FFT Computation
---------------

The FFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the DFT definition to
reduce the mathematical complexity from :math:`O(N^2)` to :math:`O(N \log N)`.

What is computed by the library? Here are the formulas:

For a 1D complex DFT:

:math:`{\tilde{x}}_j = \sum_{k=0}^{n-1}x_k\exp\left({\pm i}{{2\pi jk}\over{n}}\right)\hbox{ for } j=0,1,\ldots,n-1`

where, :math:`x_k` are the complex data to be transformed, :math:`\tilde{x}_j` are the transformed data, and the sign :math:`\pm`
determines the direction of the transform: :math:`-` for forward and :math:`+` for backward.

For a 2D complex DFT:

:math:`{\tilde{x}}_{jk} = \sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rq}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)`

for :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1`, where, :math:`x_{rq}` are the complex data to be transformed,
:math:`\tilde{x}_{jk}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.

For a 3D complex DFT:

:math:`\tilde{x}_{jkl} = \sum_{s=0}^{p-1}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rqs}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)\exp\left({\pm i}{{2\pi ls}\over{p}}\right)`

for :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1\hbox{ and } l=0,1,\ldots,p-1`, where :math:`x_{rqs}` are the complex data to
be transformed, :math:`\tilde{x}_{jkl}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.

Library Setup and Cleanup
-------------------------

At the beginning of the program, before any of the library APIs are called, the function :cpp:func:`rocfft_setup` has to be called. Similarly,
the function :cpp:func:`rocfft_cleanup` has to be called at the end of the program. These APIs ensure resources are properly allocated and freed.

Workflow
--------

In order to compute an FFT with rocFFT, a plan has to be created first. A plan is a handle to an internal data structure that
holds the details about the transform that the user wishes to compute. After the plan is created, it can be executed (a separate API call)
with the specified data buffers. The execution step can be repeated any number of times with the same plan on different input/output buffers
as needed. And when the plan is no longer needed, it gets destroyed.

To do a transform,

#. Initialize the library by calling :cpp:func:`rocfft_setup()`.
#. Create a plan, for each distinct type of FFT needed:

   * To create a plan, do either of the following

     * If the plan specification is simple, call :cpp:func:`rocfft_plan_create` and specify the value of the fundamental parameters.
     * If the plan has more details, first a plan description is created with :cpp:func:`rocfft_plan_description_create`, and additional APIs such
       as :cpp:func:`rocfft_plan_description_set_data_layout` are called to specify plan details. And then, :cpp:func:`rocfft_plan_create` is called
       with the description handle passed to it along with other details.

   * Optionally, allocate a work buffer for the plan:

     * Call :cpp:func:`rocfft_plan_get_work_buffer_size` to check the size of work buffer required by the plan.
     * If a nonzero size is required:

       * Create an execution info object with :cpp:func:`rocfft_execution_info_create`.
       * Allocate a buffer using :cpp:func:`hipMalloc` and pass the allocated buffer to :cpp:func:`rocfft_execution_info_set_work_buffer`.

#. Execute the plan:

   * The execution API :cpp:func:`rocfft_execute` is used to do the actual computation on the data buffers specified.
   * Extra execution information such as work buffers and compute streams are passed to :cpp:func:`rocfft_execute` in the :cpp:type:`rocfft_execution_info` object.
   * :cpp:func:`rocfft_execute` can be called repeatedly as needed for different data, with the same plan.
   * If the plan requires a work buffer but none was provided, :cpp:func:`rocfft_execute` will automatically allocate a work buffer and free it when execution is finished.

#. If a work buffer was allocated:

   * Call :cpp:func:`hipFree` to free the work buffer.
   * Call :cpp:func:`rocfft_execution_info_destroy` to destroy the execution info object.

#. Destroy the plan by calling :cpp:func:`rocfft_plan_destroy`.
#. Terminate the library by calling :cpp:func:`rocfft_cleanup()`.


Example
-------

.. code-block:: c

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "hip/hip_vector_types.h"
   #include "rocfft.h"
   
   int main()
   {
           // rocFFT gpu compute
           // ========================================
  
           rocfft_setup();

           size_t N = 16;
           size_t Nbytes = N * sizeof(float2);
   
           // Create HIP device buffer
           float2 *x;
           hipMalloc(&x, Nbytes);
   
           // Initialize data
           std::vector<float2> cx(N);
           for (size_t i = 0; i < N; i++)
           {
                   cx[i].x = 1;
                   cx[i].y = -1;
           }
   
           //  Copy data to device
           hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);
   
           // Create rocFFT plan
           rocfft_plan plan = nullptr;
           size_t length = N;
           rocfft_plan_create(&plan, rocfft_placement_inplace,
                rocfft_transform_type_complex_forward, rocfft_precision_single,
                1, &length, 1, nullptr);

	   // Check if the plan requires a work buffer
	   size_t work_buf_size = 0;
	   rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
	   void* work_buf = nullptr;
	   rocfft_execution_info info = nullptr;
	   if(work_buf_size)
           {
                   rocfft_execution_info_create(&info);
		   hipMalloc(&work_buf, work_buf_size);
		   rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
           }
   
           // Execute plan
           rocfft_execute(plan, (void**) &x, nullptr, info);
   
           // Wait for execution to finish
           hipDeviceSynchronize();

	   // Clean up work buffer
	   if(work_buf_size)
	   {
	           hipFree(work_buf);
		   rocfft_execution_info_destroy(info);
	   }

           // Destroy plan
           rocfft_plan_destroy(plan);
   
           // Copy result back to host
           std::vector<float2> y(N);
           hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);
   
           // Print results
           for (size_t i = 0; i < N; i++)
           {
                   std::cout << y[i].x << ", " << y[i].y << std::endl;
           }
   
           // Free device buffer
           hipFree(x);
   
           rocfft_cleanup();

           return 0;
   }

Plans
-----

A plan is the collection of (almost) all the parameters needed to specify an FFT computation. A rocFFT plan includes the
following information:

* Type of transform (complex or real)
* Dimension of the transform (1D, 2D, or 3D)
* Length or extent of data in each dimension
* Number of datasets that are transformed (batch size)
* Floating-point precision of the data
* In-place or not in-place transform
* Format (array type) of the input/output buffer
* Layout of data in the input/output buffer 

The rocFFT plan does not include the following parameters:

* The handles to the input and output data buffers.
* The handle to a temporary work buffer (if needed).
* Other information to control execution on the device.

These parameters are specified when the plan is executed.

Data
----

The input/output buffers that hold the data for the transform must be allocated, initialized and specified to the library by the
user. For larger transforms, temporary work buffers may be needed. Because the library tries to minimize its own allocation of
memory regions on the device, it expects the user to manage work buffers. The size of the buffer needed can be queried using
:cpp:func:`rocfft_plan_get_work_buffer_size` and after their allocation can be passed to the library by
:cpp:func:`rocfft_execution_info_set_work_buffer`. The samples in the source repository show how to use these.

Transform and Array types 
-------------------------

There are two main types of FFTs in the library:

* Complex FFT - Transformation of complex data (forward or backward); the library supports the following two
  array types to store complex numbers:

  #. Planar format - where the real and imaginary components are kept in 2 separate arrays:

     * Buffer1: ``RRRRR...`` 
     * Buffer2: ``IIIII...``
  #. Interleaved format - where the real and imaginary components are stored as contiguous pairs in the same array: 

     * Buffer: ``RIRIRIRIRIRI...``
  
* Real FFT - Transformation of real data. For transforms involving real data, there are two possibilities:

  * Real data being subject to forward FFT that results in complex data (Hermitian).
  * Complex data (Hermitian) being subject to backward FFT that results in real data.

The library provides the :cpp:enum:`rocfft_transform_type` and
:cpp:enum:`rocfft_array_type` enums to specify transform and array
types, respectively.

Batches
-------

The efficiency of the library is improved by utilizing transforms in batches. Sending as much data as possible in a single
transform call leverages the parallel compute capabilities of devices (GPU devices in particular), and minimizes the penalty
of control transfer. It is best to think of a device as a high-throughput, high-latency device. Using a networking analogy as
an example, this approach is similar to having a massively high-bandwidth pipe with very high ping response times. If the client
is ready to send data to the device for compute, it should be sent in as few API calls as possible, and this can be done by batching.
rocFFT plans have a parameter `number_of_transforms` (this value is also referred to as batch size in various places in the document)
in :cpp:func:`rocfft_plan_create` to describe the number of transforms being requested. All 1D, 2D, and 3D transforms can be batched.

.. _resultplacement:

Result placement
----------------

The API supports both in-place and not in-place transforms via the :cpp:enum:`rocfft_result_placement` enum.  With in-place transforms, only input buffers are provided to the
execution API, and the resulting data is written to the same buffer, overwriting the input data.  With not in-place transforms, distinct
output buffers are provided, and the results are written into the output buffer.

Note that rocFFT may still modify the input buffer even if a transform is requested to be not in-place.  Real-complex transforms in particular are more efficient if they can modify the original input.

Strides and Distances
---------------------

Strides and distances enable users to specify custom layout of data using :cpp:func:`rocfft_plan_description_set_data_layout`.

For 1D data, if :cpp:expr:`strides[0] == strideX == 1`, successive elements in the first dimension (dimension index 0) are stored
contiguously in memory. If :cpp:expr:`strideX` is a value greater than 1, gaps in memory exist between each element of the vector.
For multi-dimensional cases; if :cpp:expr:`strides[1] == strideY == LenX` for 2D data and :cpp:expr:`strides[2] == strideZ == LenX * LenY` for 3D data,
no gaps exist in memory between each element, and all vectors are stored tightly packed in memory. Here, :cpp:expr:`LenX`, :cpp:expr:`LenY`, and :cpp:expr:`LenZ` denote the
transform lengths :cpp:expr:`lengths[0]`, :cpp:expr:`lengths[1]`, and :cpp:expr:`lengths[2]`, respectively, which are used to set up the plan.

Distance is the stride that exists between corresponding elements of successive FFT data instances (primitives) in a batch. Distance is measured in units of the memory type;
complex data measures in complex units, and real data measures in real units. For tightly packed data, the distance between FFT primitives is the size of the FFT primitive,
such that :cpp:expr:`dist == LenX` for 1D data, :cpp:expr:`dist == LenX * LenY` for 2D data, and :cpp:expr:`dist == LenX * LenY * LenZ` for 3D data. It is possible to set the distance of a plan to be less than the size
of the FFT vector; typically 1 when doing column (strided) access on packed data. When computing a batch of 1D FFT vectors, if :cpp:expr:`distance == 1`, and :cpp:expr:`strideX == length(vector)`,
it means data for each logical FFT is read along columns (in this case along the batch). You must verify that the distance and strides are valid, such that each logical
FFT instance is not overlapping with any other; if not valid, undefined results may occur. A simple example would be to perform a 1D length 4096 on each row of an array of
1024 rows x 4096 columns of values stored in a column-major array, such as a FORTRAN program might provide. (This would be equivalent to a C or C++ program that has an
array of 4096 rows x 1024 columns stored in a row-major manner, on which you want to perform a 1D length 4096 transform on each column.) In this case, specify the
strides as [1024] and distance as 1.

Overwriting non-contiguous buffers
==================================

rocFFT guarantees that both the reading of FFT input and the writing of FFT output will respect the
specified strides.  However, temporary results can potentially be written to these buffers
contiguously, which may be unexpected if the strides would avoid certain memory locations completely
for reading and writing.

For example, a 1D FFT of length :math:`N` with input and output stride of 2 is transforming only
even-indexed elements in the input and output buffers.  But if temporary data needs to be written to
the buffers, odd-indexed elements may be overwritten.

However, rocFFT is guaranteed to respect the size of buffers.  In the above example, the
input/output buffers are :math:`2N` elements long, even if only :math:`N` even-indexed
elements are being transformed.  No more than :math:`2N` elements of temporary data will be written
to the buffers during the transform.

These policies apply to both input and output buffers, because :ref:`not in-place transforms may overwrite input data<resultplacement>`.

Transforms of real data
-----------------------

.. toctree::
   :maxdepth: 2

   real

Load and Store Callbacks
------------------------

rocFFT includes functionality to call user-defined device functions
when loading input from global memory at the start of a transform, or
when storing output to global memory at the end of a transform.

These user-defined callback functions may be optionally supplied
to the library using
:cpp:func:`rocfft_execution_info_set_load_callback` and
:cpp:func:`rocfft_execution_info_set_store_callback`.

Device functions supplied as callbacks must load and store element
data types that are appropriate for the transform being performed.

+-------------------------+--------------------+----------------------+
|Transform type           | Load element type  | Store element type   |
+=========================+====================+======================+
|Complex-to-complex,      | `float2`           | `float2`             |
|single-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-complex,      | `double2`          | `double2`            |
|double-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Real-to-complex,         | `float`            | `float2`             |
|single-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Real-to-complex,         | `double`           | `double2`            |
|double-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-real,         | `float2`           | `float`              |
|single-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-real,         | `double2`          | `double`             |
|double-precision         |                    |                      |
+-------------------------+--------------------+----------------------+

The callback function signatures must match the specifications
below.

.. code-block:: c

  T load_callback(T* buffer, size_t offset, void* callback_data, void* shared_memory);
  void store_callback(T* buffer, size_t offset, T element, void* callback_data, void* shared_memory);

The parameters for the functions are defined as:

* `T`: The data type of each element being loaded or stored from the
  input or output.
* `buffer`: Pointer to the input (for load callbacks) or
  output (for store callbacks) in device memory that was passed to
  :cpp:func:`rocfft_execute`.
* `offset`: The offset of the location being read from or written
  to.  This counts in elements, from the `buffer` pointer.
* `element`: For store callbacks only, the element to be stored.
* `callback_data`: A pointer value accepted by
  :cpp:func:`rocfft_execution_info_set_load_callback` and
  :cpp:func:`rocfft_execution_info_set_store_callback` which is passed
  through to the callback function.
* `shared_memory`: A pointer to an amount of shared memory requested
  when the callback is set.  Currently, shared memory is not supported
  and this parameter is always null.

Callback functions are called exactly once for each element being
loaded or stored in a transform.  Note that multiple kernels may be
launched to decompose a transform, which means that separate kernels
may call the load and store callbacks for a transform if both are
specified.

Currently, callbacks functions are only supported for transforms that
do not use planar format for input or output.
