/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "logging.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include "rocfft_ostream.hpp"
#include <fcntl.h>
#include <memory>

/*******************************************************************************
 * Static handle data
 ******************************************************************************/
int log_trace_fd    = -1;
int log_bench_fd    = -1;
int log_profile_fd  = -1;
int log_plan_fd     = -1;
int log_kernelio_fd = -1;
int log_rtc_fd      = -1;

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open a file descriptor for logging.
 *                  If the environment variable with name
 * environment_variable_name
 *                  is not set, then leave the fd untouched.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_fd      int&
 *              Output file descriptor.
 */

static void open_log_stream(const char* environment_variable_name, int& log_fd)

{
    // if environment variable is set, open file at logfile_pathname contained in
    // the
    // environment variable
    auto logfile_pathname = getenv(environment_variable_name);
    if(logfile_pathname)
    {
        log_fd = open(logfile_pathname, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    }
}

// library setup function, called once in program at the start of library use
rocfft_status rocfft_setup()
{
    // set layer_mode from value of environment variable ROCFFT_LAYER
    auto str_layer_mode = getenv("ROCFFT_LAYER");

    if(str_layer_mode)
    {
        rocfft_layer_mode layer_mode = static_cast<rocfft_layer_mode>(strtol(str_layer_mode, 0, 0));
        LogSingleton::GetInstance().SetLayerMode(layer_mode);

        // open log_trace file
        if(layer_mode & rocfft_layer_mode_log_trace)
            open_log_stream("ROCFFT_LOG_TRACE_PATH", log_trace_fd);

        // open log_bench file
        if(layer_mode & rocfft_layer_mode_log_bench)
            open_log_stream("ROCFFT_LOG_BENCH_PATH", log_bench_fd);

        // open log_profile file
        if(layer_mode & rocfft_layer_mode_log_profile)
            open_log_stream("ROCFFT_LOG_PROFILE_PATH", log_profile_fd);

        // open log_plan file
        if(layer_mode & rocfft_layer_mode_log_plan)
            open_log_stream("ROCFFT_LOG_PLAN_PATH", log_plan_fd);

        // open log_kernelio file
        if(layer_mode & rocfft_layer_mode_log_kernelio)
            open_log_stream("ROCFFT_LOG_KERNELIO_PATH", log_kernelio_fd);

        // open log_rtc file
        if(layer_mode & rocfft_layer_mode_log_rtc)
            open_log_stream("ROCFFT_LOG_RTC_PATH", log_rtc_fd);
    }

    log_trace(__func__);
    return rocfft_status_success;
}

// library cleanup function, called once in program after end of library use
rocfft_status rocfft_cleanup()
{
    log_trace(__func__);

    LogSingleton::GetInstance().SetLayerMode(rocfft_layer_mode_none);
    // Close log files
    if(log_trace_fd != -1)
    {
        close(log_trace_fd);
        log_trace_fd = -1;
    }
    if(log_bench_fd != -1)
    {
        close(log_bench_fd);
        log_bench_fd = -1;
    }
    if(log_profile_fd != -1)
    {
        close(log_profile_fd);
        log_profile_fd = -1;
    }
    if(log_plan_fd != -1)
    {
        close(log_plan_fd);
        log_plan_fd = -1;
    }
    if(log_kernelio_fd != -1)
    {
        close(log_kernelio_fd);
        log_kernelio_fd = -1;
    }
    if(log_rtc_fd != -1)
    {
        close(log_rtc_fd);
        log_rtc_fd = -1;
    }

    return rocfft_status_success;
}
