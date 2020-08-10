// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean buildStatic=false)
{
    project.paths.construct_build_prefix()

    String compiler = jobName.contains('hipclang') ? 'hipcc' : 'hcc'
    String clientArgs = '-DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SELFTEST=ON -DBUILD_CLIENTS_RIDER=ON'
    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String staticArg = buildStatic ? '-DBUILD_SHARED_LIBS=off' : ''
    String hipClangArgs = jobName.contains('hipclang') ? '-DUSE_HIP_CLANG=ON -DHIP_COMPILER=clang' : ''
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    String sudo = platform.jenkinsLabel.contains('sles') ? 'sudo -E' : ''

    def command = """#!/usr/bin/env bash
                set -x

                ls /fftw/lib

                export FFTW_ROOT=/fftw
                export FFTW_INCLUDE_PATH=\${FFTW_ROOT}/include
                export FFTW_LIB_PATH=\${FFTW_ROOT}/lib
                export LD_LIBRARY_PATH=\${FFTW_LIB_PATH}:\$LD_LIBRARY_PATH
                export CPLUS_INCLUDE_PATH=\${FFTW_INCLUDE_PATH}:\${CPLUS_INCLUDE_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3:\${CMAKE_PREFIX_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3f:\${CMAKE_PREFIX_PATH}

                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${sudo} ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/${compiler} ${buildTypeArg} ${clientArgs} ${hipClangArgs} ${staticArg} ../..
                ${sudo} make -j\$(nproc)
            """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, boolean debug=false)
{            
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String testBinaryName = debug ? 'rocfft-test-d' : 'rocfft-test'
    String directory = debug ? 'debug' : 'release'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/${directory}/clients/staging
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./${testBinaryName} --gtest_color=yes
            """
    platform.runCommand(this, command)
}

def runPackageCommand(platform, project, jobName, boolean debug=false)
{
    String directory = debug ? 'debug' : 'release'
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/${directory}",true)
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
