#include <iostream>
#include <memory>
#include <exception>
#include "libsocket/exception.hpp"

#include "cuda.h"               // For CUDA_VERSION
#if CUDA_VERSION < 12000
#include <cuda_runtime_api.h>       // Fixup after confirming to <cuda/runtime_api.h>
#endif

#include "ModelRunner.h"
#include "NeuronRecord.h"

using std::cout;

using libsocket::socket_exception;

using embeddedpenguins::gpu::neuron::model::ModelRunner;
using embeddedpenguins::gpu::neuron::model::NeuronRecord;


///////////////////////////////////////////////////////////////////////////
//Main program entry.
//Run the spiking neural model.
//
int main(int argc, char* argv[])
{
    int device_count = 0;
#if CUDA_VERSION < 12000
    device_count = cuda::device::count();
#else
    cudaGetDeviceCount(&device_count);
#endif
	if (device_count == 0) {
		cout << "No CUDA devices on this system\n";
        return -1;
	}

    ModelRunner<NeuronRecord> modelRunner;

    try
    {
        if (!modelRunner.Initialize(argc, argv))
        {
            cout << "Cannot initialize model: " << modelRunner.Reason() << "\nstopping\n";
            return 1;
        }

        modelRunner.RunCommandControl();
    }
    catch (socket_exception ex)
    {
        cout << "Caught exception " << ex.mesg << "\n";
    }
    catch (std::exception ex)
    {
        cout << "Caught exception " << ex.what() << "\n";
    }

    modelRunner.WaitForQuit();
    return 0;
}
