#include <iostream>

#include <cuda_runtime_api.h>

#include "NeuronRecord.h"
#include "ModelRunner.h"
#include "GpuModelCarrier.h"
#include "GpuModelHelper.h"
#include "GpuModelUi.h"


using std::cout;

using namespace embeddedpenguins::gpu::neuron::model;
using embeddedpenguins::core::neuron::model::KeyListener;
using embeddedpenguins::gpu::neuron::model::GpuModelCarrier;
using embeddedpenguins::gpu::neuron::model::GpuModelHelper;
using embeddedpenguins::gpu::neuron::model::GpuModelUi;


///////////////////////////////////////////////////////////////////////////
//Main program entry.
//Run the brain map.
//
int main(int argc, char* argv[])
{
	if (cuda::device::count() == 0) {
		cout << "No CUDA devices on this system\n";
        return -1;
	}

    ModelRunner<NeuronRecord> modelRunner(argc, argv);
    const auto& configuration = modelRunner.getConfigurationRepository();

    GpuModelCarrier carrier;
    GpuModelHelper<NeuronRecord> helper(carrier, configuration);
    if (!modelRunner.Run(carrier, helper))
    {
        cout << "Cannot run model, stopping\n";
        return 1;
    }

    GpuModelUi ui(modelRunner, helper);
    ui.ParseArguments(argc, argv);
    ui.PrintAndListenForQuit();

    modelRunner.WaitForQuit();
    return 0;
}
