#include <iostream>
#include <memory>
#include <exception>

#include <cuda_runtime_api.h>

#include "NeuronRecord.h"
#include "Recorder.h"
#include "ModelRunner.h"
#include "ModelEngineContext.h"
#include "GpuModelCarrier.h"
#include "GpuModelHelper.h"
#include "GpuModelUi.h"
#include "ICommandControlAcceptor.h"
#include "CommandControlListenSocket.h"
#include "QueryResponseListenSocket.h"
#include "CommandControlHandler.h"

using std::cout;
using std::unique_ptr;
using std::make_unique;

using namespace embeddedpenguins::gpu::neuron::model;
using embeddedpenguins::core::neuron::model::KeyListener;
using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
using embeddedpenguins::core::neuron::model::CommandControlListenSocket;
using embeddedpenguins::core::neuron::model::QueryResponseListenSocket;
using embeddedpenguins::core::neuron::model::CommandControlHandler;
using embeddedpenguins::gpu::neuron::model::GpuModelCarrier;
using embeddedpenguins::gpu::neuron::model::GpuModelHelper;
using embeddedpenguins::gpu::neuron::model::GpuModelUi;

int RunServer();


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

    embeddedpenguins::core::neuron::model::Recorder<NeuronRecord>::Enable(false);

    if (!modelRunner.Run())
    {
        cout << "Cannot run model, stopping\n";
        return 1;
    }

    try
    {
        GpuModelUi ui(
            modelRunner, 
            std::move(make_unique<QueryResponseListenSocket>(
                "0.0.0.0", 
                "8000",
                [&modelRunner](){
                    cout << "Callback lambda creating new CommandControlHandler\n";
                    return std::move(make_unique<CommandControlHandler<NeuronRecord, ModelEngineContext<NeuronRecord>>>(modelRunner.Context()));
                }
            ))
        );
        ui.ParseArguments(argc, argv);
        ui.PrintAndListenForQuit();
    } catch (libsocket::socket_exception ex)
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
