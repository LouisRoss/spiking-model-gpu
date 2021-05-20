#include <iostream>
#include <memory>
#include <exception>
#include <functional>

#include <cuda_runtime_api.h>

#include "NeuronRecord.h"
#include "Recorder.h"
#include "ModelRunner.h"
#include "ModelEngineContext.h"
#include "GpuModelCarrier.h"
#include "GpuModelHelper.h"
#include "GpuModelUi.h"
#include "ICommandControlAcceptor.h"
#include "QueryResponseListenSocket.h"
#include "CommandControlHandler.h"

using std::cout;
using std::unique_ptr;
using std::make_unique;
using std::function;

using namespace embeddedpenguins::gpu::neuron::model;
using embeddedpenguins::core::neuron::model::KeyListener;
using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
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

    ModelRunner<NeuronRecord> modelRunner;

    embeddedpenguins::core::neuron::model::Recorder<NeuronRecord>::Enable(false);

    modelRunner.AddCommandControlAcceptor(
        std::move(make_unique<GpuModelUi>(modelRunner))
    );

    modelRunner.AddCommandControlAcceptor(
        std::move(make_unique<QueryResponseListenSocket>(
            "0.0.0.0", 
            "8000",
            [&modelRunner](function<void(const string&)> commandHandler){
                cout << "Callback lambda creating new CommandControlHandler\n";
                return std::move(make_unique<CommandControlHandler<NeuronRecord, ModelEngineContext<NeuronRecord>>>(modelRunner.Context(), commandHandler));
            }
        ))
    );

    try
    {
        if (!modelRunner.Initialize(argc, argv))
        {
            cout << "Cannot initialize model: " << modelRunner.Reason() << "\nstopping\n";
            return 1;
        }

        if (!modelRunner.Run())
        {
            cout << "Cannot run model: " << modelRunner.Reason() << "\nstopping\n";
            return 1;
        }

        modelRunner.RunCommandControl();
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
