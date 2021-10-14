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

    modelRunner.AddCommandControlAcceptor(
        std::move(make_unique<GpuModelUi>(modelRunner))
    );

    modelRunner.AddCommandControlAcceptor(
        std::move(make_unique<QueryResponseListenSocket>("0.0.0.0", "8000"))
    );

    try
    {
        cout << "Calling modelRunner.Initialize()\n";
        if (!modelRunner.Initialize(argc, argv))
        {
            cout << "Cannot initialize model: " << modelRunner.Reason() << "\nstopping\n";
            return 1;
        }

        if (!modelRunner.ControlFile().empty())
        {
            cout << "Model ControlFile not empty, calling modelRunner.RunWithExistingModel()\n";
            if (!modelRunner.RunWithExistingModel())
            {
                cout << "Cannot run model: " << modelRunner.Reason() << "\nstopping\n";
                return 1;
            }
        }

        cout << "Calling modelRunner.RunCommandControl()\n";
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
