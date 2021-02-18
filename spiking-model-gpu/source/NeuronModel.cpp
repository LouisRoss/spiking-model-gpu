#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <limits>

#include <cuda/runtime_api.hpp>

#include <cuda_runtime_api.h>
#include "cuda.h"

#include "NeuronRecord.h"
#include "ModelRunner.h"
//#include "core/KeyListener.h"
#include "GpuModelUi.h"

using std::cout;

using namespace embeddedpenguins::gpu::neuron::model;
using embeddedpenguins::core::neuron::model::KeyListener;

//
// Model parameters.  Any or all of these might become configurable in the future.
//
//unsigned long int ModelSize { 10000 };
//std::string cls("\033[2J\033[H");


#ifdef OLDSTYLE
//
// Forward reference of device objects.
//
void
DeviceFixupShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    NeuronNode neurons[],
    NeuronSynapse synapses[][SynapticConnectionsPerNode]);

void
ModelSynapsesShim(
    cuda::device_t& device,
    unsigned long int modelSize);

void
ModelTimersShim(
    cuda::device_t& device,
    unsigned long int modelSize);

namespace embeddedpenguins::gpu::neuron::model
{
    using std::numeric_limits;

    NeuronModel::NeuronModel(unsigned long int modelSize) :
            modelSize_(modelSize),
            deviceId_(cuda::device::default_device_id),
            device_(cuda::device::get(deviceId_).make_current()),
            neuronsHost_(std::make_unique<NeuronNode[]>(modelSize_)),
            synapsesHost_(std::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(modelSize_)),
            neuronsDevice_(cuda::memory::device::make_unique<NeuronNode[]>(device_, modelSize_)),
            synapsesDevice_(cuda::memory::device::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(device_, modelSize_))
    {
        cout << "Using CUDA device " << device_.name() << " (having device ID " << device_.id() << ")\n";

        // Initialize model.
        Initialize();
    }

    void NeuronModel::InitializeModel()
    {
        cout << "Initializing layers..." << std::flush;
        const auto width = 100;
        const auto height = modelSize_ / width;

        for (auto row = 1; row < height; row++)
        {
            for (auto col = 0; col < width; col++)
            {
                auto& synapsesForNeuron = synapsesHost_[row * width + col];
                for (auto colForPreviousRow = 0; colForPreviousRow < width; colForPreviousRow++)
                {
                    auto& synapse = synapsesForNeuron[colForPreviousRow];
                    *(unsigned long*)&synapse.PresynapticNeuron = (row - 1) * width + colForPreviousRow;
                }
            }
        }
        cout << "done\n";
        PrintSynapses(5);
        InitializeForTest();

        cuda::memory::copy(neuronsDevice_.get(), neuronsHost_.get(), modelSize_ * sizeof(NeuronNode));
        cuda::memory::copy(synapsesDevice_.get(), synapsesHost_.get(), modelSize_ * SynapticConnectionsPerNode * sizeof(NeuronSynapse));

        cout << "Calling DeviceFixupShim(" << device_.id() << ", " << ModelSize << ", " << "pNeurons, pSynapses)\n";
        DeviceFixupShim(device_, ModelSize, neuronsDevice_.get(), synapsesDevice_.get());
        cout << "Returned from DeviceFixupShim\n";

        // Test, remove.
        //cuda::memory::copy(synapsesHost_.get(), synapsesDevice_.get(), modelSize_ * SynapticConnectionsPerNode * sizeof(NeuronSynapse));
        //PrintSynapses(20);
                            ExecuteAStep();
                            ExecuteAStep();
                            ExecuteAStep();
                            ExecuteAStep();
                            ExecuteAStep();
    }

    void NeuronModel::Run()
    {
        constexpr char KEY_UP = 'A';
        constexpr char KEY_DOWN = 'B';
        constexpr char KEY_LEFT = 'D';
        constexpr char KEY_RIGHT = 'C';

        char c {' '};
        {
            KeyListener listener;

            bool quit {false};
            while (!quit)
            {
                auto gotChar = listener.Listen(50, c);
                if (gotChar)
                {
                    switch (c)
                    {
                        case KEY_UP:
                        case KEY_DOWN:
                        case KEY_LEFT:
                        case KEY_RIGHT:
                        case '=':
                        case '+':
                        case '-':
                            break;

                        case 's':
                        case 'S':
                            ExecuteAStep();
                            break;

                        case 'q':
                        case 'Q':
                            quit = true;
                            break;

                        default:
                            break;
                    }
                }
            }
        }

        cout << "Received keystroke " << c << ", quitting\n";
    }

    /////////////////////////////////////// Private methods //////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    void NeuronModel::Initialize()
    {
        cout << "Initializing..." << std::flush;
        for (auto neuronId = 0; neuronId < modelSize_; neuronId++)
        {
            for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
            {
                *(unsigned long*)&synapsesHost_[neuronId][synapseId].PresynapticNeuron = numeric_limits<unsigned long>::max();
                synapsesHost_[neuronId][synapseId].Strength = 0;
                synapsesHost_[neuronId][synapseId].TickSinceLastSignal = 0;
                synapsesHost_[neuronId][synapseId].Type = SynapseType::Excitatory;
            }

            neuronsHost_[neuronId].Type = NeuronType::Excitatory;
            neuronsHost_[neuronId].Activation = 0;
            neuronsHost_[neuronId].TicksSinceLastSpike = 0;
        }
        cout << "done\n";
    }

    void NeuronModel::ExecuteAStep()
    {
        cout << "Calling ModelSynapsesShim(" << device_.id() << ", " << ModelSize << ")\n";
        ModelSynapsesShim(device_, ModelSize);
        cout << "Returned from ModelSynapsesShim\n";

        cout << "Calling ModelTimersShim(" << device_.id() << ", " << ModelSize << ")\n";
        ModelTimersShim(device_, ModelSize);
        cout << "Returned from ModelTimersShim\n";

        cuda::memory::copy(neuronsHost_.get(), neuronsDevice_.get(), modelSize_ * sizeof(NeuronNode));
        cuda::memory::copy(synapsesHost_.get(), synapsesDevice_.get(), modelSize_ * SynapticConnectionsPerNode * sizeof(NeuronSynapse));

        cout << cls;
        PrintSynapses(20);
        PrintNeurons(5);
    }

    void NeuronModel::InitializeForTest()
    {
        const auto width = 100;
        const auto height = modelSize_ / width;

        neuronsHost_[0].TicksSinceLastSpike = 200;
        synapsesHost_[1 * width][0].Strength = 101;
        synapsesHost_[1 * width + 1][0].Strength = 50;
    }

    void NeuronModel::PrintSynapses(int w)
    {
        const auto width = 100;
        const auto height = modelSize_ / width;

        for (auto row = 1; row < 4; row++)
        {
            for (auto col = 0; col < 6; col++)
            {
                auto& synapsesForNeuron = synapsesHost_[row * width + col];
                for (auto synapseId = 0; synapseId < 5; synapseId++)
                {
                    cout << std::setw(w) << (unsigned long int)synapsesForNeuron[synapseId].PresynapticNeuron
                    << "(" << std::setw(3) << (unsigned int)synapsesForNeuron[synapseId].Strength << ")";
                }
                cout << std::endl;
            }
            cout << std::endl;
        }
    }

    void NeuronModel::PrintNeurons(int w)
    {
        const auto width = 100;
        const auto height = modelSize_ / width;

        for (auto row = 0; row < 10; row++)
        {
            for (auto col = 0; col < 10; col++)
            {
                auto& neuron = neuronsHost_[row * width + col];
                cout << std::setw(w) << neuron.Activation << "(" << std::setw(3) << neuron.TicksSinceLastSpike << ")";
            }
            cout << std::endl;
        }
    }
}
#endif //OLDSTYLE
std::string cls("\033[2J\033[H");
bool displayOn = false;

char PrintAndListenForQuit(ModelRunner<NeuronRecord>& modelRunner, GpuModelCarrier& carrier);
void PrintNeuronScan(ModelRunner<NeuronRecord>& modelRunner, GpuModelCarrier& carrier);
char MapIntensity(int activation);


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

    //embeddedpenguins::gpu::neuron::model::NeuronModel model(ModelSize);
    //model.InitializeModel();
    //model.Run();

    ModelRunner<NeuronRecord> modelRunner(argc, argv);
    GpuModelCarrier carrier;
    modelRunner.Run(carrier);

    //PrintAndListenForQuit(modelRunner, carrier);
    //modelRunner.WaitForQuit();

    GpuModelHelper<NeuronRecord> helper(carrier, modelRunner.getConfigurationRepository());
    GpuModelUi ui(modelRunner, helper);
    ui.ParseArguments(argc, argv);
    ui.PrintAndListenForQuit();

    modelRunner.WaitForQuit();
    return 0;
}

char PrintAndListenForQuit(ModelRunner<NeuronRecord>& modelRunner, GpuModelCarrier& carrier)
{
    char c;
    {
        auto listener = KeyListener();

        bool quit {false};
        while (!quit)
        {
            if (displayOn) PrintNeuronScan(modelRunner, carrier);
            quit = listener.Listen(50'000, c);
        }
    }

    cout << "Received keystroke " << c << ", quitting\n";
    return c;
}

void PrintNeuronScan(ModelRunner<NeuronRecord>& modelRunner, GpuModelCarrier& carrier)
{
    cout << cls;
/*
    auto node = begin(carrier.NeuronsHost.get());
    for (auto high = 25; high; --high)
    {
        for (auto wide = 50; wide; --wide)
        {
            cout << MapIntensity(node->Activation);
            node++;
        }
        cout << '\n';
    }
*/
    cout << "Iterations: " << modelRunner.getModelEngine().GetIterations() << "  Total work: " << modelRunner.getModelEngine().GetTotalWork() << "                 \n";
}

char MapIntensity(int activation)
{
    static int cutoffs[] = {2,5,15,50};

    if (activation < cutoffs[0]) return ' ';
    if (activation < cutoffs[1]) return '.';
    if (activation < cutoffs[2]) return '*';
    if (activation < cutoffs[3]) return 'o';
    return 'O';
}
