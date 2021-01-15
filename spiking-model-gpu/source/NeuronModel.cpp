#include <iostream>
#include <iomanip>
#include <memory>
#include <limits>

#include "NeuronModel.h"

using std::cout;

using namespace embeddedpenguins::neuron::infrastructure;

//
// Model parameters.  Any or all of these might become configurable in the future.
//
unsigned long int ModelSize { 10000 };

//
// The model.
//
//NeuronNode* pNeurons {};
//NeuronSynapse* pSynapses {};

//
// Forward reference of device objects.
//
void
DeviceFixupShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    NeuronNode neurons[],
    NeuronSynapse synapses[][SynapticConnectionsPerNode]);

namespace embeddedpenguins::neuron::model
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

        cuda::memory::copy(neuronsDevice_.get(), neuronsHost_.get(), modelSize_ * sizeof(NeuronNode));
        cuda::memory::copy(synapsesDevice_.get(), synapsesHost_.get(), modelSize_ * SynapticConnectionsPerNode * sizeof(NeuronSynapse));

        cout << "Calling DeviceFixupShim(" << device_.id() << ", " << ModelSize << ", " << "pNeurons, pSynapses)\n";
        DeviceFixupShim(device_, ModelSize, neuronsDevice_.get(), synapsesDevice_.get());
        cout << "Returned from DeviceFixupShim\n";

        // Test, remove.
        cuda::memory::copy(synapsesHost_.get(), synapsesDevice_.get(), modelSize_ * SynapticConnectionsPerNode * sizeof(NeuronSynapse));
        PrintSynapses(20);
    }

    void NeuronModel::PrintSynapses(int w)
    {
        const auto width = 100;
        const auto height = modelSize_ / width;

        for (auto row = 1; row < 10; row++)
        {
            for (auto col = 0; col < 10; col++)
            {
                auto& synapsesForNeuron = synapsesHost_[row * width + col];
                for (auto synapseId = 0; synapseId < 25; synapseId++)
                {
                    cout << std::setw(w) << (unsigned long int)synapsesForNeuron[synapseId].PresynapticNeuron;
                }
                cout << std::endl;
            }
            cout << std::endl;
        }
    }

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
}


void InitializeModel(std::unique_ptr<NeuronNode[]>& neurons, std::unique_ptr<NeuronSynapse[][SynapticConnectionsPerNode]>& synapses)
{
    const auto width = 100;
    const auto height = ModelSize / width;

    for (auto row = 1; row < width; row++)
    {
        for (auto col = 0; col < height; col++)
        {
            for (auto rowForPreviousCol = 0; rowForPreviousCol < width; rowForPreviousCol++)
            {
                auto& synapse = synapses.get()[col * width + row][rowForPreviousCol];
                *(unsigned long*)&synapse.PresynapticNeuron = (col - 1) * width + rowForPreviousCol;
            }
        }
    }
}

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
#if false
	cuda::device::id_t device_id = cuda::device::default_device_id;
	auto device = cuda::device::get(device_id).make_current();
	cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	auto pNeuronsHost = std::make_unique<NeuronNode[]>(ModelSize);
	auto pSynapsesHost = std::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(ModelSize);

    // Initialize model.
    InitializeModel(pNeuronsHost, pSynapsesHost);

    // Allocate device memory
	auto pNeurons = cuda::memory::device::make_unique<NeuronNode[]>(device, ModelSize);
	auto pSynapses = cuda::memory::device::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(device, ModelSize);
    //cudaMalloc((void**)&pNeurons, ModelSize * sizeof(NeuronNode));
    //cudaMalloc((void**)&pSynapses, ModelSize * SynapticConnectionsPerNode * sizeof(NeuronSynapse));

	cuda::memory::copy(pNeurons.get(), pNeuronsHost.get(), ModelSize * sizeof(NeuronNode));
	cuda::memory::copy(pSynapses.get(), pSynapsesHost.get(), ModelSize * SynapticConnectionsPerNode * sizeof(NeuronSynapse));
	//cuda::memory::copy(pNeurons, pNeuronsHost.get(), ModelSize * sizeof(NeuronNode));
	//cuda::memory::copy(pSynapses, pSynapsesHost.get(), ModelSize * SynapticConnectionsPerNode * sizeof(NeuronSynapse));


    cout << "Calling DeviceFixupShim(" << device.id() << ", " << ModelSize << ", " << "pNeurons, pSynapses)\n";
    DeviceFixupShim(device, ModelSize, pNeurons.get(), pSynapses.get());
    //DeviceFixupShim(device.id(), ModelSize, pNeurons, pSynapses);
    cout << "Returned from DeviceFixupShim\n";

    // Free memory on device
    //cudaFree(pSynapses);
    //cudaFree(pNeurons);
#else
    embeddedpenguins::neuron::model::NeuronModel model(ModelSize);
    model.InitializeModel();
#endif
    return 0;
}
