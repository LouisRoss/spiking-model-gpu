#include <cuda/runtime_api.hpp>
#include <cooperative_groups.h>

#include "NeuronCommon.h"
#include "NeuronConnection.h"
#include "NeuronNode.h"
#include "NeuronSynapse.h"

using namespace embeddedpenguins::neuron::infrastructure;

//
// The device model.
//
__device__ NeuronNode* g_pNeurons {};
__device__ NeuronSynapse (*g_pSynapses)[SynapticConnectionsPerNode] {};


//
// Spike timing constants.
//
#define SynapseSignalTimeMax 20
#define RecoveryTimeMax 200
#define SpikeDuration 40
#define RampdownDuration 60

#define TimeSinceRecovery(_RecoveryTime) ((_RecoveryTime>0)? RecoveryTimeMax-_RecoveryTime: 255)
#define IsSpikeTick(_RecoveryTime) (_RecoveryTime == RecoveryTimeMax)
#define IsRefractoryTick(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) == SpikeDuration)
#define IsInSpikeTime(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) < SpikeDuration)
#define IsInRampdownTime(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) < RampdownDuration)
#define IsInRecovery(_RecoveryTime) (_RecoveryTime != 0)



__global__ 
void
DeviceFixup(
    unsigned long int modelSize,
    NeuronNode neurons[],
    NeuronSynapse synapses[][SynapticConnectionsPerNode])
{
    auto synapseId = blockIdx.x * blockDim.x + threadIdx.x;
    auto neuronId = blockIdx.y * blockDim.y + threadIdx.y;

	if( synapseId == 0 && neuronId == 0)
	{
		g_pNeurons = neurons;
        g_pSynapses = synapses;
	}

    NeuronNode* presynapticNeuron = nullptr;
    unsigned long int presynapticIndex = 0;

    presynapticNeuron = nullptr;
    presynapticIndex = (unsigned long int)synapses[neuronId][synapseId].PresynapticNeuron;
    if( (presynapticIndex != UINT_MAX) && (presynapticIndex < modelSize) )
    {
        // Normal, replace the index with a pointer.
        presynapticNeuron = &neurons[presynapticIndex];
    }

    // Coalesce memory writes.
    synapses[neuronId][synapseId].PresynapticNeuron = presynapticNeuron;
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
void
DeviceFixupShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    NeuronNode neurons[],
    NeuronSynapse synapses[][SynapticConnectionsPerNode])
{
	const auto kernel_function = DeviceFixup;
	cuda::kernel_t kernel(device, kernel_function);

	kernel.set_cache_preference(cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory);
	kernel.set_shared_memory_bank_size(cuda::multiprocessor_shared_memory_bank_size_option_t::four_bytes_per_bank);

	auto attributes = kernel.attributes();

	const cuda::grid::dimensions_t grid_dims = {
		cuda::grid::dimension_t(SynapticConnectionsPerNode),
		cuda::grid::dimension_t(modelSize),
		cuda::grid::dimension_t(1)
	};
    auto launch_configuration = cuda::make_launch_config(grid_dims, 1);
    
	cuda::launch(kernel_function, launch_configuration, modelSize, neurons, /*(NeuronSynapse (*)[SynapticConnectionsPerNode])*/synapses);
	cuda::device::current::get().synchronize();

    //unsigned int blocks = PostsynapticConnectionsPerNode / 1024 + 1;
    //DeviceFixup<<<blocks, 1024>>>(modelSize, neurons, synapses);
}

#if false
__global__ void ModelSynapses_reduce(unsigned long int modelSize)
{
    __shared__ LocalNeuronSynapse synapses[BlockSizeSynapse];

    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    auto synapseId = blockIdx.y * blockDim.y + threadIdx.y;

    synapses[threadIdx.y].Modifier = 0;     // Assume no active presynaptic neuron.
    NeuronNode* presyapticNeuron = g_pSynapses[neuronId][synapseId].PresynapticNeuron;
    if (presyapticNeuron != nullptr)
    {
        synapses[threadIdx.y].Strength = g_pSynapses[neuronId][synapseId].Strength;
        synapses[threadIdx.y].Modifier = IsSpikeTick(presyapticNeuron->TicksSinceLastSpike) ? 1 : 0;
        synapses[threadIdx.y].Modifier *= g_pSynapses[neuronId][synapseId].Type == SynapseType::Inhibitory ? -1 : 1;
    }
    synapses[threadIdx.y].NeuronActivation = g_pNeurons[neuronId].Activation;
    __syncthreads();

    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.y < stride)
        {
            synapses[threadIdx.y].NeuronActivation += (synapses[threadIdx.y].Strength * synapses[threadIdx.y].Modifier);
            synapses[threadIdx.y].NeuronActivation += (synapses[threadIdx.y + stride].Strength * synapses[threadIdx.y + stride].Modifier);
        }

        __syncthreads();
    }

    if (threadIdx.y == 0)
    {
        if (blockDim.y == 1)
        {
            g_pNeurons[neuronId].Activation = synapses[0].NeuronActivation;
        }
        else
        {
            g_SynapseScratchpad[blockIdx.y].NeuronActivation = synapses[0].NeuronActivation;
        }
    }
}
#endif

__global__ void ModelSynapses()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    auto* neuron = &g_pNeurons[neuronId];

    if (!IsInSpikeTime(neuron->TicksSinceLastSpike))
    {
        for (auto synapseId = 0; synapseId < PresynapticConnectionsPerNode; synapseId++)
        {
            auto* synapse = &g_pSynapses[neuronId][synapseId];
            auto* presyapticNeuron = synapse->PresynapticNeuron;
            if (presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike))
            {
                if (synapse->Type == SynapseType::Excitatory)
                    neuron->Activation += g_pSynapses[neuronId][synapseId].Strength;
                if (synapse->Type == SynapseType::Inhibitory)
                    neuron->Activation -= g_pSynapses[neuronId][synapseId].Strength;
    
                synapse->TickSinceLastSignal = SynapseSignalTimeMax;
            }
        }

        if (neuron->Activation >= ActivationThreshold)
        {
            neuron->TicksSinceLastSpike = RecoveryTimeMax;
            neuron->Activation = 0;
        }
    }
}
