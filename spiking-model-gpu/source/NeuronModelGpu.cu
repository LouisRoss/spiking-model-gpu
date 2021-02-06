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
__device__ unsigned long int g_modelSize {};
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
        g_modelSize = modelSize;
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
    
	cuda::launch(kernel_function, launch_configuration, modelSize, neurons, synapses);
	cuda::device::current::get().synchronize();
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

    if (neuronId < g_modelSize)
    {
        auto* neuron = &g_pNeurons[neuronId];

        for (auto synapseId = 0; synapseId < PresynapticConnectionsPerNode; synapseId++)
        {
            auto* synapse = &g_pSynapses[neuronId][synapseId];
            auto* presyapticNeuron = synapse->PresynapticNeuron;
            if (presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike))
            {
                if (synapse->Type == SynapseType::Excitatory)
                    neuron->Activation += synapse->Strength;
                if (synapse->Type == SynapseType::Inhibitory)
                    neuron->Activation -= synapse->Strength;
    
                synapse->TickSinceLastSignal = SynapseSignalTimeMax;
            }
        }

        if (neuron->Activation > ActivationThreshold)
        {
            neuron->TicksSinceLastSpike = RecoveryTimeMax;
            neuron->Activation = ActivationThreshold;
        }

        if (neuron->Activation <= -ActivationThreshold)
        {
            neuron->Activation = -ActivationThreshold;
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
void
ModelSynapsesShim(
    cuda::device_t& device,
    unsigned long int modelSize)
{
	const auto kernel_function = ModelSynapses;
	cuda::kernel_t kernel(device, kernel_function);

    const auto threadCount = 256;
    const auto blockCount = ceil((float)modelSize/(float)threadCount);

	const cuda::grid::dimensions_t grid_dims = {
		cuda::grid::dimension_t(blockCount),
		cuda::grid::dimension_t(1),
		cuda::grid::dimension_t(1)
	};
	const cuda::grid::dimensions_t block_dims = {
		cuda::grid::dimension_t(threadCount),
		cuda::grid::dimension_t(1),
		cuda::grid::dimension_t(1)
	};
    auto launch_configuration = cuda::make_launch_config(grid_dims, block_dims);
    
	cuda::launch(kernel_function, launch_configuration);
	cuda::device::current::get().synchronize();
}

__global__ void ModelTimers()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuronId < g_modelSize)
    {
        auto* neuron = &g_pNeurons[neuronId];
        auto recoveryTime = neuron->TicksSinceLastSpike; 

        if (!IsInSpikeTime(recoveryTime))
        {
            neuron->Activation = (short int)((double)neuron->Activation * DecayRate);
        }

        if (IsRefractoryTick(recoveryTime))
        {
            neuron->Activation = 0;
        }

        if (IsInRecovery(recoveryTime))
        {
            neuron->TicksSinceLastSpike--;
        }

        for (auto synapseId = 0; synapseId < PresynapticConnectionsPerNode; synapseId++)
        {
            auto* synapse = &g_pSynapses[neuronId][synapseId];
            if (synapse->TickSinceLastSignal > 0)
            {
                synapse->TickSinceLastSignal--;
            }
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
void
ModelTimersShim(
    cuda::device_t& device,
    unsigned long int modelSize)
{
	const auto kernel_function = ModelTimers;
	cuda::kernel_t kernel(device, kernel_function);

    const auto threadCount = 256;
    const auto blockCount = ceil((float)modelSize/(float)threadCount);

	const cuda::grid::dimensions_t grid_dims = {
		cuda::grid::dimension_t(blockCount),
		cuda::grid::dimension_t(1),
		cuda::grid::dimension_t(1)
	};
	const cuda::grid::dimensions_t block_dims = {
		cuda::grid::dimension_t(threadCount),
		cuda::grid::dimension_t(1),
		cuda::grid::dimension_t(1)
	};
    auto launch_configuration = cuda::make_launch_config(grid_dims, block_dims);
    
	cuda::launch(kernel_function, launch_configuration);
	cuda::device::current::get().synchronize();
}

