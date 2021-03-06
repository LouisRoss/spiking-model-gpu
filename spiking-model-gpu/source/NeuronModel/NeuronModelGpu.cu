#include <cuda/runtime_api.hpp>
#include <cooperative_groups.h>

#include "NeuronCommon.h"
#include "NeuronNode.h"
#include "NeuronSynapse.h"


using namespace embeddedpenguins::gpu::neuron::model;

//
// The device model.
//
__device__ unsigned long int g_modelSize {};
__device__ NeuronNode* g_pNeurons {};
__device__ NeuronSynapse (*g_pSynapses)[SynapticConnectionsPerNode] {};


__device__ const float PostsynapticIncreaseFunction[PostsynapticPlasticityPeriod] = 
{ 2.0, 2.0, 2.0, 2.0, 1.800, 1.800, 1.650, 1.400, 1.400, 1.430, 1.385, 1.385, 1.360, 1.310, 1.310, 1.265, 1.200, 1.200, 1.140, 1.115, 1.115, 1.100, 1.085, 1.085, 1.060, 1.030, 1.030, 1.020, 1.010, 1.010 };

__device__ const float PostsynapticDecreaseFunction[PostsynapticPlasticityPeriod] = 
{ 1.0, 0.95, 0.95, 0.90, 0.85, 0.85, 0.80, 0.75, 0.75, 0.70, 0.65, 0.65, 0.60, 0.55, 0.55, 0.5, 0.58, 0.58, 0.66, 0.74, 0.74, 0.82, 0.89, 0.89, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0 };



/*********************************************************************************************************************/
/////////////////////////////////////////////////////// Helpers ///////////////////////////////////////////////////////

//
// Make the standard 1D launch configuration, [neurons].
//
cuda::launch_configuration_t MakeOnedimLaunchConfig(unsigned long int modelSize)
{
    const auto threadCount = 1024;
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

    return cuda::make_launch_config(grid_dims, block_dims);
}


//
// Make the standard 2D launch configuration, [neurons, synapses].
//
cuda::launch_configuration_t MakeTwodimLaunchConfig(unsigned long int modelSize)
{
    const auto threadCount = 32;
    const auto blockNeuronCount = ceil((float)modelSize/(float)threadCount);
    const auto blockSynapseCount = ceil((float)SynapticConnectionsPerNode/(float)threadCount);

	const cuda::grid::dimensions_t grid_dims = {
		cuda::grid::dimension_t(blockNeuronCount),
		cuda::grid::dimension_t(blockSynapseCount),
		cuda::grid::dimension_t(1)
	};
	const cuda::grid::dimensions_t block_dims = {
		cuda::grid::dimension_t(threadCount),
		cuda::grid::dimension_t(threadCount),
		cuda::grid::dimension_t(1)
	};

    return cuda::make_launch_config(grid_dims, block_dims);
}


/*********************************************************************************************************************/
///////////////////////////////////////////////////// DeviceFixup /////////////////////////////////////////////////////

//
// Device kernel.
// Since pointers are not valid across the host-device barrier, the host
// will wire connections using indexes rather than pointers.  Here we replace
// the indexes with device pointers directly to memory.
//
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

/*********************************************************************************************************************/
///////////////////////////////////////////////////// StreamInput /////////////////////////////////////////////////////

//
// Device kernel.
//
__global__ 
void
StreamInput(
    unsigned long int inputSize,
    unsigned long long int inputNeurons[])
{
    auto neuronIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuronIndex < inputSize)
    {
        auto inputNeuronId = inputNeurons[neuronIndex];
    
        if (inputNeuronId < g_modelSize)
        {
            auto& neuron = g_pNeurons[inputNeuronId];
    
            //printf("StreamInput: Neuron %ld, setting NextTickSpike true, and clamping activation at %d\n", inputNeuronId, ActivationThreshold + 1);
            neuron.NextTickSpike = true;
            neuron.Activation = ActivationThreshold + 1;
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
void
StreamInputShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    unsigned long int inputSize,
    unsigned long long int inputNeurons[])
{
	const auto kernel_function = StreamInput;
	cuda::kernel_t kernel(device, kernel_function);

    auto launch_configuration = MakeOnedimLaunchConfig(inputSize);
    
	cuda::launch(kernel, launch_configuration, inputSize, inputNeurons);
	cuda::device::current::get().synchronize();
}

/*********************************************************************************************************************/
//////////////////////////////////////////////////// ModelSynapses ////////////////////////////////////////////////////

//
// Device kernel.
// Examine all neurons that are presynaptic to the neuron indicated by the 1-dimensional block.x
// launch configuration.  Modify this neuron's activation level according to the states of the
// various presynaptic neurons.
// Also, mark each synapse that participated in modifying the neuron activation by starting a
// timer that can be used during learning.
// Also, mark any neuron that exceeds threshold with a boolean flag for processing in a later kernel.
//
// Modifies:
// NeuronNode.TicksSinceLastSpike
// NeuronNode.NextTickSpike
// NeruonNode.Activation
// NeuronSynapse.TickSinceLastSignal
//
__global__ void ModelSynapses(int synapseCount, int synapseBase)
{
    __shared__  int tempActivations[1024];

    auto neuronId = blockIdx.x;
    auto synapseOffset = threadIdx.x;
    auto synapseId =  synapseBase + synapseOffset;

    if (neuronId < g_modelSize && synapseId < SynapticConnectionsPerNode)
    {
        auto& neuron = g_pNeurons[neuronId];
        
        if (!IsInRecovery(neuron.TicksSinceLastSpike))
        {
            tempActivations[synapseOffset] = 0;
            auto& synapse = g_pSynapses[neuronId][synapseId];
            auto* presyapticNeuron = synapse.PresynapticNeuron;
            
#ifdef BRANCHLESS_CODE
            auto gate = presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime);
            auto eGate = (synapse.Type == SynapseType::Excitatory);
            auto iGate = (synapse.Type == SynapseType::Inhibitory);
            tempActivations[synapseOffset] = gate * (eGate * synapse.Strength - iGate * synapse.Strength);
            synapse.TickSinceLastSignal = gate * PostsynapticPlasticityPeriod + (1 - gate) * synapse.TickSinceLastSignal;
#else
            if (presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime))
            {
                if (synapse.Type == SynapseType::Excitatory)
                tempActivations[synapseOffset] = synapse.Strength;
                if (synapse.Type == SynapseType::Inhibitory)
                tempActivations[synapseOffset] = -synapse.Strength;
                
                synapse.TickSinceLastSignal = PostsynapticPlasticityPeriod;
            }
#endif

            __syncthreads();
            
            if (synapseOffset == 0)
            {
                //auto oldActivation = neuron.Activation;
                auto activation = neuron.Activation;
                for (auto activationId = 0; activationId < synapseCount; activationId++)
                {
                    activation += tempActivations[activationId];
                }
                
                neuron.Activation = activation;
                //printf("ModelSynapses: Neuron %d got signal from synapse %d with strength %d, changing activation from %d to %d\n", neuronId, synapseId, synapse.Strength, oldActivation, neuron.Activation);
                
                if (neuron.Activation > (ActivationThreshold + 1))
                {
                    //printf("ModelSynapses: Neuron %d above threshold (%d), setting NextTickSpike true, and clamping activation at %d\n", neuronId, neuron.Activation, ActivationThreshold + 1);
                    neuron.NextTickSpike = true;
                    neuron.Activation = ActivationThreshold + 1;
                }
                else if (neuron.Activation <= -ActivationThreshold)
                {
                    neuron.Activation = -ActivationThreshold;
                }
            }
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

    const int iterations = ceil((float)SynapticConnectionsPerNode/(float)1024);
    const int synapsesPerIteration = SynapticConnectionsPerNode / iterations;
    const int synapsesLastIteration = SynapticConnectionsPerNode - (synapsesPerIteration * (iterations - 1));

    for (auto iteration = 0; iteration < iterations; iteration++)
    {
        auto synapseCount = (iteration == iterations - 1) ? synapsesLastIteration : synapsesPerIteration;
        const cuda::grid::dimensions_t grid_dims = {
            cuda::grid::dimension_t(modelSize),
            cuda::grid::dimension_t(1),
            cuda::grid::dimension_t(1)
        };    
        const cuda::grid::dimensions_t block_dims = {
            cuda::grid::dimension_t(synapseCount),
            cuda::grid::dimension_t(1),
            cuda::grid::dimension_t(1)
        };    

        auto launch_configuration = cuda::make_launch_config(grid_dims, block_dims);
        cuda::launch(kernel, launch_configuration, synapseCount, iteration * synapsesPerIteration);
        
        device.synchronize();
    }    
}

/*********************************************************************************************************************/
///////////////////////////////////////////////////// ModelTimers /////////////////////////////////////////////////////

//
// Device kernel.
// Based on the outcomes from the ModelSynapses() kernel, capture the
// correct timer for a neuron, plus modify its activation.
//
// NOTE: This kernel depends on the ModelSynapses() kernel, and must not
//       be run in parallel with it.
//
// Modifies:
// NeuronNode.TicksSinceLastSpike
// NeuronNode.NextTickSpike
// NeruonNode.Activation
//
__global__ void ModelTimers()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuronId < g_modelSize)
    {
        auto* neuron = &g_pNeurons[neuronId];
        auto recoveryTime = neuron->TicksSinceLastSpike; 

        if (neuron->NextTickSpike)
        {
            //printf("ModelTimers: Neuron %d activation crossed spike threshold, setting TickSinceLastSpike to %d\n", neuronId, RecoveryTimeMax);
            neuron->TicksSinceLastSpike = RecoveryTimeMax;
        }
        else
        {
            if (IsRefractoryTick(recoveryTime))
            {
                //printf("ModelTimers: Neuron %d reached refractory tick, setting activation to 0\n", neuronId);
                neuron->Activation = 0;
            }
            else if (!IsInSpikeTime(recoveryTime))
            {
                //auto oldActivation = neuron->Activation;
                neuron->Activation = (short int)((double)neuron->Activation * DecayRate);
                //if (oldActivation != 0) printf("ModelTimers: Neuron %d decaying activation from %d to %d\n", neuronId, oldActivation, neuron->Activation);
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

    auto launch_configuration = MakeOnedimLaunchConfig(modelSize);
    
	cuda::launch(kernel, launch_configuration);
	cuda::device::current::get().synchronize();
}

/*********************************************************************************************************************/
/////////////////////////////////////////////////// ModelPlasticity ///////////////////////////////////////////////////

//
// Device kernel.
// Based on the outcomes from the ModelSynapses() and ModelTimers() kernels, 
//
// NOTE: This kernel depends on the ModelSynapses() and ModelTimers() kernels, 
//       and must not be run in parallel with either of them.
//
// Modifies:
//
__global__ void ModelPlasticity()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    auto synapseId = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuronId < g_modelSize && synapseId < SynapticConnectionsPerNode)
    {
        auto& neuron = g_pNeurons[neuronId];
        auto& synapse = g_pSynapses[neuronId][synapseId];

        if (neuron.NextTickSpike)
        {
            if (synapse.TickSinceLastSignal > 0)
            {
                auto newStrength = synapse.Strength * PostsynapticIncreaseFunction[synapse.TickSinceLastSignal - 1];
                if (newStrength > MaxSynapseStrength) newStrength = MaxSynapseStrength;
                if (newStrength < MinSynapseStrength) newStrength = MinSynapseStrength;
                //printf("ModelPlasticity: Neuron %d spiked, synapse %d was signaled %d ticks ago, changing synaptic strength from %d to %d\n", neuronId, synapseId, PostsynapticPlasticityPeriod - synapse.TickSinceLastSignal, (int)synapse.Strength, (int)newStrength);
                synapse.Strength = newStrength;
            }
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
void
ModelPlasticityShim(
    cuda::device_t& device,
    unsigned long int modelSize)
{
	const auto kernel_function = ModelPlasticity;
	cuda::kernel_t kernel(device, kernel_function);

    auto launch_configuration = MakeTwodimLaunchConfig(modelSize);

	cuda::launch(kernel, launch_configuration);
	cuda::device::current::get().synchronize();
}

/*********************************************************************************************************************/
////////////////////////////////////////////////////// ModelTick //////////////////////////////////////////////////////

//
// Device kernel.
// Based on the outcomes from the ModelSynapses() and ModelTimers() kernels, 
// Run all currently active timers in a neuron and its synapses.
//
// NOTE: This kernel depends on the ModelSynapses() and ModelTimers() kernels, 
//       and must not be run in parallel with either of them.
//
// Modifies:
// NeuronNode.TicksSinceLastSpike
// NeuronSynapse.TickSinceLastSignal
//
__global__ void ModelTick()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    auto synapseId = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuronId < g_modelSize && synapseId < SynapticConnectionsPerNode)
    {
        auto& neuron = g_pNeurons[neuronId];
        auto& synapse = g_pSynapses[neuronId][synapseId];

        if (synapseId == 0 && neuron.TicksSinceLastSpike > 0)
        {
            //auto oldTicks = neuron.TicksSinceLastSpike;
            neuron.TicksSinceLastSpike--;
            //printf("ModelTick:   Neuron %d ticking TicksSinceLastSpike for from %d to %d\n", neuronId, oldTicks, neuron.TicksSinceLastSpike);
        }
        
        neuron.NextTickSpike = false;

        if (synapse.TickSinceLastSignal > 0)
        {
            //auto oldTicks = synapse.TickSinceLastSignal;
            synapse.TickSinceLastSignal--;
            //printf("ModelTick:   Neuron %d synapse %d ticking TickSinceLastSignal for from %d to %d\n", neuronId, synapseId, oldTicks, synapse.TickSinceLastSignal);
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
void
ModelTickShim(
    cuda::device_t& device,
    unsigned long int modelSize)
{
	const auto kernel_function = ModelTick;
	cuda::kernel_t kernel(device, kernel_function);

    auto launch_configuration = MakeTwodimLaunchConfig(modelSize);
    
	cuda::launch(kernel, launch_configuration);
	cuda::device::current::get().synchronize();
}

