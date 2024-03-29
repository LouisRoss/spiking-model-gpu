//#include <cuda/runtime_api.hpp>
#include <tuple>
#include <cooperative_groups.h>

#include "NeuronCommon.h"
#include "NeuronNode.h"
#include "NeuronPostSynapse.h"
#include "NeuronPreSynapse.h"


using namespace embeddedpenguins::gpu::neuron::model;

//
// The device model.
//
__device__ unsigned long int g_modelSize {};
__device__ float* g_synapticIncreaseFunction {};
__device__ NeuronNode* g_pNeurons {};
__device__ NeuronPostSynapse (*g_pPostSynapses)[SynapticConnectionsPerNode] {};
__device__ NeuronPreSynapse (*g_pPreSynapses)[SynapticConnectionsPerNode] {};


__device__ const float PostsynapticIncreaseFunction[PostsynapticPlasticityPeriod] = 
{ 2.0,   2.0,   2.0,   2.0,   1.800, 
  1.800, 1.650, 1.400, 1.400, 1.430, 
  1.385, 1.385, 1.360, 1.310, 1.310, 
  1.265, 1.200, 1.200, 1.140, 1.115, 
  1.115, 1.100, 1.085, 1.085, 1.060, 
  1.030, 1.030, 1.020, 1.010, 1.010 };

__device__ const float PostsynapticDecreaseFunction[RecoveryTimeMax] = 
{ 1.0, 0.95, 0.95, 0.90, 0.85,  
  0.85, 0.80, 0.75, 0.75, 0.70,  

  0.65, 0.65, 0.60, 0.55, 0.55, 
  0.5, 0.58, 0.58, 0.66, 0.74,  

  0.74, 0.82, 0.89, 0.89, 0.96,  
  1.0, 1.0, 1.0, 1.0, 1.0,

  1.0, 1.0, 1.0, 1.0, 1.0,
  1.0, 1.0, 1.0, 1.0, 1.0,

  1.0, 1.0, 1.0, 1.0, 1.0,
  1.0, 1.0, 1.0, 1.0, 1.0 };



/*********************************************************************************************************************/
/////////////////////////////////////////////////////// Helpers ///////////////////////////////////////////////////////

//
// Make the standard 1D launch configuration, [neurons].
//
std::tuple<dim3, dim3>
/*cuda::launch_configuration_t*/ MakeOnedimLaunchConfig(unsigned long int modelSize)
{
    const auto threadCount = 1024;
    const auto blockCount = ceil((float)modelSize/(float)threadCount);

#if false
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
#endif
    return std::make_tuple(dim3(blockCount, 1, 1), dim3(threadCount, 1, 1));
}


//
// Make the standard 2D launch configuration, [neurons, synapses].
//
std::tuple<dim3, dim3>
/*cuda::launch_configuration_t*/ MakeTwodimLaunchConfig(unsigned long int modelSize)
{
    const auto threadCount = 32;
    const auto blockNeuronCount = ceil((float)modelSize/(float)threadCount);
    const auto blockSynapseCount = ceil((float)SynapticConnectionsPerNode/(float)threadCount);
#if false
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
#endif
    return std::make_tuple(dim3(blockNeuronCount, blockSynapseCount, 1), dim3(threadCount, threadCount, 1));
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
    float postsynapticIncreaseFunction[],
    NeuronNode neurons[],
    NeuronPostSynapse postSynapses[][SynapticConnectionsPerNode],
    NeuronPreSynapse preSynapses[][SynapticConnectionsPerNode])
{
    g_modelSize = modelSize;
    g_synapticIncreaseFunction = postsynapticIncreaseFunction;
    g_pNeurons = neurons;
    g_pPostSynapses = postSynapses;
    g_pPreSynapses = preSynapses;

    for (auto i = 0; i < PostsynapticPlasticityPeriod; i++)
    {
        printf("Synapse increase function element %02d = %f\n", i, g_synapticIncreaseFunction[i]);
    }

    for (auto postSynapticNeuronId = 0; postSynapticNeuronId < modelSize; postSynapticNeuronId++)
    {
        for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
        {
            NeuronNode* presynapticNeuron = nullptr;
            unsigned long int presynapticIndex = (unsigned long int)postSynapses[postSynapticNeuronId][synapseId].PresynapticNeuron;
            if(presynapticIndex < modelSize)
            {
                // Normal, replace the indexes with a pointers.
                presynapticNeuron = &neurons[presynapticIndex];
            }
        
            postSynapses[postSynapticNeuronId][synapseId].PresynapticNeuron = presynapticNeuron;
        }
    }

    for (auto preSynapticNeuronId = 0; preSynapticNeuronId < modelSize; preSynapticNeuronId++)
    {
        for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
        {
            NeuronPostSynapse* postsynapse = nullptr;
            unsigned long int postsynapticNeuronIndex = (unsigned long int)preSynapses[preSynapticNeuronId][synapseId].Postsynapse;
            unsigned short int postsynapticSynapseIndex = preSynapses[preSynapticNeuronId][synapseId].PostSynapseIndex;
            if(postsynapticNeuronIndex < modelSize && postsynapticSynapseIndex < SynapticConnectionsPerNode)
            {
                // Normal, replace the indexes with a pointers.
                postsynapse = &postSynapses[postsynapticNeuronIndex][postsynapticSynapseIndex];
            }
        
            preSynapses[preSynapticNeuronId][synapseId].Postsynapse = postsynapse;
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
#if false
void
DeviceFixupShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    float postsynapticIncreaseFunction[],
    NeuronNode neurons[],
    NeuronPostSynapse postSynapses[][SynapticConnectionsPerNode],
    NeuronPreSynapse preSynapses[][SynapticConnectionsPerNode])
{
    const auto kernel_function = DeviceFixup;
	cuda::kernel_t kernel(device, kernel_function);

	kernel.set_cache_preference(cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory);
	kernel.set_shared_memory_bank_size(cuda::multiprocessor_shared_memory_bank_size_option_t::four_bytes_per_bank);

	auto attributes = kernel.attributes();

	const cuda::grid::dimensions_t grid_dims = {
		cuda::grid::dimension_t(1),
		cuda::grid::dimension_t(1),
		cuda::grid::dimension_t(1)
	};
    auto launch_configuration = cuda::make_launch_config(grid_dims, 1);
    
	cuda::launch(kernel_function, launch_configuration, modelSize, postsynapticIncreaseFunction, neurons, postSynapses, preSynapses);
	cuda::device::current::get().synchronize();
}
#else
void
DeviceFixupShim(
    unsigned long int modelSize,
    float postsynapticIncreaseFunction[],
    NeuronNode neurons[],
    NeuronPostSynapse postSynapses[][SynapticConnectionsPerNode],
    NeuronPreSynapse preSynapses[][SynapticConnectionsPerNode])
{

    cudaFuncSetCacheConfig(DeviceFixup, cudaFuncCachePreferL1);

    DeviceFixup <<< 1, 1 >>>(modelSize, postsynapticIncreaseFunction, neurons, postSynapses, preSynapses);
    cudaDeviceSynchronize();
}
#endif


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
#if false
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
#else
void
StreamInputShim(
    unsigned long int modelSize,
    unsigned long int inputSize,
    unsigned long long int inputNeurons[])
{
    auto [grid_dim, block_dim] = MakeOnedimLaunchConfig(inputSize); // Should this be modelSize?????
    StreamInput <<< grid_dim, block_dim >>>(inputSize, inputNeurons);
    cudaDeviceSynchronize();
}
#endif

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
// NeuronPostSynapse.TickSinceLastSignal
//
__global__ void ModelSynapses()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronId < g_modelSize)
    {
        auto& neuron = g_pNeurons[neuronId];

        //auto isSpikeTick = IsSpikeTick(neuron.TicksSinceLastSpike);
        auto isInRecovery = IsInRecovery(neuron.TicksSinceLastSpike);
        
        short int newActivation = 0;
        for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
        {
            auto& synapse = g_pPostSynapses[neuronId][synapseId];
            auto* presyapticNeuron = synapse.PresynapticNeuron;

            auto gate = presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime);
            auto eGate = (synapse.Type == SynapseType::Excitatory);
            auto iGate = (synapse.Type == SynapseType::Inhibitory);
            auto aGate = (synapse.Type == SynapseType::Attention);

            if (gate)
            {
                if (!isInRecovery)
                {
                    if (eGate) newActivation += synapse.Strength;
                    if (iGate) newActivation -= synapse.Strength;
                    if (aGate) neuron.Hypersensitive = HyperSensitivePeriod;
                }

                if (isInRecovery)
                {
                    if (eGate)
                    {
                        synapse.Strength = synapse.Strength * PostsynapticDecreaseFunction[RecoveryTimeMax - neuron.TicksSinceLastSpike];
                        //synapse.Flags |= AdjustTickFlagMask;
                    }
                }

                synapse.TickSinceLastSignal = PostsynapticPlasticityPeriod;
            }

            if (isInRecovery)
            {
                synapse.Flags &= ~(AdjustTickFlagMask | HypersensitiveFlagMask);
            }
        }

        neuron.Activation += newActivation;

        auto activationHigh = neuron.Activation > (ActivationThreshold + 1);
        auto activationLow = neuron.Activation <= -ActivationThreshold;
        if (!isInRecovery)
        {
            // In hypersensitive mode, the slightest new activation will trigger a spike, otherwise the total activation must be over threshold.
            if ( (neuron.Hypersensitive > 0 && newActivation > 0) || activationHigh)
            {
                //printf("ModelSynapses: Neuron %d above threshold (%d), setting NextTickSpike true, and clamping activation at %d\n", neuronId, neuron.Activation, ActivationThreshold + 1);
                neuron.NextTickSpike = true;
            }
        }

        if (activationHigh) neuron.Activation = ActivationThreshold + 1;
        if (activationLow)  neuron.Activation = -ActivationThreshold;
    }
}

__global__ void ModelSynapsesNobranch()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronId < g_modelSize)
    {
        auto& neuron = g_pNeurons[neuronId];
        //auto isSpikeTick = IsSpikeTick(neuron.TicksSinceLastSpike);
        auto isInRecovery = IsInRecovery(neuron.TicksSinceLastSpike);
        
        short int newActivation = 0;
        for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
        {
            auto& synapse = g_pPostSynapses[neuronId][synapseId];
            auto* presyapticNeuron = synapse.PresynapticNeuron;

            auto gate = presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime);
            auto eGate = (synapse.Type == SynapseType::Excitatory);
            auto iGate = (synapse.Type == SynapseType::Inhibitory);

            auto notInRecoveryGate = gate && !isInRecovery;
            auto activationChange = notInRecoveryGate * (eGate * synapse.Strength - iGate * synapse.Strength);
            newActivation += activationChange;

            auto InInRecoveeryGate = gate && isInRecovery;
            auto InRecoveryEGate = eGate && InInRecoveeryGate;
            auto multiplier = InRecoveryEGate * PostsynapticDecreaseFunction[RecoveryTimeMax - neuron.TicksSinceLastSpike] + (1 - InRecoveryEGate);
            auto strengthDecreased = multiplier < 1;
            synapse.Strength = multiplier * synapse.Strength;

            auto adjustTickOffMask = (isInRecovery * HypersensitiveFlagMask) | AdjustTickFlagMask;
            auto adjustTickOnMask = strengthDecreased * AdjustTickFlagMask;
            synapse.Flags = synapse.Flags & ~adjustTickOffMask | adjustTickOnMask;

            synapse.TickSinceLastSignal = gate * PostsynapticPlasticityPeriod + (1 - gate) * synapse.TickSinceLastSignal;
        }

        neuron.Activation += newActivation;

        auto activationHigh = neuron.Activation > (ActivationThreshold + 1);
        auto needSpikeNextTick = !isInRecovery && ((neuron.Hypersensitive > 0 && newActivation > 0) || activationHigh);
        neuron.NextTickSpike = needSpikeNextTick + (1 - needSpikeNextTick) * neuron.NextTickSpike;
        neuron.Activation = (activationHigh * (ActivationThreshold + 1)) + ((1 - activationHigh) * neuron.Activation);

        auto activationLow = neuron.Activation <= -ActivationThreshold;
        neuron.Activation = (activationLow * -ActivationThreshold) + ((1 - activationLow) * neuron.Activation);
    }
}


__global__ void ModelSynapsesOld()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuronId < g_modelSize)
    {
        auto& neuron = g_pNeurons[neuronId];
        
        // The recovery period is the first few ticks after a spike.  If this neuron
        // is past that period, it may integrate new presynaptic spikes.
        if (!IsInRecovery(neuron.TicksSinceLastSpike))
        {
            short int newActivation = 0;
            for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
            {
                auto& synapse = g_pPostSynapses[neuronId][synapseId];
                auto* presyapticNeuron = synapse.PresynapticNeuron;

                /// if (presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime))
                /// {
                ///     if (synapse.Type == SynapseType::Excitatory) newActivation += synapse.Strength;
                ///     if (synapse.Type == SynapseType::Inhibitory) newActivation -= synapse.Strength;
                ///    
                ///     synapse.TickSinceLastSignal = PostsynapticPlasticityPeriod;
                /// }
                auto gate = presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime);
                auto eGate = (synapse.Type == SynapseType::Excitatory);
                auto iGate = (synapse.Type == SynapseType::Inhibitory);
                auto activationChange = gate * (eGate * synapse.Strength - iGate * synapse.Strength);
                newActivation += activationChange;
                synapse.TickSinceLastSignal = gate * PostsynapticPlasticityPeriod + (1 - gate) * synapse.TickSinceLastSignal;

                synapse.Flags &= ~(AdjustTickFlagMask | HypersensitiveFlagMask);
            }

            neuron.Activation += newActivation;
            //printf("ModelSynapses: Neuron %d got signal from synapse %d with strength %d, changing activation from %d to %d\n", neuronId, synapseId, synapse.Strength, oldActivation, neuron.Activation);
        
            // In hypersensitive mode, the slightest new activation will trigger a spike, otherwise the total activation must be over threshold.
            if ( (neuron.Hypersensitive > 0 && newActivation > 0) || neuron.Activation > (ActivationThreshold + 1))
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
        // If this neuron is in the recovery period, incoming presynaptic spikes are ignored,
        // except to reduce the strength of those synapses.
        else
        {
            for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
            {
                auto& synapse = g_pPostSynapses[neuronId][synapseId];
                auto* presyapticNeuron = synapse.PresynapticNeuron;

                /// if (presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime))
                /// {
                ///     if (synapse.Type == SynapseType::Excitatory)
                ///     {
                ///         auto newStrength = synapse.Strength * PostsynapticDecreaseFunction[RecoveryTimeMax - neuron.TicksSinceLastSpike];
                ///         //printf("ModelPlasticity: Synapse %d was signaled after neuron %d spiked %d ticks ago, changing synaptic strength from %d to %d\n", synapseId, neuronId, RecoveryTimeMax - neuron.TicksSinceLastSpike, (int)synapse.Strength, (int)newStrength);
                ///         synapse.Strength = newStrength;
                ///     }
                ///    
                ///     synapse.TickSinceLastSignal = PostsynapticPlasticityPeriod;
                /// }
                auto gate = presyapticNeuron != nullptr && IsSpikeTick(presyapticNeuron->TicksSinceLastSpike + SignalDelayTime);
                auto eGate = gate * (synapse.Type == SynapseType::Excitatory);
                auto multiplier = eGate * PostsynapticDecreaseFunction[RecoveryTimeMax - neuron.TicksSinceLastSpike] + (1 - eGate);
                auto newStrength = multiplier * synapse.Strength;
                synapse.Strength = newStrength;
                synapse.TickSinceLastSignal = gate * PostsynapticPlasticityPeriod + (1 - gate) * synapse.TickSinceLastSignal;
                synapse.Flags |= eGate * AdjustTickFlagMask;
            }
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
#if false
void
ModelSynapsesShim(
    cuda::device_t& device,
    unsigned long int modelSize)
{
    const auto kernel_function = ModelSynapses;
	cuda::kernel_t kernel(device, kernel_function);

    auto launch_configuration = MakeOnedimLaunchConfig(modelSize);
    
	cuda::launch(kernel, launch_configuration);
	cuda::device::current::get().synchronize();
}
#else
void
ModelSynapsesShim(
    unsigned long int modelSize)
{
    auto [grid_dim, block_dim] = MakeOnedimLaunchConfig(modelSize);
    ModelSynapses <<< grid_dim, block_dim >>>();
    cudaDeviceSynchronize();
}
#endif

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
#if false
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
#else
void
ModelTimersShim(
    unsigned long int modelSize)
{
    auto [grid_dim, block_dim] = MakeOnedimLaunchConfig(modelSize);
    ModelTimers <<< grid_dim, block_dim >>>();
    cudaDeviceSynchronize();
}
#endif


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

    if (neuronId < g_modelSize)
    {
        auto& neuron = g_pNeurons[neuronId];
        auto neuronHypersensitive = neuron.Hypersensitive > 0;
        const unsigned char setFlagMask = neuronHypersensitive ? HypersensitiveFlagMask | AdjustTickFlagMask : AdjustTickFlagMask;

        if (neuron.NextTickSpike)
        {
            for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
            {
                auto& synapse = g_pPostSynapses[neuronId][synapseId];

                if (synapse.TickSinceLastSignal > 0)
                {
                    auto newStrength = synapse.Strength * g_synapticIncreaseFunction/*PostsynapticIncreaseFunction*/[PostsynapticPlasticityPeriod - synapse.TickSinceLastSignal];
                    if (newStrength > MaxSynapseStrength) newStrength = MaxSynapseStrength;
                    if (newStrength < MinSynapseStrength) newStrength = MinSynapseStrength;
                    //printf("ModelPlasticity: Neuron %d spiked, synapse %d was signaled %d ticks ago, changing synaptic strength from %d to %d\n", neuronId, synapseId, PostsynapticPlasticityPeriod - synapse.TickSinceLastSignal, (int)synapse.Strength, (int)newStrength);
                    synapse.Strength = newStrength;

                    //printf("Setting flags in neuron %d post-synapse %d with flag mask %d\n", neuronId, synapseId, setFlagMask);
                    synapse.Flags |= setFlagMask;
                }
            }
        }

        if (neuronHypersensitive)
        {
            neuron.Hypersensitive--;
        }
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
#if false
void
ModelPlasticityShim(
    cuda::device_t& device,
    unsigned long int modelSize)
{
	const auto kernel_function = ModelPlasticity;
	cuda::kernel_t kernel(device, kernel_function);

    auto launch_configuration = MakeOnedimLaunchConfig(modelSize);
    
	cuda::launch(kernel, launch_configuration);
	cuda::device::current::get().synchronize();
}
#else
void
ModelPlasticityShim(
    unsigned long int modelSize)
{
    auto [grid_dim, block_dim] = MakeOnedimLaunchConfig(modelSize);
    ModelPlasticity <<< grid_dim, block_dim >>>();
    cudaDeviceSynchronize();
}
#endif


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
// NeuronPostSynapse.TickSinceLastSignal
//
__global__ void ModelTick()
{
    auto neuronId = blockIdx.x * blockDim.x + threadIdx.x;
    auto synapseId = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuronId < g_modelSize && synapseId < SynapticConnectionsPerNode)
    {
        auto& neuron = g_pNeurons[neuronId];

        if (synapseId == 0)
        {
            if (neuron.TicksSinceLastSpike > 0)
            {
                //auto oldTicks = neuron.TicksSinceLastSpike;
                neuron.TicksSinceLastSpike--;
                //printf("ModelTick:   Neuron %d ticking TicksSinceLastSpike for from %d to %d\n", neuronId, oldTicks, neuron.TicksSinceLastSpike);
            }

            neuron.NextTickSpike = false;
        }
        
        auto& postsynapse = g_pPostSynapses[neuronId][synapseId];
        auto& presynapse = g_pPreSynapses[neuronId][synapseId];
        if (postsynapse.TickSinceLastSignal > 0)
        {
            //auto oldTicks = synapse.TickSinceLastSignal;
            postsynapse.TickSinceLastSignal--;
            //printf("ModelTick:   Neuron %d synapse %d ticking TickSinceLastSignal for from %d to %d\n", neuronId, synapseId, oldTicks, synapse.TickSinceLastSignal);
        }

        const auto flagsMask = AdjustTickFlagMask | HypersensitiveFlagMask;
        unsigned char postSynapseFlagsMasked = 0;
        if (presynapse.Postsynapse != nullptr)
        {
            postSynapseFlagsMasked = presynapse.Postsynapse->Flags & flagsMask;
        }
        if (postSynapseFlagsMasked == flagsMask)
        {
            //printf("Neuron %d being set hypersensitive through presynapse %d (%d==%d) for the next %d ticks\n", neuronId, synapseId, (unsigned int)presynapse.Postsynapse->Flags, (unsigned int)(AdjustTickFlagMask | HypersensitiveFlagMask), HyperSensitivePeriod);
            neuron.Hypersensitive = HyperSensitivePeriod;
        }
        
    }
}

//
//  Called from CPU.  Launch the CUDA kernel.
//
#if false
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
#else
void
ModelTickShim(
    unsigned long int modelSize)
{
    auto [grid_dim, block_dim] = MakeTwodimLaunchConfig(modelSize);
    ModelTick <<< grid_dim, block_dim >>>();
    cudaDeviceSynchronize();
}
#endif


