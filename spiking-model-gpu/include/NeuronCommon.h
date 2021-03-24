#pragma once

#include <chrono>

namespace embeddedpenguins::gpu::neuron::model
{
    using std::chrono::milliseconds;

//#define STREAM_CPU
//#define SYNAPSE_RECORD
#define BRANCHLESS_CODE


    constexpr unsigned int SynapticConnectionsPerNode = 1500;
    constexpr unsigned int InputBufferSize = 10'000;
    constexpr unsigned int BlockSizeSynapse = 256;
    constexpr int MaxSynapseStrength = 110;
    constexpr int MinSynapseStrength = -110;
    constexpr int ActivationThreshold = 100;
    constexpr int PostsynapticPlasticityPeriod = 30;
    constexpr int SignalDelayTime = 7;
    constexpr int RefractoryTime = 7;
    constexpr double DecayRate = 0.80;
    constexpr int DecayProcessRate = 1;

#define SynapseSignalTimeMax 8
#define RecoveryTimeMax 50
#define SpikeDuration 6
#define RecoverDuration 8
//#define RampdownDuration 60

#define TimeSinceRecovery(_RecoveryTime) ((_RecoveryTime>0)? RecoveryTimeMax-_RecoveryTime: 255)
#define IsSpikeTick(_RecoveryTime) (_RecoveryTime == RecoveryTimeMax)
#define IsRefractoryTick(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) == SpikeDuration)
#define IsInSpikeTime(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) < SpikeDuration)
//#define IsInRampdownTime(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) < RampdownDuration)
#define IsInRecovery(_RecoveryTime) (TimeSinceRecovery(_RecoveryTime) < RecoverDuration)
#define IsActiveRecently(_RecoveryTime) ((_RecoveryTime) != 0 && !IsInRecovery(_RecoveryTime))


    enum class Operation
    {
        Spike,          // The NeuronNode indexed by a NeuronOperation.Index received a spike from a synapse.
        Decay,          // The NeuronNode indexed by a NeuronOperation.Index is in the decay phase.
        Refractory      // The NeuronNode indexed by a NeuronOperation.Index is in refractory period after spiking.
    };
}
