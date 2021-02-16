#pragma once

#include "NeuronNode.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::numeric_limits;

    enum class SynapseType : char
    {
        Excitatory,
        Inhibitory,
        Attention
    };

    struct __align__(8) NeuronSynapse
    {
        NeuronNode* PresynapticNeuron;
        char Strength;
        unsigned char TickSinceLastSignal;
        SynapseType Type;
    };
}
