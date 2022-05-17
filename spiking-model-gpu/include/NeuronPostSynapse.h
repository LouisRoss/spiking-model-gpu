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

    struct __align__(8) NeuronPostSynapse
    {
        enum class SynapseFlags : unsigned char
        {
            AdjustTick = 0x1,
            Spare1     = 0x2,
            Spare2     = 0x4,
            Spare3     = 0x8,
            Spare4     = 0x10,
            Spare5     = 0x20,
            Spare6     = 0x40,
            Spare7     = 0x80

        };

        NeuronNode* PresynapticNeuron;
        char Strength;
        unsigned char TickSinceLastSignal;
        SynapseType Type;
        unsigned char Flags;
    };
}
