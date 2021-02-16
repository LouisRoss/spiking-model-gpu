#pragma once

#include "NeuronCommon.h"

namespace embeddedpenguins::gpu::neuron::model
{
    enum class NeuronType : char
    {
        Excitatory,
        Inhibitory
    };

    struct __align__(4) NeuronNode
    {
        NeuronType Type;
        bool NextTickSpike;
        short int Activation;
        unsigned short int TicksSinceLastSpike;
    };
}
