#pragma once

#include "NeuronCommon.h"

namespace embeddedpenguins::neuron::infrastructure
{
    using std::numeric_limits;

    enum class NeuronType : char
    {
        Excitatory,
        Inhibitory
    };

    struct __align__(4) NeuronNode
    {
        NeuronType Type;
        short int Activation;
        unsigned short int TicksSinceLastSpike;
    };
}
