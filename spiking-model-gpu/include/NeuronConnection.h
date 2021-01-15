#pragma once

#include "NeuronNode.h"

namespace embeddedpenguins::neuron::infrastructure
{
    using std::numeric_limits;

    struct NeuronConnection
    {
        NeuronNode* PostsynapticNeuron { nullptr };
        unsigned short int Synapse { 0 };
    };
}
