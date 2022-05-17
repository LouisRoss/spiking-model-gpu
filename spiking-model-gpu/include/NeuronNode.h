#pragma once

#include "NeuronCommon.h"
#include "NeuronRecordCommon.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::NeuronType;

    struct NeuronNode
    {
        NeuronType Type;
        bool NextTickSpike;
        short int Hypersensitive;
        short int Activation;
        unsigned short int TicksSinceLastSpike;
    };
}
