#pragma once

#include "NeuronPostSynapse.h"

namespace embeddedpenguins::gpu::neuron::model
{
    struct __attribute__ ((aligned(8))) NeuronPreSynapse
    {
        NeuronPostSynapse* Postsynapse;
        unsigned short PostSynapseIndex;
    };
}
