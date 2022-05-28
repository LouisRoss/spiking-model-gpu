#pragma once

#include "NeuronPostSynapse.h"

namespace embeddedpenguins::gpu::neuron::model
{
    struct __align__(8) NeuronPreSynapse
    {
        NeuronPostSynapse* Postsynapse;
        unsigned short PostSynapseIndex;
    };
}
