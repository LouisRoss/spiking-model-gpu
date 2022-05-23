#pragma once

namespace embeddedpenguins::gpu::neuron::model
{
    struct NeuronNode
    {
        bool NextTickSpike;
        short int Hypersensitive;
        short int Activation;
        unsigned short int TicksSinceLastSpike;
    };
}
