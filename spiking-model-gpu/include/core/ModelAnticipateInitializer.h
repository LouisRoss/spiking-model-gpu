#pragma once

#include "nlohmann/json.hpp"

#include "ModelNeuronInitializer.h"
#include "ConfigurationRepository.h"

namespace embeddedpenguins::core::neuron::model
{
    using nlohmann::json;

    //
    // This custom initializer sets up a spiking neuron model for 
    // the 'anticipate' test, which demonstrates STDP over repeated
    // spikes.
    //
    template<class MODELHELPERTYPE>
    class ModelAnticipateInitializer : public ModelNeuronInitializer<MODELHELPERTYPE>
    {
    public:
        ModelAnticipateInitializer(MODELHELPERTYPE helper) :
            ModelNeuronInitializer<MODELHELPERTYPE>(helper)
        {
        }

    public:
        // IModelInitializer implementaton
        virtual void Initialize() override
        {
            if (!this->helper_.InitializeModel()) 
                return;

            const Neuron2Dim I1 { this->ResolveNeuron("I1") };
            const Neuron2Dim I2 { this->ResolveNeuron("I2") };
            const Neuron2Dim Inh1  { this->ResolveNeuron("Inh1") };
            const Neuron2Dim Inh2  { this->ResolveNeuron("Inh2") };
            const Neuron2Dim N1  { this->ResolveNeuron("N1") };
            const Neuron2Dim N2  { this->ResolveNeuron("N2") };

            this->strength_ = 102;
            this->InitializeAConnection(I1, N1);
            this->strength_ = 51;
            this->InitializeAConnection(N2, N1);

            this->strength_ = 102;
            this->InitializeAConnection(I2, N2);
            this->strength_ = 51;
            this->InitializeAConnection(N1, N2);

            this->strength_ = 102;
            this->InitializeAnInput(I1);
            this->InitializeAnInput(I2);

            this->strength_ = 102;
            this->SetInhibitoryNeuronType(Inh1);
            this->InitializeAConnection(N1, Inh1);
            this->InitializeAConnection(Inh1, I1);

            this->strength_ = 102;
            this->SetInhibitoryNeuronType(Inh2);
            this->InitializeAConnection(N2, Inh2);
            this->InitializeAConnection(Inh2, I2);
        }
    };
}
