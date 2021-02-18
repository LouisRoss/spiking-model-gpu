#pragma once

#include "core/ModelUi.h"

#include "ModelRunner.h"
#include "GpuModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ModelUi;

    class GpuModelUi : public ModelUi<ModelRunner<NeuronRecord>, GpuModelHelper<NeuronRecord>>
    {
        string legend_ {};

    public:
        GpuModelUi(ModelRunner<NeuronRecord>& modelRunner, GpuModelHelper<NeuronRecord>& helper) :
            ModelUi(modelRunner, helper)
        {
            
        }

        virtual char EmitToken(unsigned long neuronIndex) override
        {
            auto activation = helper_.GetNeuronActivation(neuronIndex);
            return MapIntensity(activation);
        }

        virtual const string& Legend() override
        {
            return legend_;
        }

    private:
        char MapIntensity(int activation)
        {
            static int cutoffs[] = {2,5,15,50};

            if (activation < cutoffs[0]) return ' ';
            if (activation < cutoffs[1]) return '.';
            if (activation < cutoffs[2]) return '*';
            if (activation < cutoffs[3]) return 'o';
            return 'O';
        }
    };
}
