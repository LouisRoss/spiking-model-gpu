#pragma once

//#include "ModelUi.h"
#include "CommandControlConsoleUi.h"

#include "ModelRunner.h"
#include "GpuModelHelper.h"
#include "NeuronRecord.h"
//#include "ICommandControlAcceptor.h"

namespace embeddedpenguins::gpu::neuron::model
{
    //using embeddedpenguins::core::neuron::model::ModelUi;
    using embeddedpenguins::core::neuron::model::CommandControlConsoleUi;
    //using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;

    //class GpuModelUi : public ModelUi<ModelRunner<NeuronRecord>, GpuModelHelper<NeuronRecord>>
    class GpuModelUi : public CommandControlConsoleUi<ModelRunner<NeuronRecord>, GpuModelHelper<NeuronRecord>>
    {
        unsigned long int modelSize_ {};
        string legend_ {};

    public:
        //GpuModelUi(ModelRunner<NeuronRecord>& modelRunner, unique_ptr<ICommandControlAcceptor> commandControl) :
        GpuModelUi(ModelRunner<NeuronRecord>& modelRunner) :
            //ModelUi(modelRunner, std::move(commandControl))
            CommandControlConsoleUi(modelRunner)
        {
            modelSize_ = helper_.Carrier().ModelSize();
        }

        virtual char EmitToken(unsigned long neuronIndex) override
        {
            if (neuronIndex >= modelSize_) return '=';
            
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
