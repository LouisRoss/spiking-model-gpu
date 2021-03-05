#include "ConfigurationRepository.h"
#include "Initializers/ModelSonataInitializer.h"

#include "GpuModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelSonataInitializer;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;

    // the class factories

    extern "C" IModelInitializer<GpuModelHelper<NeuronRecord>>* create(GpuModelHelper<NeuronRecord>& helper) {
        return new ModelSonataInitializer<GpuModelHelper<NeuronRecord>>(helper);
    }

    extern "C" void destroy(IModelInitializer<GpuModelHelper<NeuronRecord>>* p) {
        delete p;
    }
}
