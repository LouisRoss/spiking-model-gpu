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

    extern "C" IModelInitializer<GpuModelHelper>* create(GpuModelHelper& helper) {
        return new ModelSonataInitializer<GpuModelHelper>(helper);
    }

    extern "C" void destroy(IModelInitializer<GpuModelHelper>* p) {
        delete p;
    }
}
