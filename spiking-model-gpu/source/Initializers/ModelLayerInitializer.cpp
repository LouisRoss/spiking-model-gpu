#include "ConfigurationRepository.h"
#include "Initializers/ModelLayerInitializer.h"

#include "GpuModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelLayerInitializer;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;

    // the class factories

    extern "C" IModelInitializer<GpuModelHelper>* create(GpuModelHelper& helper) {
        return new ModelLayerInitializer<GpuModelHelper>(helper);
    }

    extern "C" void destroy(IModelInitializer<GpuModelHelper>* p) {
        delete p;
    }
}
