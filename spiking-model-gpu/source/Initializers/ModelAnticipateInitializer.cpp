#include "ConfigurationRepository.h"
#include "Initializers/ModelAnticipateInitializer.h"

#include "GpuModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelAnticipateInitializer;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;

    // the class factories

    extern "C" IModelInitializer<GpuModelHelper>* create(GpuModelHelper& helper) {
        return new ModelAnticipateInitializer<GpuModelHelper>(helper);
    }

    extern "C" void destroy(IModelInitializer<GpuModelHelper>* p) {
        delete p;
    }
}
