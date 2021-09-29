#include "ConfigurationRepository.h"
#include "Initializers/ModelSonataInitializer.h"

#include "IModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::IModelHelper;
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelSonataInitializer;

    // the class factories

    extern "C" IModelInitializer* create(IModelHelper* helper) {
        return new ModelSonataInitializer(helper);
    }

    extern "C" void destroy(IModelInitializer* p) {
        delete p;
    }
}
