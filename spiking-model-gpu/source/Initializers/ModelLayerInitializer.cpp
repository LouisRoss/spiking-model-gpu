#include "ConfigurationRepository.h"
#include "Initializers/ModelLayerInitializer.h"

#include "IModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ModelContext;
    using embeddedpenguins::core::neuron::model::IModelHelper;
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelLayerInitializer;

    // the class factories

    extern "C" IModelInitializer* create(IModelHelper* helper, ModelContext* context) {
        return new ModelLayerInitializer(helper, context);
    }

    extern "C" void destroy(IModelInitializer* p) {
        delete p;
    }
}
