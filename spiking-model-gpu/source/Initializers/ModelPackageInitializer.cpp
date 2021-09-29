#include "Initializers/IModelInitializer.h"
#include "Initializers/ModelPackageInitializer.h"

#include "IModelHelper.h"
#include "NeuronRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::IModelHelper;
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelPackageInitializer;

    // the class factories

    extern "C" IModelInitializer* create(IModelHelper* helper) {
        return new ModelPackageInitializer(helper);
    }

    extern "C" void destroy(IModelInitializer* p) {
        delete p;
    }
}
