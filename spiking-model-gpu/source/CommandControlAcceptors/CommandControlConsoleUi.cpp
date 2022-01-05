#include "IModelRunner.h"
#include "CommandControlAcceptors/GpuModelUi.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
    using embeddedpenguins::core::neuron::model::IModelRunner;

    // the class factories

    extern "C" ICommandControlAcceptor* create(IModelRunner& modelRunner) {
        return new GpuModelUi(modelRunner);
    }

    extern "C" void destroy(ICommandControlAcceptor* p) {
        delete p;
    }
}
