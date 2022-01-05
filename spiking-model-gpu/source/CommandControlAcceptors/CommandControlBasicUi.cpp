#include "IModelRunner.h"
#include "CommandControlAcceptors/CommandControlBasicUi.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
    using embeddedpenguins::core::neuron::model::CommandControlBasicUi;
    using embeddedpenguins::core::neuron::model::IModelRunner;

    // the class factories

    extern "C" ICommandControlAcceptor* create(IModelRunner& modelRunner) {
        return new CommandControlBasicUi(modelRunner);
    }

    extern "C" void destroy(ICommandControlAcceptor* p) {
        delete p;
    }
}
