#include "IModelRunner.h"
#include "CommandControlAcceptors/QueryResponseListenSocket.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
    using embeddedpenguins::core::neuron::model::QueryResponseListenSocket;
    using embeddedpenguins::core::neuron::model::IModelRunner;

    // the class factories

    extern "C" ICommandControlAcceptor* create(IModelRunner& modelRunner) {
        return new QueryResponseListenSocket(modelRunner);
    }

    extern "C" void destroy(ICommandControlAcceptor* p) {
        delete p;
    }
}
