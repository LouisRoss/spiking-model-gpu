#include "ModelContext.h"
#include "SpikeOutputs/SpikeOutputSocket.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ModelContext;
    using embeddedpenguins::core::neuron::model::ISpikeOutput;
    using embeddedpenguins::core::neuron::model::SpikeOutputSocket;

    // the class factories

    extern "C" ISpikeOutput* create(ModelContext& context) {
        return new SpikeOutputSocket(context);
    }

    extern "C" void destroy(ISpikeOutput* p) {
        delete p;
    }
}
