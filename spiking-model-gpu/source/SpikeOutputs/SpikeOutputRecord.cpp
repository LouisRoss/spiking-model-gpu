#include "NeuronRecord.h"
#include "ModelEngineContext.h"
#include "SpikeOutputs/SpikeOutputRecord.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::Recorder;
    using embeddedpenguins::core::neuron::model::ISpikeOutput;
    using embeddedpenguins::core::neuron::model::SpikeOutputRecord;


    // the class factories

    extern "C" ISpikeOutput* create(ModelEngineContext& context) {
        return new SpikeOutputRecord<NeuronRecord>(context);
    }

    extern "C" void destroy(ISpikeOutput* p) {
        delete p;
    }
}
