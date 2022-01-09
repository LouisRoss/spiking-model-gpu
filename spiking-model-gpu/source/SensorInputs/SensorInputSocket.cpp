#include "ConfigurationRepository.h"
#include "SensorInputs/SensorInputSocket.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::ISensorInput;
    using embeddedpenguins::core::neuron::model::SensorInputSocket;

    // the class factories

    extern "C" ISensorInput* create(ConfigurationRepository& configuration) {
        return new SensorInputSocket(configuration);
    }

    extern "C" void destroy(ISensorInput* p) {
        delete p;
    }
}
