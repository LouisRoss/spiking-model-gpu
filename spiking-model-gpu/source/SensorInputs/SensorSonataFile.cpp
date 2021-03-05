#include "ConfigurationRepository.h"
#include "SensorInputs/SensorSonataFile.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::ISensorInput;
    using embeddedpenguins::core::neuron::model::SensorSonataFile;

    // the class factories

    extern "C" ISensorInput* create(const ConfigurationRepository& configuration) {
        return new SensorSonataFile(configuration);
    }

    extern "C" void destroy(ISensorInput* p) {
        delete p;
    }
}
