#include "ConfigurationRepository.h"
#include "SensorInputs/SensorInputFile.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::ISensorInput;
    using embeddedpenguins::core::neuron::model::SensorInputFile;

    // the class factories

    extern "C" ISensorInput* create(ConfigurationRepository& configuration) {
        return new SensorInputFile(configuration);
    }

    extern "C" void destroy(ISensorInput* p) {
        delete p;
    }
}
