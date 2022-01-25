#include "Log.h"
#include "ConfigurationRepository.h"
#include "SensorInputs/SensorInputFile.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using embeddedpenguins::core::neuron::model::LogLevel;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::ISensorInput;
    using embeddedpenguins::core::neuron::model::SensorInputFile;

    // the class factories

    extern "C" ISensorInput* create(ConfigurationRepository& configuration, unsigned long long int& iterations, LogLevel& loggingLevel) {
        return new SensorInputFile(configuration, iterations, loggingLevel);
    }

    extern "C" void destroy(ISensorInput* p) {
        delete p;
    }
}
