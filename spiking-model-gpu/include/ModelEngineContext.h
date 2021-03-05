#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <vector>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include "nlohmann/json.hpp"

#include "ConfigurationRepository.h"
#include "Log.h"
#include "Recorder.h"

#include "GpuModelHelper.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::atomic;
    using std::mutex;
    using std::condition_variable;
    using std::vector;
    using std::unique_ptr;
    using std::chrono::microseconds;

    using nlohmann::json;
    
    using embeddedpenguins::core::neuron::model::Log;
    using embeddedpenguins::core::neuron::model::LogLevel;
    using embeddedpenguins::core::neuron::model::Recorder;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;

    //
    // Carry the public information defining the model engine.
    // This includes synchronization between the model engine and its thread;
    // configuration and logging; all workers; and statistics about the run.
    //
    template<class RECORDTYPE>
    struct ModelEngineContext
    {
        atomic<bool> Run { false };
        bool Quit { false };
        mutex Mutex;
        condition_variable Cv;

        const ConfigurationRepository& Configuration;
        GpuModelHelper<RECORDTYPE>& Helper;
        Log Logger {};
        LogLevel LoggingLevel { LogLevel::Status };
        Recorder<RECORDTYPE> Record;
        string LogFile {"ModelEngine.log"};
        string RecordFile {"ModelEngineRecord.csv"};

        microseconds EnginePeriod;
        atomic<bool> EngineInitialized { false };
        microseconds PartitionTime { };
        unsigned long long int Iterations { 1LL };
        long long int TotalWork { 0LL };

        ModelEngineContext(const ConfigurationRepository& configuration, GpuModelHelper<RECORDTYPE>& helper) :
            Configuration(configuration),
            Helper(helper),
            Record(Iterations),
            EnginePeriod(1000)
        {
            // Create and run the model engine.
            const json& modelJson = Configuration.Configuration()["Model"];
            if (modelJson.is_null()) return;

            const json& modelTicksJson = modelJson["ModelTicks"];
            if (modelTicksJson.is_number_integer() || modelTicksJson.is_number_unsigned())
                EnginePeriod = microseconds(modelTicksJson.get<int>());

            RecordFile = Configuration.ComposeRecordPath();
            LogFile = Configuration.ExtractRecordDirectory() + LogFile;
        }
    };
}
