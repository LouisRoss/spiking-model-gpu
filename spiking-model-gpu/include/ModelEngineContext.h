#pragma once

#include <iostream>
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
#include "Performance.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::cout;
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
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::Performance;

    //
    // Carry the public information defining the model engine.
    // This includes synchronization between the model engine and its thread;
    // configuration and logging; all workers; and statistics about the run.
    //
    struct ModelEngineContext
    {
        atomic<bool> Run { false };
        atomic<bool> Pause { false };
        atomic<bool> RecordEnable { false };
        atomic<bool> RecordSynapseEnable { false };
        mutex Mutex;
        condition_variable Cv;

        ConfigurationRepository& Configuration;
        Log Logger {};
        LogLevel LoggingLevel { LogLevel::Status };
        string LogFile {"ModelEngine.log"};

        microseconds EnginePeriod;
        atomic<bool> EngineInitialized { false };
        atomic<bool> EngineInitializeFailed { false };
        microseconds PartitionTime { };
        unsigned long long int Iterations { 1LL };
        long long int TotalWork { 0LL };
        Performance PerformanceCounters { };

        ModelEngineContext(ConfigurationRepository& configuration) :
            Configuration(configuration),
            EnginePeriod(1000)
        {
        }

        //
        // Initialize configured context properties after creation.
        // NOTE: assumes the configuration has been loaded first.
        //
        bool Initialize()
        {
            // Create and run the model engine.
            if (Configuration.Configuration().contains("Model"))
            {
                const json& modelJson = Configuration.Configuration()["Model"];
                if (modelJson.is_object() && modelJson.contains("ModelTicks"))
                {
                    const json& modelTicksJson = modelJson["ModelTicks"];
                    if (modelTicksJson.is_number_integer() || modelTicksJson.is_number_unsigned())
                        EnginePeriod = microseconds(modelTicksJson.get<int>());
                }
            }

            LogFile = Configuration.ExtractRecordDirectory() + LogFile;
            cout << "Context initialized with ticks = " << EnginePeriod.count() << " us\n";

            return true;
        }

        //
        // Render all the context properties as a single JSON object.
        //
        json Render()
        {
            return json {
                {"run", Run ? true : false},
                {"pause", Pause ? true : false},
                {"loglevel", LoggingLevel},
                {"logfile", LogFile.c_str()},
                {"recordfile", Configuration.ComposeRecordPath().c_str()},
                {"recordenable", RecordEnable ? true : false},
                {"recordsynapses", RecordSynapseEnable ? true : false},
                {"engineperiod", EnginePeriod.count()},
                {"engineinit", EngineInitialized ? true : false},
                {"enginefail", EngineInitializeFailed ? true : false},
                {"iterations", Iterations},
                {"totalwork", TotalWork},
                {"cpu", PerformanceCounters.GetActiveTotalCpu()}
            };
        }

        //
        // Render all the dynamic context properties as a single JSON object.
        //
        json RenderDynamic()
        {
            return json {
                {"run", Run ? true : false},
                {"pause", Pause ? true : false},
                {"engineinit", EngineInitialized ? true : false},
                {"enginefail", EngineInitializeFailed ? true : false},
                {"iterations", Iterations},
                {"totalwork", TotalWork},
                {"cpu", PerformanceCounters.GetActiveTotalCpu()}
            };
        }

        //
        // Set one or more context properties from the subset passed as a JSON object.
        //
        bool SetValue(const json& controlValues)
        {
            bool success {true};
            try
            {
                if (controlValues.contains("recordenable"))
                {
                    RecordEnable = controlValues["recordenable"].get<bool>();
                    cout << "Changed record enable to " << RecordEnable << "\n";
                }
                if (controlValues.contains("recordsynapses"))
                {
                    RecordSynapseEnable = controlValues["recordsynapses"].get<bool>();
                    cout << "Changed record synapse enable to " << RecordSynapseEnable << "\n";
                }
                if (controlValues.contains("loglevel"))
                {
                    LoggingLevel = (LogLevel)controlValues["loglevel"].get<int>();
                    cout << "Changed logging level to " << (int)LoggingLevel << "\n";
                }
                if (controlValues.contains("engineperiod"))
                {
                    EnginePeriod = microseconds(controlValues["engineperiod"].get<int>());
                    cout << "Changed engine period to " << EnginePeriod.count() << "\n";
                }
                // Do more as they come up...
            }
            catch(const std::exception& e)
            {
                success = false;
            }
            
            return success;
        }
    };
}
