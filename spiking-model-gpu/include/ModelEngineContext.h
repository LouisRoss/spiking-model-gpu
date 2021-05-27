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
#include "Recorder.h"

#include "GpuModelHelper.h"

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
        atomic<bool> Pause { false };
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
        atomic<bool> EngineInitializeFailed { false };
        microseconds PartitionTime { };
        unsigned long long int Iterations { 1LL };
        long long int TotalWork { 0LL };

        ModelEngineContext(const ConfigurationRepository& configuration, GpuModelHelper<RECORDTYPE>& helper) :
            Configuration(configuration),
            Helper(helper),
            Record(Iterations),
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
            if (!Configuration.Configuration().contains("Model"))
                return false;

            const json& modelJson = Configuration.Configuration()["Model"];
            if (!modelJson.is_object())
                return false;

            if (modelJson.contains("ModelTicks"))
            {
                const json& modelTicksJson = modelJson["ModelTicks"];
                if (modelTicksJson.is_number_integer() || modelTicksJson.is_number_unsigned())
                    EnginePeriod = microseconds(modelTicksJson.get<int>());
            }

            RecordFile = Configuration.ComposeRecordPath();
            LogFile = Configuration.ExtractRecordDirectory() + LogFile;

            return true;
        }

        //
        // Render all the context properties as a single JSON object.
        //
        json Render()
        {
            return json {
                {"run", Run ? true : false},
                {"loglevel", LoggingLevel},
                {"logfile", LogFile.c_str()},
                {"recordfile", RecordFile.c_str()},
                {"engineperiod", EnginePeriod.count()},
                {"engineinit", EngineInitialized ? true : false},
                {"enginefail", EngineInitializeFailed ? true : false},
                {"iterations", Iterations},
                {"totalwork", TotalWork}
            };
        }

        //
        // Render all the dynamic context properties as a single JSON object.
        //
        json RenderDynamic()
        {
            return json {
                {"run", Run ? true : false},
                {"engineinit", EngineInitialized ? true : false},
                {"enginefail", EngineInitializeFailed ? true : false},
                {"iterations", Iterations},
                {"totalwork", TotalWork}
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
