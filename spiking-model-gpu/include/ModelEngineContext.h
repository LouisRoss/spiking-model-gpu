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
    using std::chrono::_V2::system_clock;
    using std::chrono::high_resolution_clock;

    using nlohmann::json;
    
    using embeddedpenguins::core::neuron::model::Log;
    using embeddedpenguins::core::neuron::model::LogLevel;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::Performance;

    //
    // During a run, capture the measurements and statistics about that
    // run here.  This struct can then be copied out to a permanent location
    // before its containing context is deleted.
    //
    struct RunMeasurements
    {
        microseconds PartitionTime { };
        system_clock::time_point EngineStartTime { };
        system_clock::time_point EngineStopTime { };
        unsigned long long int Iterations { 1LL };
        long long int TotalWork { 0LL };
        Performance PerformanceCounters { };
    };

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
        atomic<bool> RecordActivationEnable { false };
        atomic<bool> RecordHyperSensitiveEnable { false };
        mutex Mutex;
        condition_variable Cv;

        ConfigurationRepository& Configuration;
        Log Logger {};
        LogLevel LoggingLevel { LogLevel::Status };
        string LogFile {"ModelEngine.log"};

        microseconds EnginePeriod;
        atomic<bool> EngineInitialized { false };
        atomic<bool> EngineInitializeFailed { false };
        RunMeasurements& Measurements;

        ModelEngineContext(ConfigurationRepository& configuration, RunMeasurements& runMeasurements) :
            Configuration(configuration),
            EnginePeriod(1000),
            Measurements(runMeasurements)
        {
        }

        //
        // Initialize configured context properties after creation.
        // NOTE: assumes the configuration has been loaded first.
        //
        bool Initialize()
        {
            LogFile = Configuration.ComposeRecordPathForModel(Configuration.ExtractRecordDirectory(), LogFile);
            cout << "Context initialized with ticks = " << EnginePeriod.count() << " us\n";

            return true;
        }

        //
        //  Capture now as the start time.
        //
        void TriggerStartTime()
        {
            Measurements.EngineStartTime = high_resolution_clock::now();
            Measurements.Iterations = 0LL;
        }

        //
        //  Capture now as the stop time.
        //
        void TriggerStopTime()
        {
            Measurements.EngineStopTime = high_resolution_clock::now();
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
                {"iterations", Measurements.Iterations},
                {"totalwork", Measurements.TotalWork},
                {"cpu", Measurements.PerformanceCounters.GetActiveTotalCpu()}
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
                {"iterations", Measurements.Iterations},
                {"totalwork", Measurements.TotalWork},
                {"cpu", Measurements.PerformanceCounters.GetActiveTotalCpu()}
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
                if (controlValues.contains("recordactivation"))
                {
                    RecordActivationEnable = controlValues["recordactivation"].get<bool>();
                    cout << "Changed record activation enable to " << RecordActivationEnable << "\n";
                }
                if (controlValues.contains("recordhypersensitive"))
                {
                    RecordHyperSensitiveEnable = controlValues["recordhypersensitive"].get<bool>();
                    cout << "Changed record hypersensitive enable to " << RecordHyperSensitiveEnable << "\n";
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
                if (controlValues.contains("startmeasurement"))
                {
                    TriggerStartTime();
                    cout << "Restarted start time and iterations\n";
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
