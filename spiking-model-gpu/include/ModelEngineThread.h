#pragma once

#include <iostream>
#include <string>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <vector>
#include <limits>

#include "nlohmann/json.hpp"

#include "CudaApi.h"

#include "core/ConfigurationRepository.h"
#include "core/Log.h"
#include "core/Recorder.h"
#include "core/ModelInitializerProxy.h"
#include "core/SensorInputProxy.h"

#include "ModelEngineContext.h"
#include "GpuModelHelper.h"
#include "GpuModelCarrier.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::vector;
    using std::unique_ptr;
    using std::make_unique;
    using std::unique_lock;
    using std::chrono::high_resolution_clock;
    using time_point = std::chrono::high_resolution_clock::time_point;
    using std::chrono::milliseconds;
    using std::chrono::microseconds;
    using std::chrono::duration_cast;
    using std::numeric_limits;
    using std::cout;
    using std::cerr;

    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::Log;
    using embeddedpenguins::core::neuron::model::Recorder;
    using embeddedpenguins::core::neuron::model::ModelInitializerProxy;
    using embeddedpenguins::core::neuron::model::ISensorInput;
    using embeddedpenguins::core::neuron::model::SensorInputProxy;

    //
    // The model engine does its work in this thread object.
    // The main loop of this thread object runs a two-phase process where
    // 1) this thread partitions existing work to the worker threads, and
    // 2) the worker threads execute the work.
    // The two phases do not run simultaneously, but use synchronization barriers
    // to ensure the phases run serially.
    //
    template<class RECORDTYPE>
    class ModelEngineThread
    {
        ModelEngineContext<RECORDTYPE>& context_;
        GpuModelCarrier& carrier_;
        GpuModelHelper<RECORDTYPE> helper_;

        time_point nextScheduledTick_;
        unique_ptr<ISensorInput> sensorInput_ { };

    public:
        ModelEngineThread() = delete;
        ModelEngineThread(
                        ModelEngineContext<RECORDTYPE>& context, 
                        GpuModelCarrier& carrier) :
            context_(context),
            carrier_(carrier),
            helper_(carrier_, context_.Configuration),
            nextScheduledTick_(high_resolution_clock::now() + context_.EnginePeriod)
        {
            context_.Logger.SetId(0);
        }

        void operator() ()
        {
            while (!context_.Run) { std::this_thread::yield(); }

            try
            {
                Initialize();
                MainLoop();
                Cleanup();
            }
            catch(const std::exception& e)
            {
                cout << "ModelEngine exception while running: " << e.what() << '\n';
            }

            Log::Merge(context_.Logger);
            Recorder<NeuronRecord>::Merge(context_.Record);
            cout << "Writing log file to " << context_.LogFile << "... " << std::flush;
            Log::Print(context_.LogFile.c_str());
            cout << "Done\n";
            cout << "Writing record file to " << context_.RecordFile << "... " << std::flush;
            Recorder<NeuronRecord>::Print(context_.RecordFile.c_str());
            cout << "Done\n";
        }

        unsigned long long int GetIterations()
        {
            return context_.Iterations;
        }

    private:
        void Initialize()
        {
            if (!helper_.AllocateModel())
                return;
            
            string modelInitializerLocation { "" };
            const json& executionJson = context_.Configuration.Configuration()["Execution"];
            if (!executionJson.is_null())
            {
                const json& initializerLocationJson = executionJson["InitializerLocation"];
                if (initializerLocationJson.is_string())
                    modelInitializerLocation = initializerLocationJson.get<string>();
            }

            if (modelInitializerLocation.empty())
            {
                cout << "No initialization location configured, cannot initialize\n";
                return;
            }

            // Create the proxy with a two-step ctor-create sequence.
            ModelInitializerProxy<GpuModelHelper<RECORDTYPE>> initializer(modelInitializerLocation);
            initializer.CreateProxy(helper_);

            // Let the initializer initialize the model's static state.
            initializer.Initialize();

            cuda::memory::copy(carrier_.NeuronsDevice.get(), carrier_.NeuronsHost.get(), carrier_.ModelSize() * sizeof(NeuronNode));
            cuda::memory::copy(carrier_.SynapsesDevice.get(), carrier_.SynapsesHost.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronSynapse));

            cout << "Calling DeviceFixupShim(" << carrier_.Device.id() << ", " << carrier_.ModelSize() << ", " << "pNeurons, pSynapses)\n";
            DeviceFixupShim(carrier_.Device, carrier_.ModelSize(), carrier_.NeuronsDevice.get(), carrier_.SynapsesDevice.get());
            cout << "Returned from DeviceFixupShim\n";

            string inputStreamerLocation { "" };
            if (!executionJson.is_null())
            {
                const json& inputStreamerJson = executionJson["InputStreamer"];
                if (inputStreamerJson.is_string())
                    inputStreamerLocation = inputStreamerJson.get<string>();
            }

            if (!inputStreamerLocation.empty())
            {
                sensorInput_ = make_unique<SensorInputProxy>(inputStreamerLocation);
                sensorInput_->CreateProxy(context_.Configuration);
                sensorInput_->Connect("");
            }

            context_.Iterations = 1ULL;
            context_.EngineInitialized = true;
        }

        void MainLoop()
        {
            auto engineStartTime = high_resolution_clock::now();
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine starting main loop\n";
            context_.Logger.Logit();
#endif

            auto quit {false};
            do
            {
                quit = WaitForWorkOrQuit();
                if (!quit)
                {
                    // TODO - investigate pipelining by running ExecutaAStep asynchronously and doing StreamInput for the next tick.
                    //StreamInput();
                    auto& streamedInput = sensorInput_->StreamInput(context_.Iterations);
                    helper_.SpikeInputNeurons(streamedInput, context_.Record);

                    cuda::memory::copy(carrier_.NeuronsDevice.get(), carrier_.NeuronsHost.get(), carrier_.ModelSize() * sizeof(NeuronNode));

                    ModelTickShim(carrier_.Device, carrier_.ModelSize());
                    ++context_.Iterations;

                    //ExecuteAStep();
                    ModelSynapsesShim(carrier_.Device, carrier_.ModelSize());
                    ModelTimersShim(carrier_.Device, carrier_.ModelSize());

                    //RecordOutput();
                    cuda::memory::copy(carrier_.NeuronsHost.get(), carrier_.NeuronsDevice.get(), carrier_.ModelSize() * sizeof(NeuronNode));
                    helper_.RecordRelevantNeurons(context_.Record);
                    helper_.PrintMonitoredNeurons();
                } 
            }
            while (!quit);
            auto engineElapsed = duration_cast<microseconds>(high_resolution_clock::now() - engineStartTime).count();
            auto partitionElapsed = context_.PartitionTime.count();

#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine quitting main loop\n";
            context_.Logger.Logit();
#endif

            double partitionRatio = (double)partitionElapsed / (double)engineElapsed;
            cout 
                << "Iterations: " << context_.Iterations 
                << " Total Work: " << context_.TotalWork 
                << " items  Partition Time: " << partitionElapsed << '/' << engineElapsed << " us = " 
                << partitionRatio 
                << "\n";
        }

        bool WaitForWorkOrQuit()
        {
            auto quit { true };
            nextScheduledTick_ += context_.EnginePeriod;
            {
                unique_lock<mutex> lock(context_.Mutex);
                context_.Cv.wait_until(lock, nextScheduledTick_, [this](){ return context_.Quit; });

                quit = context_.Quit;
                nextScheduledTick_ += context_.EnginePeriod;
            }

            return quit;
        }

        void StreamInput()
        {
            auto& streamedInput = sensorInput_->StreamInput(context_.Iterations);
            helper_.SpikeInputNeurons(streamedInput);

            cuda::memory::copy(carrier_.NeuronsDevice.get(), carrier_.NeuronsHost.get(), carrier_.ModelSize() * sizeof(NeuronNode));
        }

        void ExecuteAStep()
        {
            ModelSynapsesShim(carrier_.Device, carrier_.ModelSize());
            ModelTimersShim(carrier_.Device, carrier_.ModelSize());

            //cuda::memory::copy(carrier_.NeuronsHost.get(), carrier_.NeuronsDevice.get(), carrier_.ModelSize() * sizeof(NeuronNode));
            //cuda::memory::copy(carrier_.SynapsesHost.get(), carrier_.SynapsesDevice.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronSynapse));
        }

        void RecordOutput()
        {
            cuda::memory::copy(carrier_.NeuronsHost.get(), carrier_.NeuronsDevice.get(), carrier_.ModelSize() * sizeof(NeuronNode));
            helper_.RecordRelevantNeurons(context_.Record);
            helper_.PrintMonitoredNeurons();
        }

        void Cleanup()
        {
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine closing sensor streaming input\n";
            context_.Logger.Logit();
#endif
                sensorInput_->Disconnect();
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine sensor streaming input closed\n";
            context_.Logger.Logit();
#endif
        }
   };
}
