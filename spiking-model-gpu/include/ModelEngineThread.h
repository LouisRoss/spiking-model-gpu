#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <mutex>

#include "nlohmann/json.hpp"

#include "CudaApi.h"

#include "ConfigurationRepository.h"
#include "Log.h"
#include "Recorder.h"
#include "ModelInitializerProxy.h"
#include "WorkerInputStreamer.h"
#include "WorkerOutputStreamer.h"
#include "WorkerThread.h"

#include "IModelHelper.h"
#include "ModelEngineContext.h"
#include "GpuModelCarrier.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::vector;
    using std::unique_ptr;
    using std::make_unique;
    using std::mutex;
    using std::unique_lock;
    using std::chrono::high_resolution_clock;
    using time_point = std::chrono::high_resolution_clock::time_point;
    using std::chrono::microseconds;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using namespace std::chrono_literals;
    using std::cout;
    using std::cerr;

    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::Log;
    using embeddedpenguins::core::neuron::model::Recorder;
    using embeddedpenguins::core::neuron::model::IModelInitializer;
    using embeddedpenguins::core::neuron::model::ModelInitializerProxy;
    using embeddedpenguins::core::neuron::model::WorkerThread;

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
        ModelEngineContext& context_;
        GpuModelCarrier& carrier_;
        IModelHelper* helper_;

        time_point nextScheduledTick_;
        WorkerThread<WorkerInputStreamer<RECORDTYPE>>& inputStreamThread_;
        WorkerOutputStreamer<RECORDTYPE> outputStreamer_;
        vector<IModelInitializer::SpikeOutputDescriptor> initializedSpikeOutputs_ {};

    public:
        ModelEngineThread() = delete;
        ModelEngineThread(
                        ModelEngineContext& context, 
                        GpuModelCarrier& carrier,
                        WorkerThread<WorkerInputStreamer<RECORDTYPE>>& inputStreamThread,
                        IModelHelper* helper) :
            context_(context),
            carrier_(carrier),
            helper_(helper),
            nextScheduledTick_(high_resolution_clock::now() + context_.EnginePeriod),
            inputStreamThread_(inputStreamThread),
            outputStreamer_(context_, helper_)
        {
            context_.Logger.SetId(0);
        }

        void operator() ()
        {
            do
            {
                while (!context_.Run) { std::this_thread::yield(); }

                try
                {
                    if (Initialize())
                        MainLoop();
                    Cleanup();
                }
                catch(const std::exception& e)
                {
                    cout << "ModelEngine exception while running: " << e.what() << '\n';
                }

                context_.Run = false;

                Log::Merge(context_.Logger);
                cout << "Writing log file to " << context_.LogFile << "... " << std::flush;
                Log::Print(context_.LogFile.c_str());
                cout << "Done\n";
            } while (context_.Run);
        }

        unsigned long long int GetIterations()
        {
            return context_.Measurements.Iterations;
        }

    private:
        bool Initialize()
        {
            if (!InitializeModel())
            {
                cout << "ModelEngineThread.Initialize failed at InitializeModel()\n";
                context_.EngineInitializeFailed = true;
                return false;
            }

#if CUDA_VERSION < 12000
            cuda::memory::copy(carrier_.PostsynapticIncreaseFuncDevice.get(), carrier_.PostsynapticIncreaseFuncHost.get(), PostsynapticPlasticityPeriod * sizeof(float));
            cuda::memory::copy(carrier_.NeuronsDevice.get(), carrier_.NeuronsHost.get(), carrier_.ModelSize() * sizeof(NeuronNode));
            cuda::memory::copy(carrier_.SynapsesDevice.get(), carrier_.PostSynapseHost.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronPostSynapse));
            cuda::memory::copy(carrier_.PreSynapsesDevice.get(), carrier_.PreSynapsesHost.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronPreSynapse));

            DeviceFixupShim(carrier_.Device, carrier_.ModelSize(), carrier_.PostsynapticIncreaseFuncDevice.get(), carrier_.NeuronsDevice.get(), carrier_.SynapsesDevice.get(), carrier_.PreSynapsesDevice.get());
#else
            cudaMemcpy(carrier_.PostsynapticIncreaseFuncDevice(), carrier_.PostsynapticIncreaseFuncHost.get(), PostsynapticPlasticityPeriod * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(carrier_.NeuronsDevice(), carrier_.NeuronsHost.get(), carrier_.ModelSize() * sizeof(NeuronNode), cudaMemcpyHostToDevice);
            cudaMemcpy(carrier_.SynapsesDevice(), carrier_.PostSynapseHost.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronPostSynapse), cudaMemcpyHostToDevice);
            cudaMemcpy(carrier_.PreSynapsesDevice(), carrier_.PreSynapsesHost.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronPreSynapse), cudaMemcpyHostToDevice);

            DeviceFixupShim(carrier_.ModelSize(), carrier_.PostsynapticIncreaseFuncDevice(), carrier_.NeuronsDevice(), carrier_.SynapsesDevice(), carrier_.PreSynapsesDevice());
#endif

            if (!inputStreamThread_.GetImplementation().Valid())
            {
                cout << "ModelEngineThread.Initialize failed at inputStreamer_.Valid()\n";
                context_.EngineInitializeFailed = true;
                return false;
            }

            if (!outputStreamer_.Valid())
            {
                cout << "ModelEngineThread.Initialize failed at outputStreamer_.Valid()\n";
                context_.EngineInitializeFailed = true;
                return false;
            }

            cout << "\n*** Creating " <<  initializedSpikeOutputs_.size() << " Spike Output interconnect proxies\n";
            for (const auto& spikeOutput: initializedSpikeOutputs_)
            {
                outputStreamer_.AddSpikeOutput(spikeOutput);
            }

            context_.Measurements.Iterations = 0ULL;
            context_.EngineInitialized = true;

            return true;
        }

        void MainLoop()
        {
            context_.TriggerStartTime();
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine starting main loop\n";
            context_.Logger.Logit();
#endif
            long long int engineElapsed;

            WorkerThread<WorkerOutputStreamer<RECORDTYPE>> outputStreamThread(outputStreamer_);

            auto quit {false};
            nextScheduledTick_ = high_resolution_clock::now();
            do
            {
                auto needResync { false };
                while (context_.Pause) { needResync = true; std::this_thread::sleep_for(5ms); std::this_thread::yield(); }
                if (needResync) nextScheduledTick_ = high_resolution_clock::now();


                quit = WaitForWorkOrQuit();
                if (!quit)
                {
                    context_.Logger.Logger() << "ModelEngine executing a model step " << context_.Measurements.Iterations << "\n";
                    context_.Logger.Logit();
                    ExecuteAStep(inputStreamThread_, outputStreamThread);
                    context_.Logger.Logger() << "ModelEngine completed a model step\n";
                    context_.Logger.Logit();

                }

                engineElapsed = duration_cast<microseconds>(high_resolution_clock::now() - context_.Measurements.EngineStartTime).count();
            }
            while (!quit);

            context_.TriggerStopTime();
            auto partitionElapsed = context_.Measurements.PartitionTime.count();

#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine quitting main loop\n";
            context_.Logger.Logit();
#endif

            double usPerTick = (double)engineElapsed / (double)context_.Measurements.Iterations;
            cout 
                << "Iterations: " << context_.Measurements.Iterations 
                << " Total Work: " << context_.Measurements.TotalWork 
                << " items  Partition Time: " << partitionElapsed << '/' << engineElapsed << " tick = " 
                << usPerTick 
                << " us\n";
        }

        bool WaitForWorkOrQuit()
        {
            auto quit { true };

            {
                unique_lock<mutex> lock(context_.Mutex);
                context_.Cv.wait_until(lock, nextScheduledTick_, [this](){ return !context_.Run; });

                quit = !context_.Run;
                nextScheduledTick_ += context_.EnginePeriod;
            }

            return quit;
        }

        bool InitializeModel()
        {
            string modelInitializerLocation { "" };
            if (context_.Configuration.Control().contains("Execution"))
            {
                const json& executionJson = context_.Configuration.Control()["Execution"];
                if (executionJson.contains("InitializerLocation"))
                {
                    const json& initializerLocationJson = executionJson["InitializerLocation"];
                    if (initializerLocationJson.is_string())
                        modelInitializerLocation = initializerLocationJson.get<string>();
                }
            }

            if (modelInitializerLocation.empty())
            {
                cout << "No initialization location configured, cannot initialize\n";
                return false;
            }

            // Create the proxy with a two-step ctor-create sequence.
            ModelInitializerProxy initializer(modelInitializerLocation);
            initializer.CreateProxy(helper_);

            // Let the initializer initialize the model's static state.
            auto success = initializer.Initialize();

            initializedSpikeOutputs_.clear();
            if (success)
            {
                initializedSpikeOutputs_ = initializer.GetInitializedOutputs();
            }

            return success;
        }

        void ExecuteAStep(WorkerThread<WorkerInputStreamer<RECORDTYPE>>& inputStreamThread,
                WorkerThread<WorkerOutputStreamer<RECORDTYPE>>& outputStreamThread)
        {
            // Get input for this tick, copy input to device.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine streaming input into model\n";
            context_.Logger.Logit();
#endif
            inputStreamThread.WaitForPreviousScan();
            auto& streamedInput = inputStreamThread_.GetImplementation().StreamedInput();
#ifdef STREAM_CPU
            helper_->SpikeInputNeurons(streamedInput, context_.Record);
            cuda::memory::copy(carrier_.NeuronsDevice.get(), carrier_.NeuronsHost.get(), carrier_.ModelSize() * sizeof(NeuronNode));
#else
            if (!streamedInput.empty())
            {
#if CUDA_VERSION < 12000
                cuda::memory::copy(carrier_.InputSignalsDevice.get(), &streamedInput[0], streamedInput.size() * sizeof(unsigned long long));
                StreamInputShim(carrier_.Device, carrier_.ModelSize(), streamedInput.size(), carrier_.InputSignalsDevice.get());
#else
                cout << "Copying " << streamedInput.size() << " inputs to CUDA device memory\n";
                cudaMemcpy(carrier_.InputSignalsDevice(), &streamedInput[0], streamedInput.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice);
                StreamInputShim(carrier_.ModelSize(), streamedInput.size(), carrier_.InputSignalsDevice());
#endif
            }
#endif
            inputStreamThread.Scan();

            // Execute the model.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling synapse kernel\n";
            context_.Logger.Logit();
#endif
#if CUDA_VERSION < 12000
            ModelSynapsesShim(carrier_.Device, carrier_.ModelSize());
#else
            ModelSynapsesShim(carrier_.ModelSize());
#endif
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling timer kernel\n";
            context_.Logger.Logit();
#endif
#if CUDA_VERSION < 12000
            ModelTimersShim(carrier_.Device, carrier_.ModelSize());
#else
            ModelTimersShim(carrier_.ModelSize());
#endif
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling plasticity kernel\n";
            context_.Logger.Logit();
#endif
#if CUDA_VERSION < 12000
            ModelPlasticityShim(carrier_.Device, carrier_.ModelSize());
#else
            ModelPlasticityShim(carrier_.ModelSize());
#endif

            // Copy device to host, capture output.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine recording neurons\n";
            context_.Logger.Logit();
#endif
            outputStreamThread.WaitForPreviousScan();
#if CUDA_VERSION < 12000
            cuda::memory::copy(carrier_.NeuronsHost.get(), carrier_.NeuronsDevice.get(), carrier_.ModelSize() * sizeof(NeuronNode));
            if (context_.RecordSynapseEnable)
            {
                cuda::memory::copy(carrier_.PostSynapseHost.get(), carrier_.SynapsesDevice.get(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronPostSynapse));
            }
#else
            cudaMemcpy(carrier_.NeuronsHost.get(), carrier_.NeuronsDevice(), carrier_.ModelSize() * sizeof(NeuronNode), cudaMemcpyDeviceToHost);
            if (context_.RecordSynapseEnable)
            {
                cudaMemcpy(carrier_.PostSynapseHost.get(), carrier_.SynapsesDevice(), carrier_.ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronPostSynapse), cudaMemcpyDeviceToHost);
            }
#endif
            outputStreamThread.Scan();

            // Advance all ticks in the model.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling tick kernel\n";
            context_.Logger.Logit();
#endif
#if CUDA_VERSION < 12000
            ModelTickShim(carrier_.Device, carrier_.ModelSize());
#else
            ModelTickShim(carrier_.ModelSize());
#endif
            ++context_.Measurements.Iterations;
        }

        void Cleanup()
        {
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine closing sensor streaming input\n";
            context_.Logger.Logit();
#endif
            //inputStreamer_.DisconnectInputStream();
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine sensor streaming input closed\n";
            context_.Logger.Logit();
#endif
        }
   };
}
