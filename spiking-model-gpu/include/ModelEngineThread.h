#pragma once

#include <iostream>
#include <string>
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

#include "ModelEngineContext.h"
#include "GpuModelHelper.h"
#include "GpuModelCarrier.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::unique_ptr;
    using std::make_unique;
    using std::mutex;
    using std::unique_lock;
    using std::chrono::high_resolution_clock;
    using time_point = std::chrono::high_resolution_clock::time_point;
    using std::chrono::microseconds;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::cout;
    using std::cerr;

    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::Log;
    using embeddedpenguins::core::neuron::model::Recorder;
    using embeddedpenguins::core::neuron::model::ModelInitializerProxy;

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
        GpuModelHelper<RECORDTYPE>& helper_;

        time_point nextScheduledTick_;
        WorkerInputStreamer<RECORDTYPE> inputStreamer_;
        WorkerOutputStreamer<RECORDTYPE> outputStreamer_;

    public:
        ModelEngineThread() = delete;
        ModelEngineThread(
                        ModelEngineContext<RECORDTYPE>& context, 
                        GpuModelHelper<RECORDTYPE>& helper) :
            context_(context),
            helper_(helper),
            nextScheduledTick_(high_resolution_clock::now() + context_.EnginePeriod),
            inputStreamer_(context_),
            outputStreamer_(context_, helper_)
        {
            context_.Logger.SetId(0);
        }

        void operator() ()
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

            Log::Merge(context_.Logger);
            Recorder<NeuronRecord>::Merge(context_.Record);
            cout << "Writing log file to " << context_.LogFile << "... " << std::flush;
            Log::Print(context_.LogFile.c_str());
            cout << "Done\n";
            cout << "Writing record file to " << context_.RecordFile << "... " << std::flush;
            Recorder<NeuronRecord>::Print(context_.RecordFile.c_str());
            cout << "Done\n";

            context_.Run = false;
        }

        unsigned long long int GetIterations()
        {
            return context_.Iterations;
        }

    private:
        bool Initialize()
        {
            /*
            if (!context_.Helper.AllocateModel())
            {
                cout << "ModelEngineThread.Initialize failed at context_.Helper.AllocateModel()\n";
                context_.EngineInitializeFailed = true;
                return false;
            }
            */

            if (!InitializeModel())
            {
                cout << "ModelEngineThread.Initialize failed at InitializeModel()\n";
                context_.EngineInitializeFailed = true;
                return false;
            }

            cuda::memory::copy(helper_.Carrier().NeuronsDevice.get(), helper_.Carrier().NeuronsHost.get(), helper_.Carrier().ModelSize() * sizeof(NeuronNode));
            cuda::memory::copy(helper_.Carrier().SynapsesDevice.get(), helper_.Carrier().SynapsesHost.get(), helper_.Carrier().ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronSynapse));

            DeviceFixupShim(helper_.Carrier().Device, helper_.Carrier().ModelSize(), helper_.Carrier().NeuronsDevice.get(), helper_.Carrier().SynapsesDevice.get());

            if (!inputStreamer_.Valid())
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

            context_.Iterations = 1ULL;
            context_.EngineInitialized = true;

            return true;
        }

        void MainLoop()
        {
            auto engineStartTime = high_resolution_clock::now();
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine starting main loop\n";
            context_.Logger.Logit();
#endif
            long long int engineElapsed;

            WorkerThread<WorkerInputStreamer<RECORDTYPE>, RECORDTYPE> inputStreamThread(inputStreamer_);
            WorkerThread<WorkerOutputStreamer<RECORDTYPE>, RECORDTYPE> outputStreamThread(outputStreamer_);

            auto quit {false};
            do
            {
                quit = WaitForWorkOrQuit();
                if (!quit)
                {
                    context_.Logger.Logger() << "ModelEngine executing a model step\n";
                    context_.Logger.Logit();
                    ExecuteAStep(inputStreamThread, outputStreamThread);
                    context_.Logger.Logger() << "ModelEngine completed a model step\n";
                    context_.Logger.Logit();

                }

                engineElapsed = duration_cast<microseconds>(high_resolution_clock::now() - engineStartTime).count();
            }
            while (!quit);

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
                return false;
            }

            // Create the proxy with a two-step ctor-create sequence.
            ModelInitializerProxy<GpuModelHelper<RECORDTYPE>> initializer(modelInitializerLocation);
            initializer.CreateProxy(context_.Helper);

            // Let the initializer initialize the model's static state.
            initializer.Initialize();

            return true;
        }

        void ExecuteAStep(WorkerThread<WorkerInputStreamer<RECORDTYPE>, RECORDTYPE>& inputStreamThread,
                WorkerThread<WorkerOutputStreamer<RECORDTYPE>, RECORDTYPE>& outputStreamThread)
        {
            // Get input for this tick, copy input to device.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine streaming input into model\n";
            context_.Logger.Logit();
#endif
            inputStreamThread.WaitForPreviousScan();
            auto& streamedInput = inputStreamer_.StreamedInput();
#ifdef STREAM_CPU
            helper_.SpikeInputNeurons(streamedInput, context_.Record);
            cuda::memory::copy(helper_.Carrier().NeuronsDevice.get(), helper_.Carrier().NeuronsHost.get(), helper_.Carrier().ModelSize() * sizeof(NeuronNode));
#else
            if (!streamedInput.empty())
            {
                cuda::memory::copy(helper_.Carrier().InputSignalsDevice.get(), &streamedInput[0], streamedInput.size() * sizeof(unsigned long long));
                StreamInputShim(helper_.Carrier().Device, helper_.Carrier().ModelSize(), streamedInput.size(), helper_.Carrier().InputSignalsDevice.get());
            }
#endif
            inputStreamThread.Scan();

            // Execute the model.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling synapse kernel\n";
            context_.Logger.Logit();
#endif
            ModelSynapsesShim(helper_.Carrier().Device, helper_.Carrier().ModelSize());
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling timer kernel\n";
            context_.Logger.Logit();
#endif
            ModelTimersShim(helper_.Carrier().Device, helper_.Carrier().ModelSize());
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling plasticity kernel\n";
            context_.Logger.Logit();
#endif
            ModelPlasticityShim(helper_.Carrier().Device, helper_.Carrier().ModelSize());

            // Advance all ticks in the model.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine calling tick kernel\n";
            context_.Logger.Logit();
#endif
            ModelTickShim(helper_.Carrier().Device, helper_.Carrier().ModelSize());
            ++context_.Iterations;

            // Copy device to host, capture output.
#ifndef NOLOG
            context_.Logger.Logger() << "  ModelEngine recording neurons\n";
            context_.Logger.Logit();
#endif
            outputStreamThread.WaitForPreviousScan();
            cuda::memory::copy(helper_.Carrier().NeuronsHost.get(), helper_.Carrier().NeuronsDevice.get(), helper_.Carrier().ModelSize() * sizeof(NeuronNode));
            //cuda::memory::copy(helper_.Carrier().SynapsesHost.get(), helper_.Carrier().SynapsesDevice.get(), helper_.Carrier().ModelSize() * SynapticConnectionsPerNode * sizeof(NeuronSynapse));
            outputStreamThread.Scan();
        }

        void Cleanup()
        {
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine closing sensor streaming input\n";
            context_.Logger.Logit();
#endif
            inputStreamer_.DisconnectInputStream();
#ifndef NOLOG
            context_.Logger.Logger() << "ModelEngine sensor streaming input closed\n";
            context_.Logger.Logit();
#endif
        }
   };
}
