#pragma once

#include <thread>
#include <chrono>

#include "IModelHelper.h"
#include "ConfigurationRepository.h"
#include "ModelEngineContext.h"
#include "ModelEngineThread.h"
#include "GpuModelCarrier.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::thread;
    using std::lock_guard;
    using std::chrono::microseconds;
    using std::chrono::nanoseconds;
    using std::chrono::high_resolution_clock;
    using time_point = std::chrono::high_resolution_clock::time_point;

    //
    // The top-level control engine for running a model.
    // Create and run a single thread, which will create N worker objects, each
    // with its own thread.  By default, N will be the number of hardware cores - 1
    // so that between the model engine thread and the worker thread, all cores
    // will be kept busy.
    //
    template<class RECORDTYPE>
    class ModelEngine
    {
        ModelEngineContext& context_;
        thread workerThread_;
        nanoseconds duration_ {};
        time_point startTime_ {};

    public:
        const long long int GetTotalWork() const { return context_.Measurements.TotalWork; }
        const long long int GetIterations() const { return context_.Measurements.Iterations; }
        const nanoseconds GetDuration() const { return duration_; }
        const microseconds EnginePeriod() const { return context_.EnginePeriod; }
        microseconds& EnginePeriod() { return context_.EnginePeriod; }
        ModelEngineContext& Context() { return context_; }
        void Pause() { context_.Pause = true; }
        void Continue() { context_.Pause = false; }

    public:
        ModelEngine() = delete;

        ModelEngine(ModelEngineContext& context, 
                    GpuModelCarrier& carrier, 
                    ConfigurationRepository& configuration, 
                    RunMeasurements& runMeasurements, 
                    WorkerThread<WorkerInputStreamer<RECORDTYPE>>& inputStreamThread,
                    IModelHelper* helper) :
            context_(context)
        {
            cout << "\nCreating new ModelEngine\n";
            workerThread_ = thread(ModelEngineThread<RECORDTYPE>(context, carrier, inputStreamThread, helper));
        }

        ~ModelEngine()
        {
            WaitForQuit();

            cout << "Model Engine stopping\n";
        }

    public:
        bool Initialize()
        {
            cout << "\n***Initializing model engine\n";
            auto initialized = context_.Initialize();
            cout << "***Model engine initialized\n\n";

            return initialized;
        }

        bool Run()
        {
            cout << "\n***Running model engine\n";
            startTime_ = high_resolution_clock::now();
            context_.Run = true;

            // Don't return until the woker thread is initialized.
            while (!context_.EngineInitialized && !context_.EngineInitializeFailed)
                std::this_thread::yield();

            context_.TriggerStartTime();
            return context_.EngineInitialized && !context_.EngineInitializeFailed;
        }

        void Quit()
        {
            cout << "\n***Stopping model engine\n";
            context_.Run = false;
            /*
            {
                lock_guard<mutex> lock(context_.Mutex);
                context_.Quit = true;
            }
            */
            context_.Cv.notify_one();
        }

        void WaitForQuit()
        {
            auto endTime = high_resolution_clock::now();
            duration_ = endTime - startTime_;

            Quit();

            if (workerThread_.joinable())
                workerThread_.join();
        }
    };
}
