#pragma once

#include <thread>
#include <chrono>

#include "core/ConfigurationRepository.h"
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
        ModelEngineContext<RECORDTYPE> context_;
        thread workerThread_;
        nanoseconds duration_ {};
        time_point startTime_ {};

    public:
        const long long int GetTotalWork() const { return context_.TotalWork; }
        const long long int GetIterations() const { return context_.Iterations; }
        const nanoseconds GetDuration() const { return duration_; }
        //const string& LogFile() const { return context_.LogFile; }
        //void LogFile(const string& logfile) { context_.LogFile = logfile; }
        //const string& RecordFile() const { return context_.RecordFile; }
        //void RecordFile(const string& recordfile) { context_.RecordFile = recordfile; }
        const microseconds EnginePeriod() const { return context_.EnginePeriod; }
        microseconds& EnginePeriod() { return context_.EnginePeriod; }

    public:
        ModelEngine() = delete;

        ModelEngine(GpuModelCarrier& carrier, const ConfigurationRepository& configuration, GpuModelHelper<RECORDTYPE>& helper) :
            context_(configuration, helper)
        {
            workerThread_ = thread(ModelEngineThread<RECORDTYPE>(context_, helper));
        }

        ~ModelEngine()
        {
            WaitForQuit();

            cout << "Model Engine stopping\n";
        }

    public:
        void Run()
        {
            startTime_ = high_resolution_clock::now();
            context_.Run = true;

            // Don't return until the woker thread is initialized.
            while (!context_.EngineInitialized)
                std::this_thread::yield();
        }

        void Quit()
        {
            {
                lock_guard<mutex> lock(context_.Mutex);
                context_.Quit = true;
            }
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
