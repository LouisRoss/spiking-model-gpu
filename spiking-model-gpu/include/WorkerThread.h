#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace embeddedpenguins::gpu::neuron::model
{
    using std::cout;
    using std::thread;
    using std::mutex;
    using std::condition_variable;
    using std::unique_lock;
    using std::lock_guard;

    //
    //
    // NOTE the derived class is expected to implement the required methods,
    //      but no interface exists to enforce the implementation.  Run time
    //      will be faster using compile-time polymorphism with templates rather
    //      than run-time polymorphism, as having the derived class override 
    //      virtual methods.
    //      In any case, if a required method is missing or malformed in the
    //      derived class, it will fail to compile with a similar error as
    //      would happen with an interface class.
    //
    template<class IMPLEMENTATIONTYPE, class RECORDTYPE>
    class WorkerThread
    {
        enum class WorkCode
        {
            Run,
            Quit,
            Scan
        };

        mutex mutex_ {};
        condition_variable cv_ {};
        mutex mutexReturn_ {};
        condition_variable cvReturn_ {};
        bool cycleStart_{false};
        bool cycleDone_{false};
        bool firstScan_ { true };

        WorkCode code_{WorkCode::Run};
        thread workerThread_;

        IMPLEMENTATIONTYPE& implementation_;

    public:
        WorkerThread(IMPLEMENTATIONTYPE& implementation) :
            implementation_(implementation)
        {
            workerThread_ = thread(std::ref(*this));
        }

        ~WorkerThread()
        {
            Join();
        }

        void operator() ()
        {
            while (code_ != WorkCode::Quit)
            {
                WaitForSignal();

                if (code_ == WorkCode::Scan)
                {
                    implementation_.Process();
                }

                SignalDone();
            }

            SignalDone();
        }

        void WaitForPreviousScan()
        {
            if (firstScan_)
            {
                return;
            }

            unique_lock<mutex> lock(mutexReturn_);
            cvReturn_.wait(lock, [this]{ return cycleDone_; });
        }

        void Scan()
        {
            Scan(WorkCode::Scan);
        }

        void Join()
        {
            Scan(WorkCode::Quit);
            workerThread_.join();
        }

    private:
        void WaitForSignal()
        {
            unique_lock<mutex> lock(mutex_);
            cv_.wait(lock, [this]{ return cycleStart_; });
            cycleStart_ = false;
            firstScan_ = false;
        }

        void SignalDone()
        {
            {
                lock_guard<mutex> lock(mutexReturn_);
                cycleDone_ = true;
            }
            cvReturn_.notify_one();
        }

        void Scan(WorkCode code)
        {
            WaitForPreviousScan();
            SetDataForScan(code);

            cv_.notify_one();
        }

        void SetDataForScan(WorkCode code)
        {
            lock_guard<mutex> lock(mutex_);
            code_ = code;
            cycleDone_ = false;
            cycleStart_ = true;
        }
    };
}
