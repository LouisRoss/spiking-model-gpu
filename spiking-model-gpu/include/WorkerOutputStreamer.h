#pragma once

#include "ModelEngineContext.h"
#include "GpuModelHelper.h"

namespace embeddedpenguins::gpu::neuron::model
{
    //
    // Encapsulate the streaming of output to the configured output streaming source
    // as well as recording in a Process() method, suitable for calling by WorkerThread.
    // Thus, this class may be called directly or may be provided to WorkerThread
    // so it may be run in a separate thread.
    //
    template<class RECORDTYPE>
    class WorkerOutputStreamer
    {
        ModelEngineContext<RECORDTYPE>& context_;
        GpuModelHelper<RECORDTYPE>& helper_;
        bool valid_;

    public:
        const bool Valid() const { return valid_; }

    public:
        WorkerOutputStreamer(ModelEngineContext<RECORDTYPE>& context, GpuModelHelper<RECORDTYPE>& helper) :
            context_(context),
            helper_(helper),
            valid_(true)
        {
        }

        void Process()
        {
            helper_.RecordRelevantNeurons(context_.Record);
            //helper_.PrintMonitoredNeurons();
        }
    };
}
