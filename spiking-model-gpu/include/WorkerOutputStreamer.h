#pragma once
#include <memory>
#include <vector>

#include "SpikeOutputProxy.h"

#include "ModelEngineContext.h"
#include "GpuModelHelper.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::unique_ptr;
    using std::make_unique;
    using std::vector;

    using embeddedpenguins::core::neuron::model::ISpikeOutput;
    using embeddedpenguins::core::neuron::model::SpikeOutputProxy;

    //
    // Encapsulate the streaming of output to the configured output streaming source
    // as well as recording in a Process() method, suitable for calling by WorkerThread.
    // Thus, this class may be called directly or may be provided to WorkerThread
    // so it may be run in a separate thread.
    //
    template<class RECORDTYPE>
    class WorkerOutputStreamer
    {
        ModelEngineContext& context_;
        vector<unique_ptr<ISpikeOutput>> spikeOutputs_ {};
        GpuModelHelper& helper_;
        bool valid_;

    public:
        const bool Valid() const { return valid_; }

    public:
        WorkerOutputStreamer(ModelEngineContext& context, GpuModelHelper& helper) :
            context_(context),
            helper_(helper),
            valid_(true)
        {
            CreateProxies();
        }

        void Process()
        {
            auto relevantNeurons = helper_.CollectRelevantNeurons();
            for (auto [index, activation, type] : relevantNeurons)
            {
                for (auto& spikeOutput : spikeOutputs_)
                {
                    spikeOutput->StreamOutput(index, activation, type);
                }
            }

            //helper_.PrintMonitoredNeurons();
        }

    private:
        void CreateProxies()
        {
            string outputStreamerLocation { "" };
            const json& executionJson = context_.Configuration.Configuration()["Execution"];
            if (!executionJson.is_null())
            {
                const json& outputStreamersJson = executionJson["OutputStreamers"];
                if (outputStreamersJson.is_array())
                {
                    vector<string> outputStreamerNames = outputStreamersJson.get<vector<string>>();
                    for (auto& outputStreamerLocation : outputStreamerNames)
                    {
                        ISpikeOutput* proxy { nullptr };
                        if (!outputStreamerLocation.empty())
                        {
                            cout << "Creating output streamer proxy " << outputStreamerLocation << "\n";
                            auto proxy = make_unique<SpikeOutputProxy>(outputStreamerLocation);
                            proxy->CreateProxy(context_);
                            spikeOutputs_.push_back(std::move(proxy));
                        }
                    }
                }
            }
        }
    };
}
