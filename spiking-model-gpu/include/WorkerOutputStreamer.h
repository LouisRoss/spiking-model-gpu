#pragma once
#include <string>
#include <memory>
#include <vector>

#include "SpikeOutputProxy.h"

#include "IModelHelper.h"
#include "ModelEngineContext.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
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
        IModelHelper* helper_;
        bool valid_;

    public:
        const bool Valid() const { return valid_; }

    public:
        WorkerOutputStreamer(ModelEngineContext& context, IModelHelper* helper) :
            context_(context),
            helper_(helper),
            valid_(true)
        {
            CreateProxies();
        }

        void Process()
        {
            auto relevantNeurons = helper_->CollectRelevantNeurons(context_.RecordSynapseEnable);
            for (auto [index, activation, synapseIndex, synapseStrength, type] : relevantNeurons)
            {
                for (auto& spikeOutput : spikeOutputs_)
                {
                    // If an output streamer respects the disable flag, it should be skipped if the enable flag is off.
                    if (!spikeOutput->RespectDisableFlag() || context_.RecordEnable)
                    {
                        spikeOutput->StreamOutput(index, activation, synapseIndex, synapseStrength, type);
                    }
                }
            }

            for (auto& spikeOutput : spikeOutputs_)
            {
                spikeOutput->Flush();
            }

            //helper_->PrintMonitoredNeurons();
        }

    private:
        void CreateProxies()
        {
            cout << "\n*** Creating Spike Output proxies from 'Execution' section of configuration file\n";
            if (!context_.Configuration.Control().contains("Execution"))
            {
                cout << "Control contains no 'Execution' element, not creating any output streamers\n";
                return;
            }

            const json& executionJson = context_.Configuration.Control()["Execution"];
            if (!executionJson.contains("OutputStreamers"))
            {
                cout << "Control 'Execution' element contains no 'OutputStreamers' subelement, not creating any output streamers\n";
                return;
            }

            const json& outputStreamersJson = executionJson["OutputStreamers"];
            if (outputStreamersJson.is_array())
            {
                for (auto& [key, outputStreamerJson] : outputStreamersJson.items())
                {
                    if (outputStreamerJson.is_object())
                    {
                        string outputStreamerLocation { "" };
                        if (outputStreamerJson.contains("Location"))
                        {
                            const json& locationJson = outputStreamerJson["Location"];
                            if (locationJson.is_string())
                            {
                                outputStreamerLocation = locationJson.get<string>();
                            }
                        }

                        if (!outputStreamerLocation.empty())
                        {
                            cout << "Creating output streamer proxy " << outputStreamerLocation << "\n";
                            auto proxy = make_unique<SpikeOutputProxy>(outputStreamerLocation);
                            proxy->CreateProxy(context_);

                            cout << "Connecting output streamer " << outputStreamerLocation << "'\n";
                            if (proxy->Connect())
                                spikeOutputs_.push_back(std::move(proxy));
                            else
                                cout << "Unable to connect to output streamer " << outputStreamerLocation << "\n";
                        }
                    }
                }
            }

            cout << "***Created " << spikeOutputs_.size() << " spike output proxy objects\n";
        }
    };
}
