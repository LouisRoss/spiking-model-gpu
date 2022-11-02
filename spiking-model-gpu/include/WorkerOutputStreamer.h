#pragma once
#include <string>
#include <sstream>
#include <memory>
#include <vector>

#include "SpikeOutputProxy.h"

#include "IModelHelper.h"
#include "Initializers/IModelInitializer.h"
#include "ModelEngineContext.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::stringstream;
    using std::unique_ptr;
    using std::make_unique;
    using std::vector;

    using embeddedpenguins::core::neuron::model::ISpikeOutput;
    using embeddedpenguins::core::neuron::model::IModelInitializer;
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

        void AddSpikeOutput(const IModelInitializer::SpikeOutputDescriptor& spikeOutput)
        {
            CreateInterconnectProxy(spikeOutput);
        }

        void Process()
        {
            auto relevantNeurons = helper_->CollectRelevantNeurons(context_.RecordSynapseEnable, context_.RecordActivationEnable, context_.RecordHyperSensitiveEnable);
            for (auto [index, activation, hyperactive, synapseIndex, synapseStrength, type] : relevantNeurons)
            {
                for (auto& spikeOutput : spikeOutputs_)
                {
                    // If an output streamer respects the disable flag, it should be skipped if the enable flag is off.
                    if (!spikeOutput->RespectDisableFlag() || context_.RecordEnable)
                    {
                        spikeOutput->StreamOutput(index, activation, hyperactive, synapseIndex, synapseStrength, type);
                    }
                }
            }

            for (auto& spikeOutput : spikeOutputs_)
            {
                spikeOutput->Flush();
            }

            //helper_->PrintMonitoredNeurons();
        }

        void Cleanup()
        {
            if (!valid_)
                return;

            for (auto& spikeOutput : spikeOutputs_)
            {
                spikeOutput->Disconnect();
            }
        }

    private:
        void CreateInterconnectProxy(const IModelInitializer::SpikeOutputDescriptor& spikeOutput)
        {
            cout << "\n*** Creating Spike Output interconnect proxies from 'Execution' section of configuration file\n";
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
                        if (outputStreamerJson.contains("Interconnect"))
                        {
                            const json& interconnectJson = outputStreamerJson["Interconnect"];
                            if (interconnectJson.is_string())
                            {
                                outputStreamerLocation = interconnectJson.get<string>();
                            }
                        }

                        int basePort { 0 };
                        if (outputStreamerJson.contains("BasePort"))
                        {
                            const json& basePortJson = outputStreamerJson["BasePort"];
                            if (basePortJson.is_number_integer())
                            {
                                basePort = basePortJson.get<int>();
                            }
                        }

                        if (!outputStreamerLocation.empty())
                        {
                            cout << "Creating interconnect output streamer proxy " << outputStreamerLocation << "\n";
                            auto proxy = make_unique<SpikeOutputProxy>(outputStreamerLocation);
                            proxy->CreateProxy(context_);


                            stringstream ss;
                            ss << spikeOutput.Host << ":" << basePort;
                            auto connectionString { ss.str() };
                            unsigned long localStart = helper_->GetExpansionMap().ExpansionOffset(spikeOutput.LocalModelSequence) + spikeOutput.LocalModelOffset;

                            cout << "Connecting output streamer to " << connectionString << "\n";
                            if (proxy->Connect(connectionString, localStart, spikeOutput.Size, spikeOutput.ModelSequence, spikeOutput.ModelOffset))
                                spikeOutputs_.push_back(std::move(proxy));
                            else
                                cout << "Unable to connect to interconnect output streamer at " << connectionString << "\n";

                            break;
                        }
                    }
                }
            }

            cout << "***Created " << spikeOutputs_.size() << " spike output proxy objects\n";
        }

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
                            CreateProxy(outputStreamerLocation);
                        }
                    }
                }
            }

            cout << "***Created " << spikeOutputs_.size() << " spike output proxy objects\n";
        }

        void CreateProxy(const string& outputStreamerLocation)
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
    };
}
