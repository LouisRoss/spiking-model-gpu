#pragma once

#include <string>
#include <memory>
#include <vector>

#include "nlohmann/json.hpp"

#include "ModelEngineContext.h"
#include "SensorInputProxy.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::unique_ptr;
    using std::make_unique;
    using std::vector;

    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::ISensorInput;
    using embeddedpenguins::core::neuron::model::SensorInputProxy;

    //
    // Encapsulate the streaming of input from the configured input streaming source
    // in a Process() method, suitable for calling by WorkerThread.
    // Thus, this class may be called directly or may be provided to WorkerThread
    // so it may be run in a separate thread.
    //
    template<class RECORDTYPE>
    class WorkerInputStreamer
    {
        ModelEngineContext& context_;
        bool valid_;
        vector<unsigned long long int> streamedInput_;
        vector<unique_ptr<ISensorInput>> sensorInputs_ {};

    public:
        const bool Valid() const { return valid_; }
        const vector<unsigned long long int>& StreamedInput() const { return streamedInput_; }

    public:
        WorkerInputStreamer(ModelEngineContext& context) :
            context_(context),
            valid_(true)
        {
            CreateProxies();
        }

        void Process()
        {
            if (!valid_)
                return;

            streamedInput_.clear();
            for (auto& sensorInput : sensorInputs_)
            {
                auto& inputs = sensorInput->StreamInput(context_.Iterations);
                streamedInput_.insert(streamedInput_.end(), inputs.begin(), inputs.end());
                inputs.clear();
            }
        }

        void DisconnectInputStream()
        {
            if (!valid_)
                return;

            for (auto& sensorInput : sensorInputs_)
            {
                sensorInput->Disconnect();
            }
        }

    private:
    /*
        ISensorInput* CreateProxy()
        {
            string inputStreamerLocation { "" };
            const json& executionJson = context_.Configuration.Configuration()["Execution"];
            if (!executionJson.is_null())
            {
                const json& inputStreamerJson = executionJson["InputStreamer"];
                if (inputStreamerJson.is_string())
                    inputStreamerLocation = inputStreamerJson.get<string>();
            }

            ISensorInput* proxy { nullptr };
            if (!inputStreamerLocation.empty())
            {
                proxy = new SensorInputProxy(inputStreamerLocation);
                proxy->CreateProxy(context_.Configuration);
            }

            return proxy;
        }
    */

        void CreateProxies()
        {
            if (!context_.Configuration.Configuration().contains("Execution"))
            {
                cout << "Configuration contains no 'Execution' element, not creating any input streamers\n";
                return;
            }

            const json& executionJson = context_.Configuration.Configuration()["Execution"];
            if (!executionJson.contains("InputStreamers"))
            {
                cout << "Configuration 'Execution' element contains no 'InputStreamers' subelement, not creating any input streamers\n";
                return;
            }

            const json& inputStreamersJson = executionJson["InputStreamers"];
            if (inputStreamersJson.is_array())
            {
                for (auto& [key, inputStreamerJson] : inputStreamersJson.items())
                {
                    if (inputStreamerJson.is_object())
                    {
                        string inputStreamerLocation { "" };
                        if (inputStreamerJson.contains("Location"))
                        {
                            const json& locationJson = inputStreamerJson["Location"];
                            if (locationJson.is_string())
                            {
                                inputStreamerLocation = locationJson.get<string>();
                            }
                        }

                        string inputStreamerConnectionString { "" };
                        if (inputStreamerJson.contains("ConnectionString"))
                        {
                            const json& connectionStringJson = inputStreamerJson["ConnectionString"];
                            if (connectionStringJson.is_string())
                                inputStreamerConnectionString = connectionStringJson.get<string>();
                        }

                        if (!inputStreamerLocation.empty())
                        {
                            cout << "Creating input streamer proxy " << inputStreamerLocation << "\n";
                            auto proxy = make_unique<SensorInputProxy>(inputStreamerLocation);
                            proxy->CreateProxy(context_.Configuration);

                            cout << "Connecting output streamer " << inputStreamerLocation << " to '" << inputStreamerConnectionString << "'\n";
                            if (proxy->Connect(inputStreamerConnectionString))
                                sensorInputs_.push_back(std::move(proxy));
                            else
                                cout << "Unable to connect to output streamer " << inputStreamerLocation << "\n";
                        }
                    }
                }
            }

            cout << "Created " << sensorInputs_.size() << " spike output proxy objects\n";
        }
    };
}
