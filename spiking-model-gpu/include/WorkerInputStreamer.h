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
        vector<unsigned long long int> dummyStreamedInput_ { };

        // NOTE: The declaration order of these fields is important.
        ModelEngineContext<RECORDTYPE>& context_;
        unique_ptr<ISensorInput> sensorInput_;
        bool valid_;
        vector<unsigned long long int>& streamedInput_;

    public:
        const bool Valid() const { return valid_; }
        const vector<unsigned long long int>& StreamedInput() const { return streamedInput_; }

    public:
        WorkerInputStreamer(ModelEngineContext<RECORDTYPE>& context) :
            context_(context),
            sensorInput_(CreateProxy()),
            valid_(sensorInput_->Connect("")),
            streamedInput_(valid_ ? sensorInput_->AcquireBuffer() : dummyStreamedInput_)
        {
        }

        void Process()
        {
            if (valid_)
                streamedInput_ = sensorInput_->StreamInput(context_.Iterations);
        }

        void DisconnectInputStream()
        {
            if (valid_)
                sensorInput_->Disconnect();
        }

    private:
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
    };
}
