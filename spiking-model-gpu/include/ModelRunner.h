#pragma once

#include "sys/stat.h"
#include <algorithm>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>

#include "nlohmann/json.hpp"

#include "ConfigurationRepository.h"
#include "ModelInitializerProxy.h"

#include "GpuModelCarrier.h"
#include "GpuModelHelper.h"
#include "ModelEngine.h"
#include "ICommandControlAcceptor.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::begin;
    using std::end;
    using std::remove;
    using std::unique_ptr;
    using std::make_unique;
    using std::vector;
    using std::cout;
    using std::ifstream;
    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::ModelInitializerProxy;
    using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;

    //
    // Wrap the most common startup and teardown sequences to run a model
    // in a single class.  A typical application can just instantiate a 
    // ModelRunner object and call its Run() method to start the model running.
    // Call its Quit() method to make the model stop at the end of the next tick.
    // Call its WaitForQuit() method to confirm that the model engine as stopped and
    // cleaned up.
    //
    // When using the ModelRunner to run a model, it owns the model (a vector of NODETYPE),
    // the model engine object, and all configuration defined for the model.
    //
    template<class RECORDTYPE>
    class ModelRunner
    {
        bool valid_ { false };
        string reason_ {};
        string controlFile_ {};

        ConfigurationRepository configuration_ {};
        GpuModelCarrier carrier_ {};
        GpuModelHelper<RECORDTYPE> helper_;
        unique_ptr<ModelEngine<RECORDTYPE>> modelEngine_ {};
        vector<unique_ptr<ICommandControlAcceptor>> commandControlAcceptors_ {};

    public:
        const string& Reason() const { return reason_; }
        const ModelEngine<RECORDTYPE>& GetModelEngine() const { return *modelEngine_.get(); }
        const ConfigurationRepository& getConfigurationRepository() const { return configuration_; }
        const json& Control() const { return configuration_.Control(); }
        const json& Configuration() const { return configuration_.Configuration(); }
        const json& Monitor() const { return configuration_.Monitor(); }
        const json& Settings() const { return configuration_.Settings(); }
        const microseconds EnginePeriod() const { return modelEngine_->EnginePeriod(); }
        microseconds& EnginePeriod() { return modelEngine_->EnginePeriod(); }
        ModelEngineContext<RECORDTYPE>& Context() const { return modelEngine_->Context(); }
        GpuModelHelper<RECORDTYPE>& Helper() { return helper_; }

    public:
        ModelRunner() :
            helper_(carrier_, configuration_)
        {
        }

        void AddCommandControlAcceptor(unique_ptr<ICommandControlAcceptor> commandControlAcceptor)
        {
            cout << "Adding command and control acceptor " << commandControlAcceptor->Description() << " to runner\n";
            commandControlAcceptors_.push_back(std::move(commandControlAcceptor));
        }

        bool Initialize(int argc, char* argv[])
        {
            cout << "Runner parsing argument\n";
            ParseArgs(argc, argv);

            if (!valid_)
                return false;

            cout << "Runner initializing configuration\n";
            valid_ = configuration_.InitializeConfiguration(controlFile_);

            if (!valid_)
            {
                reason_ = "Unable to initialize configuration from control file " + controlFile_;
                return false;
            }

            for_each(commandControlAcceptors_.begin(), commandControlAcceptors_.end(), 
                [&argc, &argv, this](auto& acceptor)
                { 
                    cout << "Runner initializing command and control acceptor" << acceptor->Description() << "\n"; 
                    if (!acceptor->Initialize(argc, argv))
                    {
                        this->reason_ = "Failed initializing command and control acceptor " + acceptor->Description();
                        this->valid_ = false;
                    }
                });

            return valid_;
        }

        //
        // Ensure the model is created and initialized, then start
        // it running asynchronously.
        //
        bool Run()
        {
            if (!valid_)
            {
                cout << "Failed to run model engine because runner is in an invalid state\n";
                return false;
            }

            if (modelEngine_)
                WaitForQuit();

            return RunModelEngine();
        }

        void RunCommandControl()
        {
            auto quit { false };
            while (!quit)
            {
                for_each(commandControlAcceptors_.begin(), commandControlAcceptors_.end(), [&quit](auto& acceptor){ quit |= acceptor->AcceptAndExecute(); });
            }
        }

        //
        // Start an async process to stop the model engine
        // and return immediately.  To guarantee it has stopped,
        // call WaitForQuit().
        //
        void Quit()
        {
            if (modelEngine_)
                modelEngine_->Quit();
        }

        //
        // Call Quit() and wait until the model engine stops.
        // It is legal to call this after Quit().
        //
        void WaitForQuit()
        {
            if (modelEngine_)
            {
                modelEngine_->WaitForQuit();
                delete(modelEngine_.release());
            }
        }

    private:
        void ParseArgs(int argc, char *argv[])
        {
            static string usage {
                " <control file>\n"
                "  <control file> is the name of the json file "
                "containing the control information (configuration"
                "and monitor) for the test to run.\n"
            };

            if (argc < 2)
            {
                cout << "Usage: " << argv[0] << usage;
                reason_ = "Cannot start with less than 1 parameter";
                valid_ = false;
                return;
            }

            for (auto i = 1; i < argc; i++)
            {
                const auto& arg = argv[i];
                if (arg[0] == '-') continue;
                controlFile_ = arg;
            }

            if (controlFile_.empty())
            {
                cout << "Usage: " << argv[0] << usage;
                reason_ = "Control file not specified";
                valid_ = false;
                return;
            }

            if (controlFile_.length() < 5 || controlFile_.substr(controlFile_.length()-5, controlFile_.length()) != ".json")
                controlFile_ += ".json";

            cout << "Using control file " << controlFile_ << "\n";

            valid_ = true;
        }

        bool RunModelEngine()
        {
            // Create and run the model engine.
            modelEngine_ = make_unique<ModelEngine<RECORDTYPE>>(
                carrier_, 
                configuration_,
                helper_);

            return modelEngine_->Run();
        }
    };
}
