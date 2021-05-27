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
#include "IQueryHandler.h"
#include "CommandControlHandler.h"

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
    using embeddedpenguins::core::neuron::model::IQueryHandler;
    using embeddedpenguins::core::neuron::model::CommandControlHandler;

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
        unique_ptr<IQueryHandler> queryHandler_ {};

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
        //
        // Usage should follow this sequence:
        // 1. Construct a new instance.
        // 2. Add any required command & control acceptors with AddCommandControlAcceptor().
        // 3. Call Initialize() (one time only) with the command line arguments.
        // 4. Give the the C&C acceptors run time on the main thread by calling RunCommandControl().
        //    This will return only after the model terminates for the final time.
        // Call sequences of the following to control as needed (typically from within a C&C acceptor)
        //    * RunWithNewModel()
        //    * RunWithExistingModel()
        //    * Pause()
        //    * Continue()
        //    * Quit()
        //    * WaitForQuit()
        //
        ModelRunner() :
            helper_(carrier_, configuration_)
        {
            //queryHandler_ = std::move(make_unique<CommandControlHandler>());
        }

        //
        // Multiple command & control acceptors (which must implement ICommandControlAcceptor)
        // may be added to the model runner, and each will be given time to run on the main thread.
        //
        void AddCommandControlAcceptor(unique_ptr<ICommandControlAcceptor> commandControlAcceptor)
        {
            cout << "Adding command and control acceptor " << commandControlAcceptor->Description() << " to runner\n";
            commandControlAcceptors_.push_back(std::move(commandControlAcceptor));
        }

        //
        // After all C&C acceptors are added, initialize with the command line argments.
        // If a control file was part of the command line, the model will be automatically
        // run with that control file.
        //
        bool Initialize(int argc, char* argv[])
        {
            cout << "Runner parsing argument\n";
            ParseArgs(argc, argv);

            if (!valid_)
                return false;

            PrepareControlFile();

            if (!valid_)
                return false;

            InitializeConfiguration();

            return valid_;
        }

        //
        // All C&C acceptors are given some run time on the main thread by calling here.
        // This will not return until one C&C acceptor sees its quit command.
        //
        void RunCommandControl()
        {
            auto quit { false };
            while (!quit)
            {
                for_each(
                    commandControlAcceptors_.begin(), 
                    commandControlAcceptors_.end(), 
                    [this, &quit](auto& acceptor)
                    {
                        quit |= acceptor->AcceptAndExecute(
                        [this](const string& command)
                        {
                            this->HandleCommand(command);
                        });
                    });
            }

            // Make sure we are not paused before stopping.
            Continue();
        }

        bool RunWithNewModel(const string& controlFile)
        {
            controlFile_ = controlFile;

            return RunWithExistingModel();
        }


        bool RunWithExistingModel()
        {
            PrepareControlFile();

            if (!valid_)
                return false;

            InitializeConfiguration();

            if (!valid_)
                return false;

            return Run();
        }

        bool Pause()
        {
            if (modelEngine_)
            {
                modelEngine_->Pause();
                return true;
            }

            return false;
        }

        bool Continue()
        {
            if (modelEngine_)
            {
                modelEngine_->Continue();
                return true;
            }

            return false;
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
        void InitializeConfiguration()
        {
            cout << "Runner initializing configuration\n";
            valid_ = configuration_.InitializeConfiguration(controlFile_);

            if (!valid_)
            {
                reason_ = "Unable to initialize configuration from control file " + controlFile_;
                return;
            }

            if (!helper_.AllocateModel())
            {
                cout << "ModelRunner.InitializeConfiguration failed at helper_.AllocateModel()\n";
                modelEngine_->Context().EngineInitializeFailed = true;
                reason_ = "Helper failed to initialize configuration or allocate model memory";
                valid_ = false;
                return;
            }

            for_each(commandControlAcceptors_.begin(), commandControlAcceptors_.end(), 
                [this](auto& acceptor)
                { 
                    cout << "Runner initializing command and control acceptor" << acceptor->Description() << "\n"; 
                    if (!acceptor->Initialize())
                    {
                        this->reason_ = "Failed initializing command and control acceptor " + acceptor->Description();
                        this->valid_ = false;
                    }
                });
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

            WaitForQuit();

            if (!modelEngine_)
                if (!InitializeModelEngine())
                    return false;

            return RunModelEngine();
        }

        void HandleCommand(const string& command)
        {
            cout << "Handling command '" << command << "'\n";
        }

        void ParseArgs(int argc, char *argv[])
        {
            static string usage {
                " <control file>\n"
                "  <control file> is the name of the json file "
                "containing the control information (configuration"
                "and monitor) for the test to run.\n"
            };

            for (auto i = 1; i < argc; i++)
            {
                const auto& arg = argv[i];
                if (arg[0] == '-') continue;
                controlFile_ = arg;
            }

            valid_ = true;

            for_each(commandControlAcceptors_.begin(), commandControlAcceptors_.end(), 
                [&argc, &argv, this](auto& acceptor)
                { 
                    cout << "Runner parsing arguments for command and control acceptor" << acceptor->Description() << "\n"; 
                    if (!acceptor->ParseArguments(argc, argv))
                    {
                        this->reason_ = "Failed initializing command and control acceptor " + acceptor->Description();
                        this->valid_ = false;
                    }
                });
        }

        bool PrepareControlFile()
        {
            if (controlFile_.empty())
            {
                valid_ = false;
                cout << "No control file given, not running any model\n";
            }
            else
            {
                if (controlFile_.length() < 5 || controlFile_.substr(controlFile_.length()-5, controlFile_.length()) != ".json")
                    controlFile_ += ".json";

                cout << "Using control file " << controlFile_ << "\n";
            }

            return valid_;
        }

        bool InitializeModelEngine()
        {
            // Ensure no model engine, stop and delete first if needed.
            WaitForQuit();

            // Create the model engine.
            modelEngine_ = make_unique<ModelEngine<RECORDTYPE>>(
                carrier_, 
                configuration_,
                helper_);

            return modelEngine_->Initialize();
        }

        bool RunModelEngine()
        {
            if (!modelEngine_)
                return false;

            // Run the model engine.
            return modelEngine_->Run();
        }
    };
}
