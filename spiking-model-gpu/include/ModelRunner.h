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

#include "IModelRunner.h"
#include "IModelHelper.h"
#include "GpuModelCarrier.h"
//#include "GpuModelHelper.h"
#include "GpuPackageHelper.h"
#include "ModelEngine.h"
#include "IQueryHandler.h"
#include "CommandControlAcceptors/ICommandControlAcceptor.h"
#include "CommandControlAcceptors/ICommandControlAcceptor.h"
#include "CommandControlAcceptors/GpuModelUi.h"
#include "CommandControlAcceptors/CommandControlBasicUi.h"
#include "CommandControlAcceptors/QueryResponseListenSocket.h"
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

    using embeddedpenguins::core::neuron::model::IModelRunner;
    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::ModelInitializerProxy;
    using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
    using embeddedpenguins::core::neuron::model::IQueryHandler;
    using embeddedpenguins::core::neuron::model::CommandControlHandler;
    using embeddedpenguins::core::neuron::model::CommandControlBasicUi;
    using embeddedpenguins::core::neuron::model::QueryResponseListenSocket;

    namespace runner
    {
        class ModelRunnerInitializer
        {
            IModelRunner& runner_;
            ConfigurationRepository& configuration_;
            bool& valid_;
            vector<unique_ptr<ICommandControlAcceptor>>& commandControlAcceptors_;

        public:
            ModelRunnerInitializer(IModelRunner& runner, ConfigurationRepository& configuration, bool& valid, vector<unique_ptr<ICommandControlAcceptor>>& commandControlAcceptors) :
                runner_(runner),
                configuration_(configuration),
                valid_(valid),
                commandControlAcceptors_(commandControlAcceptors)
            {
            }

            bool InitializeCommandControlAccpetors(int argc, char* argv[])
            {
                cout << "\n****Model runner initializing command and control acceptors\n";

                if (!configuration_.Control().contains("Execution"))
                {
                    cout << "Initialization of Command/Control acceptors failed: no 'Execution' element in control file\n";
                    valid_ = false;
                    return valid_;
                }
                
                const json& executionJson = configuration_.Control()["Execution"];
                if (!executionJson.contains("CommandControlAcceptors"))
                {
                    cout << "Initialization of Command/Control acceptors failed: 'Execution' element in control file contains no 'CommandControlAcceptors' element\n";
                    valid_ = false;
                    return valid_;
                }

                int ccCount { 0 };
                auto oneCC { false };
                const json& commandControlAcceptorsJson = executionJson["CommandControlAcceptors"];
                if (commandControlAcceptorsJson.is_array())
                {
                    for (auto& [key, commandControlAcceptorJson] : commandControlAcceptorsJson.items())
                    {
                        ccCount++;
                        if (InitializeCommandControlAccpetor(commandControlAcceptorJson, argc, argv))
                            oneCC = true;
                    }
                }

                valid_ = oneCC;
                cout << "****Model runner initialized " << ccCount << " command and control acceptors " << (valid_ ? "with at least one success" : "but none were successful") << "\n\n";
                return valid_;
            }

        private:
            bool InitializeCommandControlAccpetor(const json& commandControlAcceptorJson, int argc, char* argv[])
            {
                if (commandControlAcceptorJson.is_object())
                {
                    if (commandControlAcceptorJson.contains("AcceptorType"))
                    {
                        string acceptorType = commandControlAcceptorJson["AcceptorType"].get<string>();
                        std::transform(acceptorType.begin(), acceptorType.end(), acceptorType.begin(), [](unsigned char c){ return std::tolower(c); });
                        if (acceptorType == "commandcontrolbasicui")
                        {
                            return AddCommandControlAcceptor(std::move(make_unique<CommandControlBasicUi>(runner_)), argc, argv);
                        }
                        else if (acceptorType == "commandcontrolconsoleui")
                        {
                            return AddCommandControlAcceptor(std::move(make_unique<GpuModelUi>(runner_)), argc, argv);
                        }
                        else if (acceptorType == "commandcontrolsocket")
                        {
                            string connectionString { };
                            if (commandControlAcceptorJson.contains("ConnectionString"))
                            {
                                connectionString = commandControlAcceptorJson["ConnectionString"].get<string>();
                            }
                            auto [host, port] = ParseConnectionString(connectionString);

                            return AddCommandControlAcceptor(std::move(make_unique<QueryResponseListenSocket>(host, port)), argc, argv);
                        }
                    }
                }

                return false;
            }

            //
            // Multiple command & control acceptors (which must implement ICommandControlAcceptor)
            // may be added to the model runner, and each will be given time to run on the main thread.
            //
            bool AddCommandControlAcceptor(unique_ptr<ICommandControlAcceptor> commandControlAcceptor, int argc, char* argv[])
            {
                cout << "Parsing arguments for command and control acceptor" << commandControlAcceptor->Description() << "\n"; 
                if (!commandControlAcceptor->ParseArguments(argc, argv))
                {
                    cout << "Failed parsing arguments for command and control acceptor " << commandControlAcceptor->Description() << "\n";
                    valid_ = false;
                    return valid_;
                }

                cout << "Runner initializing command and control acceptor" << commandControlAcceptor->Description() << "\n"; 
                if (!commandControlAcceptor->Initialize())
                {
                    cout << "Failed initializing command and control acceptor " << commandControlAcceptor->Description() << "\n";
                    valid_ = false;
                    return valid_;
                }

                cout << "Adding command and control acceptor " << commandControlAcceptor->Description() << " to runner\n";
                commandControlAcceptors_.push_back(std::move(commandControlAcceptor));
                return valid_;
            }

            tuple<string, string> ParseConnectionString(const string& connectionString)
            {
                string host {"0.0.0.0"};
                string port {"8000"};

                auto colonPos = connectionString.find(":");
                if (colonPos != string::npos)
                {
                    auto tempHost = connectionString.substr(0, colonPos);
                    if (!tempHost.empty()) host = tempHost;
                    auto tempPort = connectionString.substr(colonPos + 1);
                    if (!tempPort.empty()) port = tempPort;
                }

                return {host, port};
            }
        };
    }

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
    class ModelRunner : public IModelRunner
    {
        bool valid_ { false };
        string reason_ {};
        string controlFile_ {};

        ConfigurationRepository configuration_ {};
        GpuModelCarrier carrier_ {};
        unique_ptr<IModelHelper> helper_;
        unique_ptr<ModelEngine<RECORDTYPE>> modelEngine_ {};
        vector<unique_ptr<ICommandControlAcceptor>> commandControlAcceptors_ {};
        unique_ptr<IQueryHandler> queryHandler_ {};

    public:
        virtual const string& Reason() const override { return reason_; }
        const ModelEngine<RECORDTYPE>& GetModelEngine() const { return *modelEngine_.get(); }
        virtual ConfigurationRepository& getConfigurationRepository() override { return configuration_; }
        virtual const json& Control() const override { return configuration_.Control(); }
        virtual const json& Configuration() const override { return configuration_.Configuration(); }
        virtual const json& Monitor() const override { return configuration_.Monitor(); }
        virtual const json& Settings() const override { return configuration_.Settings(); }
        virtual const unsigned long int ModelSize() const override { return carrier_.ModelSize(); }
        virtual const microseconds EnginePeriod() const override { return modelEngine_->EnginePeriod(); }
        virtual microseconds& EnginePeriod() override { return modelEngine_->EnginePeriod(); }
        virtual const long long int GetTotalWork() const override { return modelEngine_->GetTotalWork(); }
        virtual const long long int GetIterations() const override { return modelEngine_->GetIterations(); }
        virtual IModelHelper* Helper() const override { return helper_.get(); }
        ModelEngineContext& Context() const { return modelEngine_->Context(); }
        GpuModelCarrier& Carrier() { return carrier_; }

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
            helper_(std::move(GenerateModelHelper()))
        {
            cout << "\n***Creating new model runner with default query handler\n";
            queryHandler_ = std::move(make_unique<CommandControlHandler<RECORDTYPE>>(*this));
        }

        ModelRunner(unique_ptr<IQueryHandler> queryHandler) :
            helper_(std::move(GenerateModelHelper())),
            queryHandler_(std::move(queryHandler))
        {
            cout << "\n***Creating new model runner with specified query handler\n";
        }

        unique_ptr<IModelHelper> GenerateModelHelper()
        {
            // TODO - some configuration allows us to select the right helper.
            //return std::move(make_unique<GpuModelHelper>(carrier_, configuration_));
            return std::move(make_unique<GpuPackageHelper>(carrier_, configuration_));
        }

        //
        // After all C&C acceptors are added, initialize with the command line argments.
        // If a control file was part of the command line, the model will be automatically
        // run with that control file.
        //
        virtual bool Initialize(int argc, char* argv[]) override
        {
            cout << "\n***Model runner initializing\n";
            ParseArgs(argc, argv);

            if (!valid_)
                return false;

            if (PrepareControlFile())
                InitializeConfiguration();

            if (valid_)
            {
                runner::ModelRunnerInitializer initializer(*this, configuration_, valid_, commandControlAcceptors_);
                initializer.InitializeCommandControlAccpetors(argc, argv);
            }

            cout << "***Model runner initialized into " << (valid_ ? "valid" : "invalid") << " state\n";
            return valid_;
        }

        //
        // All C&C acceptors are given some run time on the main thread by calling here.
        // This will not return until one C&C acceptor sees its quit command.
        //
        virtual void RunCommandControl() override
        {
            cout << "\n***Model runner runnning comand and control acceptors\n";

            auto quit { false };
            while (!quit)
            {
                for_each(
                    commandControlAcceptors_.begin(), 
                    commandControlAcceptors_.end(), 
                    [this, &quit](auto& acceptor)
                    {
                        quit |= acceptor->AcceptAndExecute(this->queryHandler_);
                    }
                );
            }

            cout << "\n***Model runner quitting\n";

            // Make sure we are not paused before stopping.
            Continue();
        }

        virtual bool RunWithNewModel(const string& controlFile) override
        {
            controlFile_ = controlFile;

            return RunWithExistingModel();
        }


        virtual bool RunWithExistingModel() override
        {
            if (PrepareControlFile())
            {
                InitializeConfiguration();

                if (!valid_)
                    return false;

                return Run();
            }

            return true;
        }

        //
        // Set the model engine into the paused state.
        //
        virtual bool Pause() override
        {
            if (modelEngine_)
            {
                modelEngine_->Pause();
                return true;
            }

            return false;
        }

        //
        // Ensure the model engine is not in the paused state.
        //
        virtual bool Continue() override
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
        virtual void Quit() override
        {
            if (modelEngine_)
                modelEngine_->Quit();
        }

        //
        // Call Quit() and wait until the model engine stops.
        // It is legal to call this after Quit().
        //
        virtual void WaitForQuit() override
        {
            if (modelEngine_)
            {
                modelEngine_->WaitForQuit();
                delete(modelEngine_.release());
            }
        }

        //
        // Render the current model engine status into a single JSON object.
        //
        virtual json RenderStatus() override
        {
            if (modelEngine_)
                return modelEngine_->Context().Render();

            return json {};
        }

        //
        // Render only the dynamic portion of the status into a JSON object.
        //
        virtual json RenderDynamicStatus() override
        {
            if (modelEngine_)
                return modelEngine_->Context().RenderDynamic();

            return json {};
        }

        //
        // Accept a JSON object with a collection of name/value pairs
        // and set the value to each named parameter.
        //
        virtual bool SetValue(const json& controlValues) override
        {
            if (modelEngine_)
                return modelEngine_->Context().SetValue(controlValues);

            return false;
        }

        virtual bool DeployModel(const string& modelName, const string& deploymentName, const string& engineName) override
        {
            configuration_.ModelName(modelName);
            configuration_.DeploymentName(deploymentName);
            configuration_.EngineName(engineName);
            
            if (!valid_)
                return false;

            cout << "Runner deploying with model: " << modelName << " deployment: " << deploymentName << " engine: " << engineName << "\n";
            return Run();
        }

    private:
        void InitializeConfiguration()
        {
            cout << "\n***Model runner initializing configuration from control fle " << controlFile_ << "\n";
            valid_ = configuration_.InitializeConfiguration(controlFile_);
            cout << "*** Model runner done initializing configuration\n";
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

        void ParseArgs(int argc, char *argv[])
        {
            for (auto i = 1; i < argc; i++)
            {
                const auto& arg = argv[i];
                if (arg[0] == '-') continue;
                controlFile_ = arg;
            }


            valid_ = true;
        }

        bool PrepareControlFile()
        {
            if (controlFile_.empty())
            {
                cout << "No control file given, using default\n";
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
                helper_.get());

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
