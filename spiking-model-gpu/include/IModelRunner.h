#pragma once

#include <string>
#include <memory>
#include <chrono>

#include "nlohmann/json.hpp"

#include "CommandControlAcceptors/ICommandControlAcceptor.h"
#include "ConfigurationRepository.h"
#include "IModelHelper.h"

namespace embeddedpenguins::core::neuron::model
{
    using std::string;
    using std::unique_ptr;
    using std::chrono::microseconds;

    using nlohmann::json;

    class IModelRunner
    {
    public:
        virtual ~IModelRunner() = default;

        virtual const string& Reason() const = 0;
        virtual ConfigurationRepository& getConfigurationRepository() = 0;
        virtual json& Control() = 0;
        virtual json& Configuration() = 0;
        virtual json& Monitor() = 0;
        virtual json& Settings() = 0;
        virtual const unsigned long int ModelSize() const = 0;
        virtual const microseconds EnginePeriod() const = 0;
        virtual const long long int GetTotalWork() const = 0;
        virtual const long long int GetIterations() const = 0;
        virtual microseconds& EnginePeriod() = 0;
        virtual IModelHelper* Helper() const = 0;

        virtual bool Initialize(int argc, char* argv[]) = 0;
        virtual void RunCommandControl() = 0;
        virtual bool RunWithNewModel(const string& controlFile) = 0;
        virtual bool RunWithExistingModel() = 0;
        virtual bool Pause() = 0;
        virtual bool Continue() = 0;
        virtual void Quit() = 0;
        virtual void WaitForQuit() = 0;

        virtual json RenderStatus() = 0;
        virtual json RenderDynamicStatus() = 0;
        virtual json RenderRunMeasurements() = 0;
        virtual bool SetValue(const json& controlValues) = 0;
        virtual bool DeployModel(const string& modelName, const string& deploymentName, const string& engineName) = 0;
    };
}
