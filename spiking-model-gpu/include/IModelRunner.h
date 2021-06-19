#pragma once

#include <string>
#include <memory>
#include <chrono>

#include "nlohmann/json.hpp"

#include "ICommandControlAcceptor.h"
#include "ConfigurationRepository.h"

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
        virtual const string& ControlFile() const = 0;
        virtual const ConfigurationRepository& getConfigurationRepository() const = 0;
        virtual const json& Control() const = 0;
        virtual const json& Configuration() const = 0;
        virtual const json& Monitor() const = 0;
        virtual const json& Settings() const = 0;
        virtual const microseconds EnginePeriod() const = 0;
        virtual microseconds& EnginePeriod() = 0;

        virtual void AddCommandControlAcceptor(unique_ptr<ICommandControlAcceptor> commandControlAcceptor) = 0;
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
        virtual bool SetValue(const json& controlValues) = 0;
    };
}
