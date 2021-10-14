#pragma once

#include <string>
#include <algorithm>
#include <tuple>

namespace embeddedpenguins::core::neuron::model::initializerprotocol
{
    using std::string;
    using std::copy;
    using std::tuple;

    using LengthFieldType = unsigned int;
    using CommandFieldType = unsigned int;

    enum class PackageInitializerCommand : CommandFieldType
    {
        GetModelDescriptor = 0,
        GetModelExpansion = 1
    };

    struct PackageInitializerEnvelope
    {
        LengthFieldType PacketSize;
    };

    struct ModelDescriptorRequest : public PackageInitializerEnvelope
    {
        PackageInitializerCommand Command { PackageInitializerCommand::GetModelDescriptor };
        char ModelName[80];

        ModelDescriptorRequest() { PacketSize = sizeof(ModelDescriptorRequest) - sizeof(PackageInitializerEnvelope); }
        ModelDescriptorRequest(const string& modelName) : ModelDescriptorRequest()
        {
            string name { modelName };
            name.resize(sizeof(ModelName));
            copy(name.c_str(), name.c_str() + sizeof(ModelName), ModelName);
        }
    };

    struct ModelExpansionRequest : public PackageInitializerEnvelope
    {
        PackageInitializerCommand Command { PackageInitializerCommand::GetModelExpansion };
        unsigned int Sequence { 0 };
        char ModelName[80];

        ModelExpansionRequest() { PacketSize = sizeof(ModelExpansionRequest) - sizeof(PackageInitializerEnvelope); }
        ModelExpansionRequest(const string& modelName, unsigned int sequence) : ModelExpansionRequest()
        {
            Sequence = sequence;

            string name { modelName };
            name.resize(sizeof(ModelName));
            copy(name.c_str(), name.c_str() + sizeof(ModelName), ModelName);
        }
    };

    struct ValidateSize { };

    struct ModelDescriptorResponse : public ValidateSize
    {
        unsigned int NeuronCount { 0 };
        unsigned int ExpansionCount { 0 };
    };

    struct ModelExpansionResponse
    {
        using ConnectionType = unsigned[3];

        unsigned int StartingNeuronOffset { 0 };
        unsigned int NeuronCount { 0 };
        unsigned int ConnectionCount { 0 };

        // Immediately following this struct in memory should be an array
        //ConnectionType Connection[ConnectionCount];
        ConnectionType* GetConnections() { return (ConnectionType*)(this + 1); }
    };
}
