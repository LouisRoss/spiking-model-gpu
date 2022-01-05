#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <tuple>
#include <memory>

#include "nlohmann/json.hpp"

#include "ConfigurationRepository.h"
#include "NeuronRecordCommon.h"
#include "IModelHelper.h"

#include "GpuModelCarrier.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::cout;
    using std::string;
    using std::max_element;
    using std::vector;
    using std::tuple;
    using std::unique_ptr;

    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::IModelHelper;
    using embeddedpenguins::core::neuron::model::NeuronRecordType;

    using embeddedpenguins::core::neuron::model::ConfigurationRepository;

    class GpuModelHelper : public IModelHelper
    {
        GpuModelCarrier& carrier_;
        const ConfigurationRepository& configuration_;

        unsigned int width_ { 50 };
        unsigned int height_ { 25 };
        unsigned long long int maxIndex_ { };

    public:
        GpuModelHelper(GpuModelCarrier& carrier, const ConfigurationRepository& configuration) :
            carrier_(carrier),
            configuration_(configuration)
        {
        }

        // IModelHelper implementation
        virtual const json& Configuration() const override { return configuration_.Configuration(); }
        virtual const json& StackConfiguration() const override { return configuration_.StackConfiguration(); }
        virtual const string& ModelName() const override { return configuration_.ModelName(); }
        virtual const string& DeploymentName() const override { return configuration_.DeploymentName(); }
        virtual const string& EngineName() const override { return configuration_.EngineName(); }
        virtual const unsigned int Width() const override { return width_; }
        virtual const unsigned int Height() const override { return height_; }

        //
        // Unpack needed parameters from the configuration and allocate
        // both CPU and GPU memory necessary to contain the model.
        //
        virtual bool AllocateModel(unsigned long int modelSize = 0) override
        {
            LoadOptionalDimensions();

            if (!CreateModel(modelSize))
            {
                cout << "Unable to create model of size " << modelSize << "\n";
                return false;
            }

            return true;
        }

        virtual bool InitializeModel() override
        {
            if (!carrier_.Valid)
            {
                cout << "GPU helper cannot initialize model in invalid state\n";
                return false;
            }

            cout << "Initializing Neurons and Synapses..." << std::flush;
            for (auto neuronId = 0; neuronId < carrier_.ModelSize(); neuronId++)
            {
                for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
                {
                    *(unsigned long*)&carrier_.SynapsesHost[neuronId][synapseId].PresynapticNeuron = numeric_limits<unsigned long>::max();
                    carrier_.SynapsesHost[neuronId][synapseId].Strength = 0;
                    carrier_.SynapsesHost[neuronId][synapseId].TickSinceLastSignal = 0;
                    carrier_.SynapsesHost[neuronId][synapseId].Type = SynapseType::Excitatory;
                }

                carrier_.NeuronsHost[neuronId].Type = NeuronType::Excitatory;
                carrier_.NeuronsHost[neuronId].NextTickSpike = false;
                carrier_.NeuronsHost[neuronId].Activation = 0;
                carrier_.NeuronsHost[neuronId].TicksSinceLastSpike = 0;
            }
            cout << "done\n";

            return true;
        }

        virtual unsigned long long int GetIndex(const int row, const int column) const override
        {
            return row * width_ + column;
        }

        virtual unsigned long int GetNeuronTicksSinceLastSpike(const unsigned long int source) const override
        {
            return carrier_.NeuronsHost[source].TicksSinceLastSpike;
        }

        virtual bool IsSynapseUsed(const unsigned long int neuronIndex, const unsigned int synapseId) const override
        {
            auto& synapsesForNeuron = carrier_.SynapsesHost[neuronIndex];
            return ((unsigned long int*)synapsesForNeuron[synapseId].PresynapticNeuron) != 0;
        }

        virtual int GetSynapticStrength(const unsigned long int neuronIndex, const unsigned int synapseId) const override
        {
            auto& synapsesForNeuron = carrier_.SynapsesHost[neuronIndex];
            return synapsesForNeuron[synapseId].Strength;
        }

        virtual unsigned long int GetPresynapticNeuron(const unsigned long int neuronIndex, const unsigned int synapseId) const override
        {
            // Only available in GPU, fake it here.
            return 0;
        }

        virtual void WireInput(unsigned long int sourceNodeIndex, int synapticWeight) override
        {
            if (!carrier_.Valid)
            {
                //cout << "GPU helper cannot wire input with model in invalid state\n";
                return;
            }

            auto& sourceNode = carrier_.NeuronsHost[sourceNodeIndex];

            *(unsigned long*)&carrier_.SynapsesHost[sourceNodeIndex][0].PresynapticNeuron = numeric_limits<unsigned long>::max();
            carrier_.SynapsesHost[sourceNodeIndex][0].Strength = synapticWeight;
            carrier_.SynapsesHost[sourceNodeIndex][0].TickSinceLastSignal = 0;
            carrier_.SynapsesHost[sourceNodeIndex][0].Type = sourceNode.Type == NeuronType::Excitatory ? SynapseType::Excitatory : SynapseType::Inhibitory;

            carrier_.RequiredPostsynapticConnections[sourceNodeIndex]++;
        }

        virtual void Wire(unsigned long int sourceNodeIndex, unsigned long int targetNodeIndex, int synapticWeight) override
        {
            if (!carrier_.Valid)
            {
                cout << "GPU helper cannot wire with model in invalid state\n";
                return;
            }

            auto& sourceNode = carrier_.NeuronsHost[sourceNodeIndex];
            auto targetSynapseIndex = FindNextUnusedTargetSynapse(targetNodeIndex);

            if (targetSynapseIndex != -1)
            {
                *(unsigned long*)&carrier_.SynapsesHost[targetNodeIndex][targetSynapseIndex].PresynapticNeuron = sourceNodeIndex;
                carrier_.SynapsesHost[targetNodeIndex][targetSynapseIndex].Strength = synapticWeight;
                carrier_.SynapsesHost[targetNodeIndex][targetSynapseIndex].TickSinceLastSignal = 0;
                carrier_.SynapsesHost[targetNodeIndex][targetSynapseIndex].Type = sourceNode.Type == NeuronType::Excitatory ? SynapseType::Excitatory : SynapseType::Inhibitory;
            }

            carrier_.RequiredPostsynapticConnections[targetNodeIndex]++;
        }

        virtual NeuronType GetNeuronType(const unsigned long int source) const override
        {
            return carrier_.NeuronsHost[source].Type;
        }

        virtual short GetNeuronActivation(const unsigned long int source) const override
        {
            return carrier_.NeuronsHost[source].Activation;
        }

        virtual void SetExcitatoryNeuronType(const unsigned long int source) override
        {
            carrier_.NeuronsHost[source].Type = NeuronType::Excitatory;
        }

        virtual void SetInhibitoryNeuronType(const unsigned long int source) override
        {
            carrier_.NeuronsHost[source].Type = NeuronType::Inhibitory;
        }

#ifdef STREAM_CPU
        void SpikeInputNeurons(const vector<unsigned long long>& streamedInput, Recorder<RECORDTYPE>& recorder)
        {
            if (!carrier_.Valid)
            {
                cout << "GPU helper cannot spike inputs with model in invalid state\n";
                return;
            }

            for (auto inputIndex : streamedInput)
            {
                if (inputIndex < carrier_.ModelSize())
                {
                    auto& neuron = carrier_.NeuronsHost[inputIndex];

                    neuron.NextTickSpike = true;
                    neuron.Activation = ActivationThreshold + 1;
                    //neuron.TicksSinceLastSpike = RecoveryTimeMax;
                    //cout << "SpikeInputs: Neuron " << inputIndex << " set NextTickSpike true, Activation to " << ActivationThreshold + 1 << ", and TickSinceLastSpike to " << RecoveryTimeMax << "\n";

                    RECORDTYPE record(NeuronRecordType::InputSignal, inputIndex, neuron.Activation, 0, ActivationThreshold + 1);
                    recorder.Record(record);
                }
            }

        }
#endif

        virtual vector<tuple<unsigned long long, short int, NeuronRecordType>> CollectRelevantNeurons() override
        {
            vector<tuple<unsigned long long, short int, NeuronRecordType>> relevantNeurons;

            if (!carrier_.Valid)
            {
                cout << "GPU helper cannot record neurons with model in invalid state\n";
            }
            else
            {
                for (auto neuronIndex = 0; neuronIndex < carrier_.ModelSize(); neuronIndex++)
                {
                    auto& neuron = carrier_.NeuronsHost[neuronIndex];

                    if (IsSpikeTick(neuron.TicksSinceLastSpike))
                    {
                        relevantNeurons.push_back(std::make_tuple(neuronIndex, neuron.Activation, NeuronRecordType::Spike));
                    }
                    else if (IsRefractoryTick(neuron.TicksSinceLastSpike))
                    {
                        relevantNeurons.push_back(std::make_tuple(neuronIndex, neuron.Activation, NeuronRecordType::Refractory));
                    }
                    else if (IsActiveRecently(neuron.TicksSinceLastSpike))
                    {
                        relevantNeurons.push_back(std::make_tuple(neuronIndex, neuron.Activation, NeuronRecordType::Decay));
                    }
                }
            }

            return relevantNeurons;
        }

        unsigned long int FindRequiredSynapseCounts()
        {
            const auto& requiredPostsynapticConnection = carrier_.RequiredPostsynapticConnections;
            auto requiredSynapseCount = *max_element(&requiredPostsynapticConnection[0], &requiredPostsynapticConnection[carrier_.ModelSize()]);

            if (requiredSynapseCount > SynapticConnectionsPerNode)
                cout << "Insufficient postsynaptic space allocated: need " << requiredSynapseCount << " have " << SynapticConnectionsPerNode << '\n';

            return requiredSynapseCount;
        }

    private:
        //
        // Allocate memory for the model.
        // NOTE: Only to be called from the main process, not a load library.
        //
        bool CreateModel(unsigned long int modelSize)
        {
            auto size = modelSize;
            if (size == 0)
            {
                const json& modelJson = configuration_.Configuration()["Model"];
                if (!modelJson.is_null())
                {
                    const json& modelSizeJson = modelJson["ModelSize"];
                    if (modelSizeJson.is_number_unsigned())
                        size = modelSizeJson.get<unsigned int>();
                }
            }

            if (size == 0)
            {
                cout << "No model size configured or supplied, initializer cannot create model\n";
                carrier_.Valid = false;
                return false;
            }

            carrier_.NeuronCount = size;

            carrier_.RequiredPostsynapticConnections = std::make_unique<unsigned long[]>(carrier_.NeuronCount);
            carrier_.NeuronsHost = std::make_unique<NeuronNode[]>(carrier_.NeuronCount);
            carrier_.SynapsesHost = std::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(carrier_.NeuronCount);
            carrier_.InputSignalsHost = std::make_unique<unsigned long[]>(InputBufferSize);
            carrier_.NeuronsDevice = cuda::memory::device::make_unique<NeuronNode[]>(carrier_.Device, carrier_.NeuronCount);
            carrier_.SynapsesDevice = cuda::memory::device::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(carrier_.Device, carrier_.NeuronCount);
            carrier_.InputSignalsDevice = cuda::memory::device::make_unique<unsigned long long[]>(carrier_.Device, InputBufferSize);

            carrier_.Valid = true;
            return true;
        }

        void LoadOptionalDimensions()
        {
            // Override the dimension defaults if configured.
            const json& configuration = Configuration();
            auto& modelSection = configuration["Model"];
            if (!modelSection.is_null() && modelSection.contains("Dimensions"))
            {
                auto dimensionElement = modelSection["Dimensions"];
                if (dimensionElement.is_array())
                {
                    auto dimensionArray = dimensionElement.get<std::vector<int>>();
                    width_ = dimensionArray[0];
                    height_ = dimensionArray[1];
                }
            }

            maxIndex_ = width_ * height_;
        }

        int FindNextUnusedTargetSynapse(unsigned long int targetNodeIndex) const
        {
            for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
            {
                if (*(unsigned long*)&carrier_.SynapsesHost[targetNodeIndex][synapseId].PresynapticNeuron == numeric_limits<unsigned long>::max())
                    return synapseId;
            }

            return -1;
        }
    };
}
