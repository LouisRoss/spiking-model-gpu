#pragma once

#include <iostream>
#include <algorithm>
#include <vector>

#include "nlohmann/json.hpp"

#include "core/ConfigurationRepository.h"
#include "core/Recorder.h"
#include "core/NeuronRecordCommon.h"

#include "GpuModelCarrier.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::cout;
    using std::max_element;
    using std::vector;

    using nlohmann::json;

    using embeddedpenguins::core::neuron::model::NeuronRecordType;

    using embeddedpenguins::core::neuron::model::ConfigurationRepository;
    using embeddedpenguins::core::neuron::model::Recorder;

    template<class RECORDTYPE>
    class GpuModelHelper
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
            LoadOptionalDimensions();
        }

        GpuModelCarrier& Carrier() { return carrier_; }
        const json& Configuration() const { return configuration_.Configuration(); }
        const unsigned int Width() const { return width_; }
        const unsigned int Height() const { return height_; }

        bool AllocateModel(unsigned long int modelSize = 0)
        {
            if (!CreateModel(modelSize))
            {
                cout << "Unable to create model of size " << modelSize << "\n";
                return false;
            }

            return true;
        }

        bool InitializeModel()
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

        unsigned long long int GetIndex(const int row, const int column) const
        {
            return row * width_ + column;
        }

        void WireInput(unsigned long int sourceNodeIndex, int synapticWeight)
        {
            if (!carrier_.Valid)
            {
                cout << "GPU helper cannot wire input with model in invalid state\n";
                return;
            }

            auto& sourceNode = carrier_.NeuronsHost[sourceNodeIndex];

            *(unsigned long*)&carrier_.SynapsesHost[sourceNodeIndex][0].PresynapticNeuron = numeric_limits<unsigned long>::max();
            carrier_.SynapsesHost[sourceNodeIndex][0].Strength = synapticWeight;
            carrier_.SynapsesHost[sourceNodeIndex][0].TickSinceLastSignal = 0;
            carrier_.SynapsesHost[sourceNodeIndex][0].Type = sourceNode.Type == NeuronType::Excitatory ? SynapseType::Excitatory : SynapseType::Inhibitory;

            carrier_.RequiredPostsynapticConnections[sourceNodeIndex]++;
        }

        void Wire(unsigned long int sourceNodeIndex, unsigned long int targetNodeIndex, int synapticWeight)
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

        NeuronType GetNeuronType(const unsigned long int source) const
        {
            return carrier_.NeuronsHost[source].Type;
        }

        short GetNeuronActivation(const unsigned long int source) const
        {
            return carrier_.NeuronsHost[source].Activation;
        }

        void SetExcitatoryNeuronType(const unsigned long int source)
        {
            carrier_.NeuronsHost[source].Type = NeuronType::Excitatory;
        }

        void SetInhibitoryNeuronType(const unsigned long int source)
        {
            carrier_.NeuronsHost[source].Type = NeuronType::Inhibitory;
        }

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

                    RECORDTYPE record(NeuronRecordType::InputSignal, inputIndex, neuron.Activation, 0, ActivationThreshold + 1);
                    recorder.Record(record);
                }
            }

        }

        void RecordRelevantNeurons(Recorder<RECORDTYPE>& recorder)
        {
            if (!carrier_.Valid)
            {
                cout << "GPU helper cannot record neurons with model in invalid state\n";
                return;
            }

            for (auto neuronIndex = 0; neuronIndex < carrier_.ModelSize(); neuronIndex++)
            {
                auto& neuron = carrier_.NeuronsHost[neuronIndex];
                if (IsSpikeTick(neuron.TicksSinceLastSpike))
                {
                    RECORDTYPE record(NeuronRecordType::Spike, neuronIndex, neuron.Activation);
                    recorder.Record(record);
                }
                else if (IsRefractoryTick(neuron.TicksSinceLastSpike))
                {
                    RECORDTYPE record(NeuronRecordType::Refractory, neuronIndex, neuron.Activation);
                    recorder.Record(record);
                }
                else if (IsActiveRecently(neuron.TicksSinceLastSpike))
                {
                    RECORDTYPE record(NeuronRecordType::Decay, neuronIndex, neuron.Activation);
                    recorder.Record(record);
                }

                auto& synapsesForNeuron = carrier_.SynapsesHost[neuronIndex];
                for (auto synapseIndex = 0; synapseIndex < SynapticConnectionsPerNode; synapseIndex++)
                {
                    if (synapsesForNeuron[synapseIndex].TickSinceLastSignal == SynapseSignalTimeMax)
                    {
                        RECORDTYPE record(NeuronRecordType::InputSignal, neuronIndex, neuron.Activation, synapseIndex + 1, synapsesForNeuron[synapseIndex].Strength);
                        recorder.Record(record);
                    }
                }
            }
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
            carrier_.NeuronsDevice = cuda::memory::device::make_unique<NeuronNode[]>(carrier_.Device, carrier_.NeuronCount);
            carrier_.SynapsesDevice = cuda::memory::device::make_unique<NeuronSynapse[][SynapticConnectionsPerNode]>(carrier_.Device, carrier_.NeuronCount);

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
