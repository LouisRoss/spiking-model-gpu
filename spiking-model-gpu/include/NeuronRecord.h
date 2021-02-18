#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <limits>
#include <tuple>

#include "core/NeuronRecordCommon.h"

#include "NeuronCommon.h"
#include "NeuronNode.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::ostringstream;
    using std::vector;
    using std::numeric_limits;
    using std::tuple;

    using embeddedpenguins::core::neuron::model::NeuronRecordType;
    using embeddedpenguins::core::neuron::model::NeuronRecordConnection;
    using embeddedpenguins::core::neuron::model::NeuronRecordSynapse;

    struct NeuronRecord
    {
        NeuronRecordType Type { NeuronRecordType::Refractory };
        unsigned long long int NeuronIndex { };
        int Activation { 0 };
        vector<NeuronRecordConnection> Connections { };
        NeuronRecordSynapse Synapse { };

        NeuronRecord(NeuronRecordType type, unsigned long long int neuronIndex, int activation, unsigned int synapseIndex = 0, const int synapseStrength = 0) :
            Type(type),
            NeuronIndex(neuronIndex),
            Activation(activation)
        {
            switch (type)
            {
                case NeuronRecordType::InputSignal:
                    Synapse.SynapseIndex = synapseIndex;
                    Synapse.Strength = synapseStrength;
                    break;

                case NeuronRecordType::Decay:
                    break;

                case NeuronRecordType::Spike:
#ifdef ALLCONNECTIONS
                    for (auto& connection : neuronNode.PostsynapticConnections)
                    {
                        if (connection.IsUsed) 
                            Connections.push_back(NeuronRecordConnection { connection.PostsynapticNeuron, connection.Synapse });
                    }
#endif
                    break;

                case NeuronRecordType::Refractory:
                    break;
                    
                default:
                    break;
            }
        }

        static const string Header()
        {
            ostringstream header;
            header << "Neuron-Event-Type,Neuron-Index,Neuron-Activation,Synapse-Index,Synapse-Strength";
#ifdef ALLCONNECTIONS
            for (auto i = 0; i < PresynapticConnectionsPerNode; i++)
                header << ",Synapse" << i << "-Signaled" << ",Synapse-" << i << "-Strength";
#endif
            return header.str();
        }

        const string Format()
        {
            ostringstream row;
            row << (int)Type << "," << NeuronIndex << "," << Activation << ",";
            switch (Type)
            {
                case NeuronRecordType::InputSignal:
                    row << Synapse.SynapseIndex << ',' << Synapse.Strength;
                    break;

                case NeuronRecordType::Decay:
                    row << "N/A,N/A";
                    break;

                case NeuronRecordType::Spike:
                    row << "N/A,N/A";
#ifdef ALLCONNECTIONS
                    // TODO
#endif
                    break;

                case NeuronRecordType::Refractory:
                    row << "N/A,N/A";
                    break;
                    
                default:
                    row << "N/A,N/A";
                    break;
            }

            return row.str();
        }
    };
}
