#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <limits>

#include "nlohmann/json.hpp"

#include "ConfigurationRepository.h"
#include "IModelInitializer.h"

namespace embeddedpenguins::core::neuron::model
{
    using std::cout;
    using std::string;
    using std::vector;
    using std::map;
    using std::tuple;
    using std::make_tuple;
    using std::numeric_limits;

    using nlohmann::json;

    struct Neuron2Dim
    {
        unsigned long long int Row {};
        unsigned long long int Column {};
    };

    //
    // Intermediate base class for models implementing neuron dynamics.
    //
    template<class MODELHELPERTYPE>
    class ModelNeuronInitializer : public IModelInitializer<MODELHELPERTYPE>
    {
    protected:
        MODELHELPERTYPE helper_;

        int strength_ { 21 };

        map<string, tuple<int, int>> namedNeurons_ { };

        const json& Configuration() const { return helper_.Configuration(); }

    public:
        ModelNeuronInitializer(MODELHELPERTYPE helper) :
            helper_(helper)
        {
            LoadOptionalNamedNeurons();
        }

    public:
        // IModelInitializer implementaton
        virtual void CreateProxy(MODELHELPERTYPE helper) override { }

    protected:
        void InitializeAnInput(int row, int column)
        {
            auto sourceIndex = helper_.GetIndex(row, column);
            this->helper_.WireInput(sourceIndex, strength_);
        }

        void InitializeAnInput(const Neuron2Dim& neuron)
        {
            InitializeAnInput(neuron.Row, neuron.Column);
        }

        void InitializeAConnection(const int row, const int column, const int destRow, const int destCol)
        {
            auto sourceIndex = helper_.GetIndex(row, column);
            auto destinationIndex = helper_.GetIndex(destRow, destCol);
            this->helper_.Wire(sourceIndex, destinationIndex, strength_);
        }

        void InitializeAConnection(const Neuron2Dim& source, const Neuron2Dim& destination)
        {
            InitializeAConnection(source.Row, source.Column, destination.Row, destination.Column);
        }

        unsigned long long int GetIndex(const Neuron2Dim& source)
        {
            return helper_.GetIndex(source.Row, source.Column);
        }

        void SetExcitatoryNeuronType(const unsigned long long int source)
        {
            this->helper_.SetExcitatoryNeuronType(source);
        }

        void SetExcitatoryNeuronType(const Neuron2Dim& source)
        {
            SetExcitatoryNeuronType(GetIndex(source.Row, source.Column));
        }

        void SetInhibitoryNeuronType(const unsigned long long int source)
        {
            this->helper_.SetInhibitoryNeuronType(source);
        }

        void SetInhibitoryNeuronType(const Neuron2Dim& source)
        {
            SetInhibitoryNeuronType(helper_.GetIndex(source.Row, source.Column));
        }

        const Neuron2Dim ResolveNeuron(const string& neuronName) const
        {
            auto neuronIt = namedNeurons_.find(neuronName);
            if (neuronIt != namedNeurons_.end())
            {
                auto& [row, col] = neuronIt->second;
                Neuron2Dim yyy { .Row = (unsigned long long)row, .Column = (unsigned long long)col };
                cout << "ResolveNeuron(" << neuronName << ") found coordinates [" << yyy.Row << ", " << yyy.Column << "]\n";
                return yyy;
            }

            Neuron2Dim xxx { .Row = numeric_limits<unsigned long long>::max(), .Column = numeric_limits<unsigned long long>::max() };
            cout << "ResolveNeuron(" << neuronName << ") NOT found coordinates [" << xxx.Row << ", " << xxx.Column << "]\n";
            return xxx;
        }

    private:
        void LoadOptionalNamedNeurons()
        {
            const json& configuration = Configuration();
            auto& modelSection = configuration["Model"];
            if (!modelSection.is_null() && modelSection.contains("Neurons"))
            {
                auto& namedNeuronsElement = modelSection["Neurons"];
                if (namedNeuronsElement.is_object())
                {
                    for (auto& neuron: namedNeuronsElement.items())
                    {
                        auto neuronName = neuron.key();
                        auto positionArray = neuron.value().get<std::vector<int>>();
                        auto ypos = positionArray[0];
                        auto xpos = positionArray[1];

                        namedNeurons_[neuronName] = make_tuple(ypos, xpos);
                    }
                }
            }
        }
    };
}
