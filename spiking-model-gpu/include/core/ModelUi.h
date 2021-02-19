#pragma once

#include <string>
#include <iostream>
#include <map>
#include <chrono>
#include <numeric>

#include "KeyListener.h"

namespace embeddedpenguins::core::neuron::model
{
    using std::cout;
    using std::string;
    using std::map;
    using std::chrono::microseconds;
    using std::ceil;
    using std::floor;

    template<class MODELRUNNERTYPE, class MODELHELPERTYPE>
    class ModelUi
    {
        bool displayOn_ { true };
        bool displayMonitoredNeurons_ { false };
        unsigned int width_ { 100 };
        unsigned int height_ { 100 };
        unsigned int centerWidth_ {};
        unsigned int centerHeight_ {};

        unsigned int windowWidth_ = 15;
        unsigned int windowHeight_ = 15;
        string cls {"\033[2J\033[H"};

        map<string, tuple<int, int>> namedNeurons_ { };

    protected:
        MODELRUNNERTYPE& modelRunner_;
        MODELHELPERTYPE& helper_;

    public:
        ModelUi(MODELRUNNERTYPE& modelRunner, MODELHELPERTYPE& helper) :
            modelRunner_(modelRunner),
            helper_(helper)
        {
            width_ = helper_.Width();
            height_ = helper_.Height();

            if (windowWidth_ > width_) windowWidth_ = width_;
            if (windowHeight_ > height_) windowHeight_ = height_;

            centerHeight_ = ceil(windowHeight_ / 2);
            centerWidth_ = ceil(width_ / 2);

            LoadOptionalNamedNeurons();
        }

        char PrintAndListenForQuit()
        {
            constexpr char KEY_UP = 'A';
            constexpr char KEY_DOWN = 'B';
            constexpr char KEY_LEFT = 'D';
            constexpr char KEY_RIGHT = 'C';

            char c {' '};
            {
                KeyListener listener;

                bool quit {false};
                while (!quit)
                {
                    if (displayOn_)
                    {
                        if (displayMonitoredNeurons_)
                            PrintMonitoredNeurons();
                        else
                            PrintNetworkScan();
                    }

                    auto gotChar = listener.Listen(50'000, c);
                    if (gotChar)
                    {
                        switch (c)
                        {
                            case KEY_UP:
                                if (centerHeight_ > ceil(windowHeight_ / 2)) centerHeight_--;
                                break;

                            case KEY_DOWN:
                                centerHeight_++;
                                if (centerHeight_ >= floor(height_ - (windowHeight_ / 2))) centerHeight_ = floor(height_ - (windowHeight_ / 2));
                                break;

                            case KEY_LEFT:
                                if (centerWidth_ > ceil(windowWidth_ / 2)) centerWidth_--;
                                break;

                            case KEY_RIGHT:
                                centerWidth_++;
                                if (centerWidth_ >= floor(width_ - (windowWidth_ / 2))) centerWidth_ = floor(width_ - (windowWidth_ / 2));
                                break;

                            case '=':
                            case '+':
                            {
                                auto newPeriod = modelRunner_.EnginePeriod() / 10;
                                if (newPeriod < microseconds(100)) newPeriod = microseconds(100);
                                modelRunner_.EnginePeriod() = newPeriod;
                                break;
                            }

                            case '-':
                            {
                                auto newPeriod = modelRunner_.EnginePeriod() * 10;
                                if (newPeriod > microseconds(10'000'000)) newPeriod = microseconds(10'000'000);
                                modelRunner_.EnginePeriod() = newPeriod;
                                break;
                            }

                            case 'q':
                            case 'Q':
                                quit = true;
                                break;

                            default:
                                break;
                        }
                    }
                }
            }

            cout << "Received keystroke " << c << ", quitting\n";
            return c;
        }

        void PrintNetworkScan()
        {
            cout << cls;

            auto neuronIndex = ((width_ * (centerHeight_ - (windowHeight_ / 2))) + centerWidth_ - (windowWidth_ / 2));
            for (auto high = windowHeight_; high; --high)
            {
                for (auto wide = windowWidth_; wide; --wide)
                {
                    cout << EmitToken(neuronIndex);
                    neuronIndex++;
                }
                cout << '\n';

                neuronIndex += width_ - windowWidth_;
                if (neuronIndex > helper_.Carrier().ModelSize()) neuronIndex = 0;
            }

            cout
                <<  Legend() << ":(" << centerWidth_ << "," << centerHeight_ << ") "
                << " Tick: " << modelRunner_.EnginePeriod().count() << " us "
                << "Iterations: " << modelRunner_.getModelEngine().GetIterations() 
                << "  Total work: " << modelRunner_.getModelEngine().GetTotalWork() 
                << "                 \n";

            cout << "Arrow keys to navigate       + and - keys control speed            q to quit\n";
        }

        void PrintMonitoredNeurons()
        {
            cout << cls;

            for (auto& [neuronName, posTuple] : namedNeurons_)
            {
                auto& [ypos, xpos] = posTuple;
                auto neuronIndex = helper_.GetIndex(ypos, xpos);
                auto& neuron = helper_.Carrier().NeuronsHost[neuronIndex];
                cout << "Neuron " << std::setw(15) << neuronName << " [" << ypos << ", " << xpos << "] = " << std::setw(4) << neuronIndex << ": ";
                cout << std::setw(5) << neuron.Activation << "(" << std::setw(3) << neuron.TicksSinceLastSpike << ")";
                cout << std::endl;

                auto& synapsesForNeuron = helper_.Carrier().SynapsesHost[neuronIndex];
                for (auto synapseId = 0; synapseId < SynapticConnectionsPerNode; synapseId++)
                {
                    if (*(unsigned long*)&synapsesForNeuron[synapseId].PresynapticNeuron != 0)
                    {
                        cout << std::setw(20) << (unsigned long int)synapsesForNeuron[synapseId].PresynapticNeuron
                        << "(" << std::setw(3) << (unsigned int)synapsesForNeuron[synapseId].Strength << ")  ";
                    }
                }
                cout << std::endl;
                cout << std::endl;
            }
            cout << std::endl;

            cout
                <<  Legend() << ": "
                << " Tick: " << modelRunner_.EnginePeriod().count() << " us "
                << "Iterations: " << modelRunner_.getModelEngine().GetIterations() 
                << "  Total work: " << modelRunner_.getModelEngine().GetTotalWork() 
                << "                 \n";

            cout << "+ and - keys control speed            q to quit\n";
        }

        void LoadOptionalNamedNeurons()
        {
            const json& configuration = modelRunner_.Configuration();
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
                        auto xpos = positionArray[0];
                        auto ypos = positionArray[1];

                        namedNeurons_[neuronName] = make_tuple(xpos, ypos);
                    }
                }
            }
        }

        void ParseArguments(int argc, char* argv[])
        {
            for (auto i = 0; i < argc; i++)
            {
                string arg = argv[i];
                if (arg == "-d" || arg == "--nodisplay")
                {
                    displayOn_ = false;
                    cout << "Found " << arg << " flag, turning display off \n";
                }
                if (arg == "-m" || arg == "--monitored")
                {
                    displayMonitoredNeurons_ = true;
                    cout << "Found " << arg << " flag, displaying monitored neurons \n";
                }
            }
        }

        virtual const string& Legend() = 0;
        virtual char EmitToken(unsigned long neuronIndex) = 0;
    };
}
