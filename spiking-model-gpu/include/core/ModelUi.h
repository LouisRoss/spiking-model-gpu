#pragma once
//#include <stdio.h>
#include <string>
//#include <vector>
//#include <tuple>
#include <iostream>
//#include <memory>
#include <chrono>

#include "KeyListener.h"

namespace embeddedpenguins::core::neuron::model
{
    using std::cout;
    using std::string;
    //using std::vector;
    //using std::tuple;
    //using std::make_tuple;
    //using std::chrono::nanoseconds;
    //using std::chrono::system_clock;
    //using std::chrono::high_resolution_clock;
    //using std::chrono::duration_cast;
    //using std::chrono::milliseconds;
    using std::chrono::microseconds;
    //using std::chrono::ceil;

    template<class MODELRUNNERTYPE, class MODELHELPERTYPE>
    class ModelUi
    {
        bool displayOn_ { true };
        unsigned long int width_ { 100 };
        unsigned long int height_ { 100 };
        unsigned long int centerWidth_ {};
        unsigned long int centerHeight_ {};

        static constexpr int windowWidth = 100;
        static constexpr int windowHeight = 30;
        string cls {"\033[2J\033[H"};

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
                    if (displayOn_) PrintNetworkScan();
                    auto gotChar = listener.Listen(50'000, c);
                    if (gotChar)
                    {
                        switch (c)
                        {
                            case KEY_UP:
                                if (centerHeight_ > windowHeight / 2) centerHeight_--;
                                if (centerHeight_ < windowHeight / 2) centerHeight_ = windowHeight / 2;
                                break;

                            case KEY_DOWN:
                                centerHeight_++;
                                if (centerHeight_ >= height_ - (windowHeight / 2)) centerHeight_ = height_ - (windowHeight / 2) - 1;
                                break;

                            case KEY_LEFT:
                                if (centerWidth_ > windowWidth / 2) centerWidth_--;
                                if (centerWidth_ < windowWidth / 2) centerWidth_ = windowWidth / 2;
                                break;

                            case KEY_RIGHT:
                                centerWidth_++;
                                if (centerWidth_ >= width_ - (windowWidth / 2)) centerWidth_ = width_ - (windowWidth / 2) - 1;
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

            auto neuronIndex = ((width_ * (centerHeight_ - (windowHeight / 2))) + centerWidth_ - (windowWidth / 2));
            for (auto high = windowHeight; high; --high)
            {
                for (auto wide = windowWidth; wide; --wide)
                {
                    cout << EmitToken(neuronIndex);
                    neuronIndex++;
                }
                cout << '\n';

                neuronIndex += width_ - windowWidth;
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

        void ParseArguments(int argc, char* argv[])
        {
            for (auto i = 0; i < argc; i++)
            {
                string arg = argv[i];
                if (arg == "-d" || arg == "--nodisplay")
                {
                    displayOn_ = false;
                    cout << "Found -d flag, turning display off \n";
                }
            }
        }

        virtual const string& Legend() = 0;
        virtual char EmitToken(unsigned long neuronIndex) = 0;
    };
}
