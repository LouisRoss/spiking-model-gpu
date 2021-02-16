#pragma once

#include <string>
#include <vector>

namespace embeddedpenguins::core::neuron::model
{
    using std::string;
    using std::vector;

    class ISensorInput
    {
    public:
        virtual ~ISensorInput() = default;

        virtual void CreateProxy(const ConfigurationRepository& configuration) = 0;
        virtual bool Connect(const string& connectionString) = 0;
        virtual bool Disconnect() = 0;
        virtual vector<unsigned long long>& StreamInput(unsigned long long int tickNow) = 0;
    };
}
