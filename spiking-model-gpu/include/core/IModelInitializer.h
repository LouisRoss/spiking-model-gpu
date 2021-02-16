#pragma once

namespace embeddedpenguins::core::neuron::model
{
    //
    // All model initializers must implement this interface, to allow
    // interchangability between initializers.
    //
    template<class MODELHELPERTYPE>
    class IModelInitializer
    {
    public:
        virtual ~IModelInitializer() = default;

        virtual void CreateProxy(MODELHELPERTYPE helper) = 0;

        //
        // Called before the model is run, this required method must 
        // initialize any state needed for a specific use case.
        //
        virtual void Initialize() = 0;
    };
}
