#pragma once

#include<limits>
#include <vector>
#include <algorithm>

namespace embeddedpenguins::gpu::neuron::model
{
    using std::vector;
    using std::numeric_limits;

    //
    // The global model allows for splitting along expansion lines, 
    // where each expansion may be deployed to a different
    // engine.  So for each engine, the expansions not deployed
    // on that engine are compressed to empty expansions.
    // This class provides a mapping between the global index 
    // of a model neuron and its local storage, taking into account
    // the empty expansions.
    //
    class ModelMapper
    {
        //
        // Capture enough information about an expansion to allow
        // forward mapping from global model indexes to local indexes,
        // taking into account the empty expansions.
        struct ExpansionMap
        {
            //
            // Starting index within the local model
            // of the first neuron in this expansion.
            // Note this may be different from the global
            // model index, due to skipping expansions
            // not deployed to this engine.
            unsigned long int ExpansionStart {};

            //
            // Ending index + 1 within the local model
            // of the last neuron in this expansion.
            // ExpansionOffset == ExpansionOffset for an empty expansion.
            unsigned long int ExpansionEnd {};
        };

        //
        // All model expansions mapped into this engine.
        vector<ExpansionMap> expansionMap_ {};

    public:
        void AddExpansion(unsigned long int start, unsigned long int length)
        {
            expansionMap_.push_back(ExpansionMap{.ExpansionStart=start, .ExpansionEnd=start+length});
        }

        unsigned long int ExpansionOffset(unsigned short int expansionId) const
        {
            if (expansionId >= expansionMap_.size())
                return numeric_limits<unsigned long>::max();

            return expansionMap_[expansionId].ExpansionStart;
        }
    };
}
