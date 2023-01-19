#pragma once

#include<limits>
#include <vector>
#include <string>
#include <algorithm>

namespace embeddedpenguins::core::neuron::model
{
    using std::vector;
    using std::numeric_limits;
    using std::string;

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
            // Host name of the engine assigned this
            // expansion.  Suitable for a socket connection.
            string ExpansionEngine {};
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
        void Reset() { expansionMap_.clear(); }
        
        void AddExpansion(const string& engine, unsigned long int start, unsigned long int length)
        {
            expansionMap_.push_back(ExpansionMap{.ExpansionEngine=engine, .ExpansionStart=start, .ExpansionEnd=start+length});
        }

        unsigned long int ExpansionOffset(unsigned short int expansionId) const
        {
            if (expansionId >= expansionMap_.size())
                return numeric_limits<unsigned long>::max();

            return expansionMap_[expansionId].ExpansionStart;
        }

        unsigned long int ExpansionEnd(unsigned short int expansionId) const
        {
            if (expansionId >= expansionMap_.size())
                return numeric_limits<unsigned long>::max();

            return expansionMap_[expansionId].ExpansionEnd;
        }

        const string& ExpansionEngine(unsigned short int expansionId) const
        {
            static string nullEngine {};

            if (expansionId >= expansionMap_.size())
                return nullEngine;

            return expansionMap_[expansionId].ExpansionEngine;
        }
    };
}
