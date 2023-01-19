#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <dlfcn.h>

#include "SpikeOutputs/ISpikeOutput.h"

namespace embeddedpenguins::core::neuron::model
{
    using std::string;
    using std::vector;
    using std::cout;

    using embeddedpenguins::core::neuron::model::ConfigurationRepository;

    class SpikeOutputProxy : public ISpikeOutput
    {
        using SpikeOutputCreator = ISpikeOutput* (*)(ModelEngineContext&);
        using SpikeOutputDeleter = void (*)(ISpikeOutput*);

        const string spikeOutputSharedLibraryPath_;

        bool valid_ { false };
        void* spikeOutputLibrary_ {};
        SpikeOutputCreator createSpikeOutput_ {};
        SpikeOutputDeleter deleteSpikeOutput_ {};
        ISpikeOutput* spikeOutput_ {};
        string errorReason_ {};

        bool respectDisableFlag_ { true };
        bool isInterestedInSpikeTime_ { true };
        bool isInterestedInRefractoryTime_ { true };
        bool isInterestedInRecentTime_ { true };

    public:
        const string& ErrorReason() const { return errorReason_; }
        const bool Valid() const { return valid_; }

    public:
        //
        // Use of a two-step creation process is mandatory:  Instantiate a proxy with the
        // path to the shared library, then call CreateProxy() with the model.
        //
        SpikeOutputProxy(const string& spikeOutputSharedLibraryPath) :
            spikeOutputSharedLibraryPath_(spikeOutputSharedLibraryPath)
        {
        }

        ~SpikeOutputProxy() override
        {
            if (deleteSpikeOutput_ != nullptr)
                deleteSpikeOutput_(spikeOutput_);

            if (spikeOutputLibrary_ != nullptr)
                dlclose(spikeOutputLibrary_);
        }

    public:
        // ISpikeOutput implementaton
        virtual void CreateProxy(ModelEngineContext& context) override
        {
            LoadISpikeOutput();
            if (createSpikeOutput_ != nullptr)
                spikeOutput_ = createSpikeOutput_(context);

            if (spikeOutput_ && valid_)
            {
                respectDisableFlag_ = spikeOutput_->RespectDisableFlag();
                cout << "Spike output " << spikeOutputSharedLibraryPath_ << (respectDisableFlag_ ? " respects" : " does not respect") << " disable flag\n";
                isInterestedInSpikeTime_ = spikeOutput_->IsInterestedIn(NeuronRecordType::Spike);
                cout << "Spike output " << spikeOutputSharedLibraryPath_ << (isInterestedInSpikeTime_ ? " is" : " is not") << " interested in spike-time signals\n";
                isInterestedInRefractoryTime_ = spikeOutput_->IsInterestedIn(NeuronRecordType::Refractory);
                cout << "Spike output " << spikeOutputSharedLibraryPath_ << (isInterestedInRefractoryTime_ ? " is" : " is not") << " interested in refractory-time signals\n";
                isInterestedInRecentTime_ = spikeOutput_->IsInterestedIn(NeuronRecordType::Decay);
                cout << "Spike output " << spikeOutputSharedLibraryPath_ << (isInterestedInRecentTime_ ? " is" : " is not") << " interested in decay-time signals\n";
            }
        }

        virtual bool Connect() override
        {
            errorReason_.clear();

            if (spikeOutput_ && valid_)
                return spikeOutput_->Connect();

            if (!spikeOutput_)
            {
                std::ostringstream os;
                os << "Error calling Connect(): spike output library " 
                    << spikeOutputSharedLibraryPath_ << " not loaded";
                errorReason_ = os.str();
            }

            if (!valid_)
            {
                std::ostringstream os;
                os << "Error calling Connect(): invalid spike output library " 
                    << spikeOutputSharedLibraryPath_;
                errorReason_ = os.str();
            }

            return false;
        }

        virtual bool Connect(const string& connectionString, unsigned int filterBottom, unsigned int filterLength, unsigned int toIndex, unsigned int toOffset) override
        {
            errorReason_.clear();

            if (spikeOutput_ && valid_)
                return spikeOutput_->Connect(connectionString, filterBottom, filterLength, toIndex, toOffset);

            if (!spikeOutput_)
            {
                std::ostringstream os;
                os << "Error calling Connect(): spike output library " 
                    << spikeOutputSharedLibraryPath_ << " not loaded";
                errorReason_ = os.str();
            }

            if (!valid_)
            {
                std::ostringstream os;
                os << "Error calling Connect(): invalid spike output library " 
                    << spikeOutputSharedLibraryPath_;
                errorReason_ = os.str();
            }

            return false;
        }

        virtual bool Disconnect() override
        {
            errorReason_.clear();

            if (spikeOutput_ && valid_)
                return spikeOutput_->Disconnect();

            if (!spikeOutput_)
            {
                std::ostringstream os;
                os << "Error calling Disconnect(): spike output library " 
                    << spikeOutputSharedLibraryPath_ << " not loaded";
                errorReason_ = os.str();
            }

            if (!valid_)
            {
                std::ostringstream os;
                os << "Error calling Disconnect(): invalid spike output library " 
                    << spikeOutputSharedLibraryPath_;
                errorReason_ = os.str();
            }

            return false;
        }

        virtual bool RespectDisableFlag() override { return respectDisableFlag_; }

        virtual bool IsInterestedIn(NeuronRecordType type) override
        {
            errorReason_.clear();

            if (spikeOutput_ && valid_)
                return spikeOutput_->IsInterestedIn(type);

            if (!spikeOutput_)
            {
                std::ostringstream os;
                os << "Error calling IsInterestedIn(): spike output library " 
                    << spikeOutputSharedLibraryPath_ << " not loaded";
                errorReason_ = os.str();
                return false;
            }

            if (!valid_)
            {
                std::ostringstream os;
                os << "Error calling IsInterestedIn(): invalid spike output library " 
                    << spikeOutputSharedLibraryPath_;
                errorReason_ = os.str();
                return false;
            }

            return false;
        }

        virtual void StreamOutput(unsigned long long neuronIndex, short int activation, short int hpersensitive, unsigned short synapseIndex, short int synapseStrength, NeuronRecordType  type) override
        {
            errorReason_.clear();

            switch (type)
            {
                case NeuronRecordType::Spike:
                    if (!isInterestedInSpikeTime_) return;
                    break;

                case NeuronRecordType::Refractory:
                    if (!isInterestedInRefractoryTime_) return;
                    break;
                    
                case NeuronRecordType::Decay:
                    if (!isInterestedInRecentTime_) return;
                    break;
                    
                default:
                    break;
            }

            if (spikeOutput_ && valid_)
            {
                spikeOutput_->StreamOutput(neuronIndex, activation, hpersensitive, synapseIndex, synapseStrength, type);
                return;
            }

            if (!spikeOutput_)
            {
                std::ostringstream os;
                os << "Error calling StreamOutput(): spike output library " 
                    << spikeOutputSharedLibraryPath_ << " not loaded";
                errorReason_ = os.str();
            }

            if (!valid_)
            {
                std::ostringstream os;
                os << "Error calling StreamOutput(): invalid spike output library " 
                    << spikeOutputSharedLibraryPath_;
                errorReason_ = os.str();
            }
        }

        virtual void Flush() override
        {
            errorReason_.clear();

            if (spikeOutput_ && valid_)
            {
                spikeOutput_->Flush();
                return;
            }

            if (!spikeOutput_)
            {
                std::ostringstream os;
                os << "Error calling Flush(): spike output library " 
                    << spikeOutputSharedLibraryPath_ << " not loaded";
                errorReason_ = os.str();
            }

            if (!valid_)
            {
                std::ostringstream os;
                os << "Error calling Flush(): invalid spike output library " 
                    << spikeOutputSharedLibraryPath_;
                errorReason_ = os.str();
            }
        }

    private:
        void LoadISpikeOutput()
        {
            errorReason_.clear();
            cout << "Loading spike output library from " << spikeOutputSharedLibraryPath_ << "\n";

            spikeOutputLibrary_ = dlopen(spikeOutputSharedLibraryPath_.c_str(), RTLD_LAZY);
            if (!spikeOutputLibrary_)
            {
                if (errorReason_.empty())
                {
                    std::ostringstream os;
                    os << "Cannot load library '" << spikeOutputSharedLibraryPath_ << "': " << dlerror();
                    errorReason_ = os.str();
                }
                cout << errorReason_ << "\n";
                valid_ = false;
                return;
            }

            // reset errors
            dlerror();
            createSpikeOutput_ = (SpikeOutputCreator)dlsym(spikeOutputLibrary_, "create");
            const char* dlsym_error = dlerror();
            if (dlsym_error)
            {
                if (errorReason_.empty())
                {
                    std::ostringstream os;
                    os << "Cannot load symbol 'create': " << dlsym_error;
                    errorReason_ = os.str();
                }
                cout << errorReason_ << "\n";
                valid_ = false;
                return;
            }

            // reset errors
            dlerror();
            deleteSpikeOutput_ = (SpikeOutputDeleter)dlsym(spikeOutputLibrary_, "destroy");
            dlsym_error = dlerror();
            if (dlsym_error)
            {
                if (errorReason_.empty())
                {
                    std::ostringstream os;
                    os << "Cannot load symbol 'destroy': " << dlsym_error;
                    errorReason_ = os.str();
                }
                cout << errorReason_ << "\n";
                valid_ = false;
                return;
            }

            valid_ = true;
        }
    };
}
