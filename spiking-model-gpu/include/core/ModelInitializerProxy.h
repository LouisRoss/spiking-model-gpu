#pragma once

#include "string"
#include <vector>
#include <iostream>
#include <dlfcn.h>

#include "nlohmann/json.hpp"

#include "ConfigurationRepository.h"
#include "IModelInitializer.h"

namespace embeddedpenguins::core::neuron::model
{
    using std::string;
    using std::vector;
    using std::cout;

    using nlohmann::json;

    //
    // This proxy class implements the IModelInitializer<> interface by loading the
    // named shared library.  The shared library must contain a class that also implements
    // the IModelInitializer<> interface, plus two C-style methods to create and destroy
    // instances of its class on the heap.
    //
    template<class MODELHELPERTYPE>
    class ModelInitializerProxy : IModelInitializer<MODELHELPERTYPE>
    {
        using InitializerCreator = IModelInitializer<MODELHELPERTYPE>* (*)(MODELHELPERTYPE helper);
        using InitializerDeleter = void (*)(IModelInitializer<MODELHELPERTYPE>*);

        const string initializerSharedLibraryPath_ {};

        bool valid_ { false };
        void* initializerLibrary_ {};
        InitializerCreator createInitializer_ {};
        InitializerDeleter deleteInitializer_ {};
        IModelInitializer<MODELHELPERTYPE>* initializer_ {};

    public:
        //
        // Use of a two-step creation process is mandatory:  Instantiate a proxy with the
        // path to the shared library, then call CreateProxy() with the model.
        //
        ModelInitializerProxy(const string& initializerSharedLibraryPath) :
            initializerSharedLibraryPath_(initializerSharedLibraryPath)
        {
            cout << "ModelInitializerProxy ctor(" << initializerSharedLibraryPath << ")\n";
        }

        ~ModelInitializerProxy() override
        {
            if (deleteInitializer_ != nullptr)
                deleteInitializer_(initializer_);

            if (initializerLibrary_ != nullptr)
                dlclose(initializerLibrary_);
        }

    public:
        // IModelInitializer implementaton
        virtual void CreateProxy(MODELHELPERTYPE helper) override
        {
            LoadInitializer();
            if (createInitializer_ != nullptr)
                initializer_ = createInitializer_(helper);
        }

        virtual void Initialize() override
        {
            if (initializer_ && valid_)
                initializer_->Initialize();
        }

    private:
        void LoadInitializer()
        {
            cout << "Loading initializer library from " << initializerSharedLibraryPath_ << "\n";

            initializerLibrary_ = dlopen(initializerSharedLibraryPath_.c_str(), RTLD_LAZY);
            if (!initializerLibrary_)
            {
                cout << "Cannot load library '" << initializerSharedLibraryPath_ << "': " << dlerror() << "\n";
                valid_ = false;
                return;
            }

            // reset errors
            dlerror();
            createInitializer_ = (InitializerCreator)dlsym(initializerLibrary_, "create");
            const char* dlsym_error = dlerror();
            if (dlsym_error)
            {
                cout << "Cannot load symbol 'create': " << dlsym_error << "\n";
                valid_ = false;
                return;
            }

            // reset errors
            dlerror();
            deleteInitializer_ = (InitializerDeleter)dlsym(initializerLibrary_, "destroy");
            dlsym_error = dlerror();
            if (dlsym_error)
            {
                cout << "Cannot load symbol 'destroy': " << dlsym_error << "\n";
                valid_ = false;
                return;
            }

            valid_ = true;
        }
    };
}
