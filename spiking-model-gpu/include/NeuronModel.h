#pragma once

#include <memory>

#include <cuda/runtime_api.hpp>

#include <cuda_runtime_api.h>
#include "cuda.h"

#include "NeuronCommon.h"
#include "NeuronNode.h"
#include "NeuronSynapse.h"

namespace embeddedpenguins::gpu::neuron::model
{
    class NeuronModel
    {
        unsigned long int modelSize_;
        cuda::device::id_t deviceId_;
        cuda::device_t device_;

        std::unique_ptr<NeuronNode[]> neuronsHost_;
        std::unique_ptr<NeuronSynapse[][SynapticConnectionsPerNode]> synapsesHost_;
        cuda::memory::device::unique_ptr<NeuronNode[]> neuronsDevice_;
        cuda::memory::device::unique_ptr<NeuronSynapse[][SynapticConnectionsPerNode]> synapsesDevice_;

    public:
        NeuronModel(unsigned long int modelSize);
        void InitializeModel();
        void Run();

    private:
        void Initialize();
        void ExecuteAStep();
        void InitializeForTest();
        void PrintSynapses(int w);
        void PrintNeurons(int w);
    };
}
