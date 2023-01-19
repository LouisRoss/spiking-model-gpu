#pragma once

#include <memory>

#if CUDA_VERSION < 12000
#include <cuda/runtime_api.hpp>
#endif

#include <cuda_runtime_api.h>
#include "cuda.h"

#include "NeuronCommon.h"
#include "NeuronNode.h"
#include "NeuronPostSynapse.h"

namespace embeddedpenguins::gpu::neuron::model
{
    class NeuronModel
    {
        unsigned long int modelSize_;
        cuda::device::id_t deviceId_;
        cuda::device_t device_;

        std::unique_ptr<NeuronNode[]> neuronsHost_;
        std::unique_ptr<NeuronPostSynapse[][SynapticConnectionsPerNode]> synapsesHost_;
        cuda::memory::device::unique_ptr<NeuronNode[]> neuronsDevice_;
        cuda::memory::device::unique_ptr<NeuronPostSynapse[][SynapticConnectionsPerNode]> synapsesDevice_;

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
