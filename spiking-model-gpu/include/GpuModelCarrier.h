#pragma once

#include <string>
#include <vector>

#include <cuda/runtime_api.hpp>

#include "NeuronNode.h"
#include "NeuronNode.h"
#include "NeuronSynapse.h"

namespace embeddedpenguins::gpu::neuron::model
{
    using std::string;
    using std::vector;

    struct GpuModelCarrier
    {
        unsigned long int NeuronCount { };
        cuda::device::id_t DeviceId;
        cuda::device_t Device;

        std::unique_ptr<unsigned long[]> RequiredPostsynapticConnections { };
        std::unique_ptr<NeuronNode[]> NeuronsHost { };
        std::unique_ptr<NeuronSynapse[][SynapticConnectionsPerNode]> SynapsesHost { };
        std::unique_ptr<unsigned long[]> InputSignalsHost { };
        cuda::memory::device::unique_ptr<NeuronNode[]> NeuronsDevice { };
        cuda::memory::device::unique_ptr<NeuronSynapse[][SynapticConnectionsPerNode]> SynapsesDevice { };
        cuda::memory::device::unique_ptr<unsigned long long[]> InputSignalsDevice { };
        bool Valid { false };

        const unsigned long int ModelSize() const { return NeuronCount; }

        GpuModelCarrier() : 
            DeviceId(cuda::device::default_device_id),
            Device(cuda::device::get(DeviceId).make_current())
        {
        }

        GpuModelCarrier(const GpuModelCarrier& other) = delete;
        const GpuModelCarrier operator=(const GpuModelCarrier& other) = delete;
    };
}
