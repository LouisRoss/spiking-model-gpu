#pragma once

#include <string>
#include <vector>

#include "cuda.h"               // For CUDA_VERSION
#include <cuda_runtime.h>
#if CUDA_VERSION < 12000
#include <cuda/runtime_api.hpp>
#endif

#include "NeuronNode.h"
#include "NeuronNode.h"
#include "NeuronPostSynapse.h"
#include "NeuronPreSynapse.h"
#include "NeuronCommon.h"

namespace embeddedpenguins::gpu::neuron::model
{
    typedef NeuronPostSynapse PostSynapseArray[SynapticConnectionsPerNode];
    typedef NeuronPreSynapse PreSynapseArray[SynapticConnectionsPerNode];

    using std::string;
    using std::vector;

    struct GpuModelCarrier
    {
        unsigned long int NeuronCount { };
#if CUDA_VERSION < 12000
        cuda::device::id_t DeviceId;
        cuda::device_t Device;
#else
        int DeviceId { };
#endif

        std::unique_ptr<unsigned long[]> RequiredPostsynapticConnections { };
        std::unique_ptr<float[]> PostsynapticIncreaseFuncHost { };
        std::unique_ptr<NeuronNode[]> NeuronsHost { };
        std::unique_ptr<NeuronPostSynapse[][SynapticConnectionsPerNode]> PostSynapseHost { };
        std::unique_ptr<NeuronPreSynapse[][SynapticConnectionsPerNode]> PreSynapsesHost { };
        std::unique_ptr<unsigned long[]> InputSignalsHost { };
#if CUDA_VERSION < 12000
        cuda::memory::device::unique_ptr<float[]> PostsynapticIncreaseFuncDevice { };
        cuda::memory::device::unique_ptr<NeuronNode[]> NeuronsDevice { };
        cuda::memory::device::unique_ptr<NeuronPostSynapse[][SynapticConnectionsPerNode]> SynapsesDevice { };
        cuda::memory::device::unique_ptr<NeuronPreSynapse[][SynapticConnectionsPerNode]> PreSynapsesDevice { };
        cuda::memory::device::unique_ptr<unsigned long long[]> InputSignalsDevice { };
#else
private:
        void* postsynapticIncreaseFuncDevice_ { 0 };
        void* neuronsDevice_ { 0 };
        void* synapsesDevice_ { 0 };
        void* preSynapsesDevice_ { 0 };
        void* inputSignalsDevice_ { 0 };
public:
        float* PostsynapticIncreaseFuncDevice() { return reinterpret_cast<float*>(postsynapticIncreaseFuncDevice_); }
        NeuronNode* NeuronsDevice() { return reinterpret_cast<NeuronNode*>(neuronsDevice_); }
        PostSynapseArray* SynapsesDevice() { return reinterpret_cast<PostSynapseArray*>(synapsesDevice_); }
        PreSynapseArray* PreSynapsesDevice() { return reinterpret_cast<PreSynapseArray*>(preSynapsesDevice_); }
        unsigned long long* InputSignalsDevice() { return reinterpret_cast<unsigned long long*>(inputSignalsDevice_); }
#endif
        bool Valid { false };

        const unsigned long int ModelSize() const { return NeuronCount; }

        GpuModelCarrier() 
#if CUDA_VERSION < 12000
        :
            DeviceId(cuda::device::default_device_id),
            Device(cuda::device::get(DeviceId).make_current())
        {

        }
#else
        {
            cudaDeviceProp prop;
            int device;

            memset(&prop, 0, sizeof(cudaDeviceProp));
            prop.major = 10;
            prop.minor = 1;
            prop.multiProcessorCount = 64;
            cudaChooseDevice(&device, &prop);

            DeviceId = device;
        }
#endif

        GpuModelCarrier(const GpuModelCarrier& other) = delete;
        const GpuModelCarrier operator=(const GpuModelCarrier& other) = delete;

#if CUDA_VERSION >= 12000
        ~GpuModelCarrier()
        {
            FreeDevice();
        }
#endif

        void AllocateDevice(int neuronCount)
        {
            NeuronCount = neuronCount;

            RequiredPostsynapticConnections = std::make_unique<unsigned long[]>(NeuronCount);
            PostsynapticIncreaseFuncHost = std::make_unique<float[]>(PostsynapticPlasticityPeriod);
            NeuronsHost = std::make_unique<NeuronNode[]>(NeuronCount);
            PostSynapseHost = std::make_unique<NeuronPostSynapse[][SynapticConnectionsPerNode]>(NeuronCount);
            PreSynapsesHost = std::make_unique<NeuronPreSynapse[][SynapticConnectionsPerNode]>(NeuronCount);
            InputSignalsHost = std::make_unique<unsigned long[]>(InputBufferSize);
#if CUDA_VERSION < 12000
            PostsynapticIncreaseFuncDevice = cuda::memory::device::make_unique<float[]>(Device, PostsynapticPlasticityPeriod);
            NeuronsDevice = cuda::memory::device::make_unique<NeuronNode[]>(Device, NeuronCount);
            SynapsesDevice = cuda::memory::device::make_unique<NeuronPostSynapse[][SynapticConnectionsPerNode]>(Device, NeuronCount);
            PreSynapsesDevice = cuda::memory::device::make_unique<NeuronPreSynapse[][SynapticConnectionsPerNode]>(Device, NeuronCount);
            InputSignalsDevice = cuda::memory::device::make_unique<unsigned long long[]>(Device, InputBufferSize);
#else
            FreeDevice();

            cudaMalloc(&postsynapticIncreaseFuncDevice_, sizeof(float) * PostsynapticPlasticityPeriod);
            cudaMalloc(&neuronsDevice_, sizeof(NeuronNode) * NeuronCount);
            cudaMalloc(&synapsesDevice_, sizeof(NeuronPostSynapse) * SynapticConnectionsPerNode * NeuronCount);
            cudaMalloc(&preSynapsesDevice_, sizeof(NeuronPreSynapse) * SynapticConnectionsPerNode * NeuronCount);
            cudaMalloc(&inputSignalsDevice_, sizeof(unsigned long long) * InputBufferSize);
#endif
        }

#if CUDA_VERSION >= 12000
private:
        void FreeDevice()
        {
            cudaFree(postsynapticIncreaseFuncDevice_);
            postsynapticIncreaseFuncDevice_ = 0;

            cudaFree(neuronsDevice_);
            neuronsDevice_ = 0;

            cudaFree((void*) synapsesDevice_);
            synapsesDevice_ = 0;

            cudaFree((void*) preSynapsesDevice_);
            preSynapsesDevice_ = 0;

            cudaFree((void*) inputSignalsDevice_);
            inputSignalsDevice_ = 0;
        }
#endif
    };
}
