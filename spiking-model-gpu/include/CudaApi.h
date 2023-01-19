#pragma once

#if CUDA_VERSION < 12000
#include <cuda/runtime_api.hpp>
#endif

#include <cuda_runtime_api.h>
#include "cuda.h"

#include "NeuronNode.h"
#include "NeuronPostSynapse.h"
#include "NeuronPreSynapse.h"

using embeddedpenguins::gpu::neuron::model::NeuronNode;
using embeddedpenguins::gpu::neuron::model::NeuronPostSynapse;
using embeddedpenguins::gpu::neuron::model::NeuronPreSynapse;
using embeddedpenguins::gpu::neuron::model::SynapticConnectionsPerNode;

//
// Declare shim methods in a .cu file.
//
void
DeviceFixupShim(
#if CUDA_VERSION < 12000
    cuda::device_t& device,
#endif
    unsigned long int modelSize,
    float postsynapticIncreaseFunction[],
    NeuronNode neurons[],
    NeuronPostSynapse postSynapses[][SynapticConnectionsPerNode],
    NeuronPreSynapse preSynapses[][SynapticConnectionsPerNode]);

void
StreamInputShim(
#if CUDA_VERSION < 12000
    cuda::device_t& device,
#endif
    unsigned long int modelSize,
    unsigned long int inputSize,
    unsigned long long int inputNeurons[]);

void
ModelSynapsesShim(
#if CUDA_VERSION < 12000
    cuda::device_t& device,
#endif
    unsigned long int modelSize);

void
ModelTimersShim(
#if CUDA_VERSION < 12000
    cuda::device_t& device,
#endif
    unsigned long int modelSize);

void
ModelTickShim(
#if CUDA_VERSION < 12000
    cuda::device_t& device,
#endif
    unsigned long int modelSize);

void
ModelPlasticityShim(
#if CUDA_VERSION < 12000
    cuda::device_t& device,
#endif
    unsigned long int modelSize);
