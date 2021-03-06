#pragma once

#include <cuda/runtime_api.hpp>

#include <cuda_runtime_api.h>
#include "cuda.h"

#include "NeuronNode.h"
#include "NeuronSynapse.h"

using embeddedpenguins::gpu::neuron::model::NeuronNode;
using embeddedpenguins::gpu::neuron::model::NeuronSynapse;
using embeddedpenguins::gpu::neuron::model::SynapticConnectionsPerNode;

//
// Declare shim methods in a .cu file.
//
void
DeviceFixupShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    NeuronNode neurons[],
    NeuronSynapse synapses[][SynapticConnectionsPerNode]);

void
StreamInputShim(
    cuda::device_t& device,
    unsigned long int modelSize,
    unsigned long int inputSize,
    unsigned long long int inputNeurons[]);

void
ModelSynapsesShim(
    cuda::device_t& device,
    unsigned long int modelSize);

void
ModelTimersShim(
    cuda::device_t& device,
    unsigned long int modelSize);

void
ModelTickShim(
    cuda::device_t& device,
    unsigned long int modelSize);

void
ModelPlasticityShim(
    cuda::device_t& device,
    unsigned long int modelSize);
