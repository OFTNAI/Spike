// CUDAcode Class Header
// CUDAcode.h
//
//	Author: Nasir Ahmad
//	Date: 8/12/2015

//  Adapted by Nasir Ahmad and James Isbister
//	Date: 23/3/2016

#ifndef CUDAcode_H
#define CUDAcode_H

// cuRand Library import
#include <curand.h>
#include <curand_kernel.h>

//	CUDA library
#include <cuda.h>

// Neuron structure
#include "NeuronDynamics.h"
#include "STDPDynamics.h"
// Functions!
void GPUDeviceComputation (
					size_t numNeurons,
					size_t numConnections,
					int* presynaptic_neuron_indices,
					int* postsynaptic_neuron_indices,
					int* delays,
					float* weights,
					int* stdp,
					float* lastactive,
					struct neuron_struct* neuronpop_variables,
					int numStimuli,
					int* numEntries,
					int** genids,
					float** gentimes,
					struct stdp_struct stdp_vars,
					float timestep,
					float totaltime,
					int numEpochs,
					bool savespikes,
					bool randomPresentation
					);
// GPU Functions
__global__ void init(unsigned int seed, curandState_t* states, size_t numNeurons);
__global__ void randoms(curandState_t* states, float* numbers, size_t numNeurons);
__global__ void currentcalc(int* d_spikes,
							float* d_weights,
							float* d_lastactive,
							int* d_postsynaptic_neuron_indices,
							float* currentinj,
							float currtime,
							size_t numConns,
							size_t numNeurons);
__global__ void synapsespikes(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes,
								float* d_lastspiketime,
								int* d_spikebuffer,
								float currtime,
								size_t numConns,
								size_t numNeurons);
__global__ void spikeCollect(float* d_lastspiketime,
								int* d_tempstorenum,
								int* d_tempstoreID,
								float* d_tempstoretimes,
								float currtime,
								size_t numNeurons);
#endif