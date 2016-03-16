// Spike Class Header
// Spike.h
//
//	Author: Nasir Ahmad
//	Date: 8/12/2015

#ifndef CUDAcode_H
#define CUDAcode_H

// cuRand Library import
#include <curand.h>
#include <curand_kernel.h>

//	CUDA library
#include <cuda.h>

// Neuron structure
#include "NeuronDynamics.h"

// Functions!
void GPUDeviceComputation (
					size_t numNeurons,
					size_t numConnections,
					int* presyns,
					int* postsyns,
					int* delays,
					float* weights,
					int* stdp,
					float* lastactive,
					struct neuron_struct* neuronpop_variables,
					int numStimuli,
					int* numEntries,
					int** genids,
					float** gentimes,
					float w_max,
					float a_minus,
					float a_plus,
					float tau_minus,
					float tau_plus,
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
							int* d_postsyns,
							float* currentinj,
							float currtime,
							size_t numConns,
							size_t numNeurons);
__global__ void ltdweights(float* d_lastactive,
							float* d_weights,
							int* d_stdp,
							float* d_lastspiketime,
							int* d_postsyns,
							float currtime,
							float w_max,
							float a_minus,
							float tau_minus,
							size_t numConns,
							size_t numNeurons);
__global__ void synapsespikes(int* d_presyns,
								int* d_delays,
								int* d_spikes,
								float* d_lastspiketime,
								int* d_spikebuffer,
								float currtime,
								size_t numConns,
								size_t numNeurons);
__global__ void synapseLTP(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							float a_plus,
							float tau_plus,
							float w_max,
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