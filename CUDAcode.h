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
#include "STDPDynamics.h"

#include "Neurons.h"
#include "Connections.h"

// Functions!
void GPUDeviceComputation (
					Neurons * neurons,
					Connections * connections,

					float total_time_per_epoch,
					int number_of_epochs,
					float timestep,
					bool save_spikes,

					int numStimuli,
					int* numEntries,
					int** genids,
					float** gentimes,
					
					bool randomPresentation
					);


// GPU Functions
__global__ void init(unsigned int seed, curandState_t* states, size_t numNeurons);
__global__ void randoms(curandState_t* states, float* numbers, size_t numNeurons);
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