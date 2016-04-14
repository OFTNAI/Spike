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

#include "Neurons.h"
#include "Connections.h"
#include "Inputs.h"


// Functions!
void GPUDeviceComputation (
					Neurons * neurons,
					Connections * connections,
					Inputs * inputs,

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

__global__ void randoms(curandState_t* states, float* numbers, size_t numNeurons);

#endif