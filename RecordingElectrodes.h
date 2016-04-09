//	RecordingElectrodes Class header
//	RecordingElectrodes.h
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

#include <cuda.h>

#include "Connections.h"

class RecordingElectrodes{
public:
	// Constructor/Destructor
	RecordingElectrodes();
	~RecordingElectrodes();

	void initialise_device_pointers();

	void set_threads_per_block_and_blocks_per_grid(int threads);


private:
	dim3 number_of_neuron_blocks_per_grid;
	dim3 threads_per_block;

};



#endif