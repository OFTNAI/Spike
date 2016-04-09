//	RecordingElectrodes Class C++
//	RecordingElectrodes.cu
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#include "RecordingElectrodes.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDAErrorCheckHelpers.h"


// RecordingElectrodes Constructor
RecordingElectrodes::RecordingElectrodes() {


}


// RecordingElectrodes Destructor
RecordingElectrodes::~RecordingElectrodes() {

}


void RecordingElectrodes::initialise_device_pointers() {

}

void RecordingElectrodes::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	threads_per_block.x = threads;

}




