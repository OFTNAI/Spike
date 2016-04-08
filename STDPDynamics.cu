//	STDP Dynamics CUDA Code
//	STDPDynamics.cu
//
//	Author: Nasir Ahmad
//	Date: 15/03/2016

#include "STDPDynamics.h"
#include <stdlib.h>
#include <stdio.h>



// LTP on synapses
__global__ void synapseLTP(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t numConns,
							size_t numNeurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numConns) {
		// Get the synapses upon which we should do LTP
		// Reversed indexing to check post->pre connections
		if ((d_lastspiketime[d_postsyns[idx]] == currtime) && (d_stdp[idx] == 1)){
			// Get the last active time / weight of the synapse
			// Calc time difference and weight change
			float diff = currtime - d_lastactive[idx];
			float weightchange = (stdp_vars.w_max - d_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
			// Update weights
			d_weights[idx] += weightchange;
		}

	}
}