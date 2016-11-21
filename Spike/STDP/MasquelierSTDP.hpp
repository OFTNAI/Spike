// Masquelier STDP Class Header
// MasquelierSTDP.h
//
// This STDP learning rule is extracted from the following paper:

//	Timothee Masquelier, Rudy Guyonneau, and Simon J Thorpe. Spike timing 
//	dependent plasticity finds the start of repeating patterns in continuous spike
//	trains. PLoS One, 3(1):e1377, 2 January 2008.


// The default parameters are also those used in the above paper
//	Author: Nasir Ahmad
//	Date: 03/10/2016

#ifndef MASQUELIER_STDP_H
#define MASQUELIER_STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"
#include "../STDP/STDP.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

namespace Backend {
  class MasquelierSTDP : public virtual STDPCommon,
                    public STDP {
  public:
    virtual void prepare() {
      printf("TODO Backend::MasquelierSTDP::prepare\n");
    }
  };
}

#include "Spike/Backend/Dummy/STDP/MasquelierSTDP.hpp"

// STDP Parameters
struct masquelier_stdp_parameters_struct : stdp_parameters_struct {
	masquelier_stdp_parameters_struct() : a_minus(0.85f*0.03125f), a_plus(0.03125f), tau_minus(0.033f), tau_plus(0.0168f) { } // default Constructor
	// STDP Parameters
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;

};


class MasquelierSTDP : public STDP{
public:
	struct masquelier_stdp_parameters_struct* stdp_params = NULL;
	SpikingSynapses* syns = NULL;
	SpikingNeurons* neurs = NULL;
	int* index_of_last_afferent_synapse_to_spike = NULL;
	bool* isindexed_ltd_synapse_spike = NULL;
	int* index_of_first_synapse_spiked_after_postneuron = NULL;

	virtual void reset_state();
	// Set STDP Parameters
	virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters);
	// STDP
	virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep);
	// LTP & LTD for this model
	void apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);

};

/*CUDA
// Kernel to carry out LTP/LTD
__global__ void apply_stdp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* d_index_of_last_afferent_synapse_to_spike,
							bool* d_isindexed_ltd_synapse_spike,
							int* d_index_of_first_synapse_spiked_after_postneuron,
							struct masquelier_stdp_parameters_struct stdp_vars,
							float currtime,
							size_t total_number_of_post_neurons);

__global__ void get_indices_to_apply_stdp(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_index_of_last_afferent_synapse_to_spike,
							bool* d_isindexed_ltd_synapse_spike,
							int* d_index_of_first_synapse_spiked_after_postneuron,
							float currtime,
							size_t total_number_of_synapse);

*/
#endif
