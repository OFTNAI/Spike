#ifndef FourLayerVisionSpikingModel_H
#define FourLayerVisionSpikingModel_H

#include "SpikingModel.h"


class FourLayerVisionSpikingModel : public SpikingModel {

public:

	// Constructor/Destructor
	FourLayerVisionSpikingModel();
	~FourLayerVisionSpikingModel();

	// Network Parameters
	bool E2E_L_ON;
	bool E2E_FB_ON;
	bool LI_ON;

	bool E2E_L_STDP_ON;
	
	int number_of_non_input_layers;
	int dim_excit_layer;
	int dim_inhib_layer;


	// Layer-by-layer parameters
	int * LBL_max_number_of_connections_per_pair_E2E_FF;
	int * LBL_max_number_of_connections_per_pair_E2I_L;
	int * LBL_max_number_of_connections_per_pair_I2E_L;
	int * LBL_max_number_of_connections_per_pair_E2E_L;
	int * LBL_max_number_of_connections_per_pair_E2E_FB;

	int * LBL_fanInCount_E2E_FF;
	int * LBL_fanInCount_E2I_L;
	int * LBL_fanInCount_I2E_L;
	int * LBL_fanInCount_E2E_L;
	int * LBL_fanInCount_E2E_FB;

	float * LBL_gaussian_synapses_sd_E2E_FF;
	float * LBL_gaussian_synapses_sd_E2I_L;
	float * LBL_gaussian_synapses_sd_I2E_L;
	float * LBL_gaussian_synapses_sd_E2E_L;
	float * LBL_gaussian_synapses_sd_E2E_FB;

	float * LBL_biological_conductance_scaling_constant_lambda_E2E_FF;
	float * LBL_biological_conductance_scaling_constant_lambda_E2I_L;
	float * LBL_biological_conductance_scaling_constant_lambda_I2E_L;
	float * LBL_biological_conductance_scaling_constant_lambda_E2E_L;
	float * LBL_biological_conductance_scaling_constant_lambda_E2E_FB;

	float * LBL_decay_term_tau_g_E2E_FF;
	float * LBL_decay_term_tau_g_E2I_L;
	float * LBL_decay_term_tau_g_I2E_L;
	float * LBL_decay_term_tau_g_E2E_L;
	float * LBL_decay_term_tau_g_E2E_FB;

	// Neuronal Parameters
	float max_FR_of_input_Gabor;
	float absolute_refractory_period;


	//Synaptic Parameters
	float learning_rate_rho;//100.0;// 0.1;
	float decay_term_tau_C;
	float decay_term_tau_D;


	float E2E_FF_minDelay;
	float E2E_FF_maxDelay;
	float E2I_L_minDelay;
	float E2I_L_maxDelay;
	float I2E_L_minDelay;
	float I2E_L_maxDelay;
	float E2E_FB_minDelay;
	float E2E_FB_maxDelay;
	float E2E_L_minDelay;
	float E2E_L_maxDelay;



	LIFSpikingNeurons * lif_spiking_neurons;
	ImagePoissonInputSpikingNeurons * image_poisson_input_spiking_neurons;
	ConductanceSpikingSynapses * conductance_spiking_synapses;
	EvansSTDP * evans_stdp;



	vector<int> EXCITATORY_NEURONS;
	vector<int> INHIBITORY_NEURONS;


	void finalise_model(bool is_optimisation);


};

#endif