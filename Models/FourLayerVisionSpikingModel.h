#ifndef FourLayerVisionSpikingModel_H
#define FourLayerVisionSpikingModel_H

#include <cuda.h>
#include <stdio.h>
#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
#include <string>
#include <fstream>
#include "../Plotting/Plotter.h"
#include <vector>

#include <iostream>
using namespace std;


class FourLayerVisionSpikingModel : public Neurons {

public:

	// Constructor/Destructor
	FourLayerVisionSpikingModel(float timestep);
	~FourLayerVisionSpikingModel();

	bool E2E_L_ON;
	bool E2E_FB_ON;
	bool E2E_L_STDP_ON;

	// Network Parameters
	const int number_of_layers;
	int max_number_of_connections_per_pair;
	int dim_excit_layer;
	int dim_inhib_layer;

	int fanInCount_G2E_FF;
	int fanInCount_E2E_FF;
	int fanInCount_E2I_L;
	int fanInCount_I2E_L;
	int fanInCount_E2E_L;
	int fanInCount_E2E_FB;

	float gaussian_synapses_standard_deviation_G2E_FF;
	float gaussian_synapses_standard_deviation_E2E_FF[number_of_layers-1] = {8.0, 12.0, 16.0};
	float gaussian_synapses_standard_deviation_E2I_L;
	float gaussian_synapses_standard_deviation_I2E_L;
	float gaussian_synapses_standard_deviation_E2E_L;
	float gaussian_synapses_standard_deviation_E2E_FB;

	float biological_conductance_scaling_constant_lambda_G2E_FF;
	float biological_conductance_scaling_constant_lambda_E2E_FF;
	float biological_conductance_scaling_constant_lambda_E2I_L;
	float biological_conductance_scaling_constant_lambda_I2E_L;
	float biological_conductance_scaling_constant_lambda_E2E_L;
	float biological_conductance_scaling_constant_lambda_E2E_FB;

	float decay_term_tau_g_G2E_FF;
	float decay_term_tau_g_E2E_FF;
	float decay_term_tau_g_E2I_L;
	float decay_term_tau_g_I2E_L;
	float decay_term_tau_g_E2E_L;
	float decay_term_tau_g_E2E_FB;

	// Neuronal Parameters
	float max_FR_of_input_Gabor;
	float absolute_refractory_period;


	//Synaptic Parameters
	float weight_range_bottom;
	float weight_range_top;
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
	ImagePoissonInputSpikingNeurons * input_neurons;
	ConductanceSpikingSynapses * conductance_spiking_synapses;
	EvansSTDP * evans_stdp;





	vector<int> EXCITATORY_NEURONS;
	vector<int> INHIBITORY_NEURONS;


};

#endif