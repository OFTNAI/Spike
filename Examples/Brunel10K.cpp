// Brunel 10,000 Neuron Network with Plasticity
// Author: Nasir Ahmad (Created: 03/05/2018)

/*
  This network has been created to benchmark Spike

  Original Publication:
  Brunel N. Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. J Comput Neurosci. 2000;8: 183–208.

*/


// Include the Spike hpp for access to all model components
#include "Spike/Spike.hpp"
#include "UtilityFunctions.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <getopt.h>
#include <time.h>
#include <iomanip>
#include <vector>
#include <stdlib.h>


int main (int argc, char *argv[]){
  /* 
   * Getting Command Line Arguments
   */
  float simtime = 20.0;
  float sparseness = 0.1;
  bool fast = false;
  bool plastic = false;
  std::stringstream ss;
  const char* const short_opts = "";
  const option long_opts[] = {
    {"simtime", 1, nullptr, 0},
    {"fast", 0, nullptr, 1},
    {"plastic", 0, nullptr, 4},
  };
  // Check the set of options
  while (true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

    // If none
    if (-1 == opt) break;

    switch (opt){
      case 0:
        printf("Running with a simulation time of: %ss\n", optarg);
        ss << optarg;
        ss >> simtime;
        ss.clear();
        break;
      case 1:
        printf("Running in fast mode (no spike collection)\n");
        fast = true;
        break;
      case 4:
        printf("Running with plasticity ON\n");
        plastic = true;
        break;
    }
  };
  
  // The details below shall be used in a SpikingModel
  SpikingModel * BenchModel = new SpikingModel();
  // Since the timestep is used during model set-up, it should be set first and not changed
  float timestep = 0.0001f; // 50us for now
  BenchModel->SetTimestep(timestep);
  float delayval = 1.5f*powf(10.0, -3.0); // 1.5ms

  // Create neuron, synapse and stdp types for this model
  LIFSpikingNeurons * lif_spiking_neurons = new LIFSpikingNeurons();
  PoissonInputSpikingNeurons * poisson_input_spiking_neurons = new PoissonInputSpikingNeurons();
  // The synapses constructor has an optional constructor argument (integer) to use as the seed for random synapses
  VoltageSpikingSynapses * voltage_spiking_synapses = new VoltageSpikingSynapses(42);

  // STDP must be instantiated with the settings
  weightdependent_stdp_plasticity_parameters_struct * WDSTDP_PARAMS = new weightdependent_stdp_plasticity_parameters_struct;
  WDSTDP_PARAMS->a_plus = 1.0;
  WDSTDP_PARAMS->a_minus = 1.0;
  WDSTDP_PARAMS->tau_plus = 0.02;
  WDSTDP_PARAMS->tau_minus = 0.02;
  WDSTDP_PARAMS->lambda = 1.0f*powf(10.0, -2);
  WDSTDP_PARAMS->alpha = 2.02;
  WDSTDP_PARAMS->w_max = 0.3*powf(10.0, -3);
  //WDSTDP_PARAMS->nearest_spike_only = true;
  WeightDependentSTDPPlasticity * weightdependent_stdp = new WeightDependentSTDPPlasticity((SpikingSynapses *) voltage_spiking_synapses, (SpikingNeurons *)lif_spiking_neurons, (SpikingNeurons *) poisson_input_spiking_neurons, (stdp_plasticity_parameters_struct *) WDSTDP_PARAMS);

  // Add my components to the SpikingModel
  // Note: Spike only supports a single input neuron, neuron and synapse type for now. Multiple Plasticity rules can be added
  BenchModel->spiking_neurons = lif_spiking_neurons;
  BenchModel->input_spiking_neurons = poisson_input_spiking_neurons;
  BenchModel->spiking_synapses = voltage_spiking_synapses;
  if (plastic)
    BenchModel->AddPlasticityRule(weightdependent_stdp);

  // Adding Spike Detectors
  SpikingActivityMonitor* spike_monitor = new SpikingActivityMonitor(lif_spiking_neurons);
  if (!fast){
    BenchModel->AddActivityMonitor(spike_monitor);
  }

  // Set up Neuron Parameters
  lif_spiking_neuron_parameters_struct * EXC_NEURON_PARAMS = new lif_spiking_neuron_parameters_struct();
  lif_spiking_neuron_parameters_struct * INH_NEURON_PARAMS = new lif_spiking_neuron_parameters_struct();

  EXC_NEURON_PARAMS->somatic_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF
  INH_NEURON_PARAMS->somatic_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF

  EXC_NEURON_PARAMS->somatic_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS
  INH_NEURON_PARAMS->somatic_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS

  EXC_NEURON_PARAMS->resting_potential_v0 = 0.0f*pow(10.0, -3); // -74mV
  INH_NEURON_PARAMS->resting_potential_v0 = 0.0f*pow(10.0, -3); // -82mV
  
  EXC_NEURON_PARAMS->after_spike_reset_potential_vreset = 0.0f*pow(10.0, -3);
  INH_NEURON_PARAMS->after_spike_reset_potential_vreset = 0.0f*pow(10.0, -3);

  EXC_NEURON_PARAMS->absolute_refractory_period = 0.0f*pow(10, -3);  // ms
  INH_NEURON_PARAMS->absolute_refractory_period = 0.0f*pow(10, -3);  // ms

  EXC_NEURON_PARAMS->threshold_for_action_potential_spike = 20.0f*pow(10.0, -3);
  INH_NEURON_PARAMS->threshold_for_action_potential_spike = 20.0f*pow(10.0, -3);

  EXC_NEURON_PARAMS->background_current = 0.0f*pow(10.0, -2); //
  INH_NEURON_PARAMS->background_current = 0.0f*pow(10.0, -2); //

  /*
    Setting up INPUT NEURONS
  */
  // Creating an input neuron parameter structure
  poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new poisson_input_spiking_neuron_parameters_struct();
  // Setting the dimensions of the input neuron layer
  input_neuron_params->group_shape[0] = 1;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = 10000;    // y-dimension of the input neuron layer
  input_neuron_params->rate = 20.0f; // Hz
  int input_layer_ID = BenchModel->AddInputNeuronGroup(input_neuron_params);

  /*
    Setting up NEURON POPULATION
  */
  vector<int> EXCITATORY_NEURONS;
  vector<int> INHIBITORY_NEURONS;
  // Creating a single exc and inh population for now
  EXC_NEURON_PARAMS->group_shape[0] = 1;
  EXC_NEURON_PARAMS->group_shape[1] = 8000;
  INH_NEURON_PARAMS->group_shape[0] = 1;
  INH_NEURON_PARAMS->group_shape[1] = 2000;
  EXCITATORY_NEURONS.push_back(BenchModel->AddNeuronGroup(EXC_NEURON_PARAMS));
  INHIBITORY_NEURONS.push_back(BenchModel->AddNeuronGroup(INH_NEURON_PARAMS));

  /*
    Setting up SYNAPSES
  */
  voltage_spiking_synapse_parameters_struct * EXC_OUT_SYN_PARAMS = new voltage_spiking_synapse_parameters_struct();
  voltage_spiking_synapse_parameters_struct * INH_OUT_SYN_PARAMS = new voltage_spiking_synapse_parameters_struct();
  voltage_spiking_synapse_parameters_struct * INPUT_SYN_PARAMS = new voltage_spiking_synapse_parameters_struct();
  // Setting delays
  EXC_OUT_SYN_PARAMS->delay_range[0] = delayval;
  EXC_OUT_SYN_PARAMS->delay_range[1] = delayval;
  INH_OUT_SYN_PARAMS->delay_range[0] = delayval;
  INH_OUT_SYN_PARAMS->delay_range[1] = delayval;
  INPUT_SYN_PARAMS->delay_range[0] = delayval;
  INPUT_SYN_PARAMS->delay_range[1] = delayval;
  // Set Weight Range (in mVs)
  float weight_val = 0.1f*powf(10.0, -3.0);
  float gamma = 5.0f;
  EXC_OUT_SYN_PARAMS->weight_range[0] = weight_val;
  EXC_OUT_SYN_PARAMS->weight_range[1] = weight_val;
  INH_OUT_SYN_PARAMS->weight_range[0] = -gamma * weight_val;
  INH_OUT_SYN_PARAMS->weight_range[1] = -gamma * weight_val;
  INPUT_SYN_PARAMS->weight_range[0] = weight_val;
  INPUT_SYN_PARAMS->weight_range[1] = weight_val;

  // Biological Scaling factors (ensures that voltage is in mV)
  float weight_multiplier = 1.0; //powf(10.0, -3.0);
  EXC_OUT_SYN_PARAMS->weight_scaling_constant = weight_multiplier;
  INH_OUT_SYN_PARAMS->weight_scaling_constant = weight_multiplier;
  INPUT_SYN_PARAMS->weight_scaling_constant = weight_multiplier;

  connect_with_sparsity(
      input_layer_ID, EXCITATORY_NEURONS[0],
      input_neuron_params, EXC_NEURON_PARAMS,
      INPUT_SYN_PARAMS, sparseness,
      BenchModel);
  connect_with_sparsity(
      input_layer_ID, INHIBITORY_NEURONS[0],
      input_neuron_params, INH_NEURON_PARAMS,
      INPUT_SYN_PARAMS, sparseness,
      BenchModel);

  /*
  connect_from_mat(
    EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0],
    EXC_OUT_SYN_PARAMS, 
    "../../ei.wmat",
    BenchModel);


  if (plastic)
    EXC_OUT_SYN_PARAMS->plasticity_vec.push_back(weightdependent_stdp);
  connect_from_mat(
    EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0],
    EXC_OUT_SYN_PARAMS, 
    "../../ee.wmat",
    BenchModel);
  connect_from_mat(
    INHIBITORY_NEURONS[0], EXCITATORY_NEURONS[0],
    INH_OUT_SYN_PARAMS, 
    "../../ie.wmat",
    BenchModel);
  connect_from_mat(
    INHIBITORY_NEURONS[0], INHIBITORY_NEURONS[0],
    INH_OUT_SYN_PARAMS, 
    "../../ii.wmat",
    BenchModel);
  */
  // Creating Synapse Populations
  connect_with_sparsity(
      EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0],
      EXC_NEURON_PARAMS, INH_NEURON_PARAMS,
      EXC_OUT_SYN_PARAMS, sparseness,
      BenchModel);
  if (plastic)
    EXC_OUT_SYN_PARAMS->plasticity_vec.push_back(weightdependent_stdp);
  connect_with_sparsity(
      EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0],
      EXC_NEURON_PARAMS, EXC_NEURON_PARAMS,
      EXC_OUT_SYN_PARAMS, sparseness,
      BenchModel);
  connect_with_sparsity(
      INHIBITORY_NEURONS[0], EXCITATORY_NEURONS[0],
      INH_NEURON_PARAMS, EXC_NEURON_PARAMS,
      INH_OUT_SYN_PARAMS, sparseness,
      BenchModel);
  connect_with_sparsity(
      INHIBITORY_NEURONS[0], INHIBITORY_NEURONS[0],
      INH_NEURON_PARAMS, INH_NEURON_PARAMS,
      INH_OUT_SYN_PARAMS, sparseness,
      BenchModel);
  connect_with_sparsity(
      input_layer_ID, EXCITATORY_NEURONS[0],
      input_neuron_params, EXC_NEURON_PARAMS,
      INPUT_SYN_PARAMS, sparseness,
      BenchModel);
  connect_with_sparsity(
      input_layer_ID, INHIBITORY_NEURONS[0],
      input_neuron_params, INH_NEURON_PARAMS,
      INPUT_SYN_PARAMS, sparseness,
      BenchModel);


    //simoptions->recording_electrodes_options->collect_neuron_spikes_optional_parameters->human_readable_storage = true;
    //simoptions->recording_electrodes_options->network_state_archive_recording_electrodes_bool = true;



  /*
    RUN SIMULATION
  */
  clock_t starttime = clock();
  BenchModel->run(simtime);
  clock_t totaltime = clock() - starttime;
  if ( fast ){
    std::ofstream timefile;
    std::string filename = "timefile.dat";
    timefile.open(filename);
    timefile << std::setprecision(10) << ((float)totaltime / CLOCKS_PER_SEC);
    timefile.close();
  }
  //cudaProfilerStop();
  return(0);
}
