// Vogels Abbot Benchmark Network
// Author: Nasir Ahmad (Created: 16/09/2016)

/*
  This network has been created to benchmark Spike. It shall follow the network
  used to analyse Auryn.

  Publications:
  Vogels, Tim P., and L. F. Abbott. 2005. “Signal Propagation and Logic Gating in Networks of Integrate-and-Fire Neurons.” The Journal of Neuroscience: The Official Journal of the Society for Neuroscience 25 (46): 10786–95.
  Zenke, Friedemann, and Wulfram Gerstner. 2014. “Limits to High-Speed Simulations of Spiking Neural Networks Using General-Purpose Computers.” Frontiers in Neuroinformatics 8 (August). Frontiers. doi:10.3389/fninf.2014.00076.

*/

// Including the primary Spike header gives access to all models and components
#include "Spike/Spike.hpp"
#include "Spike/Backend/CUDA/Helpers/ErrorCheck.hpp"
// Utility functions in case you want to load from .mat file
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
#include <cstdio>

int main (int argc, char *argv[]){
  /*
   *    GETTING COMMAND LINE ARGUMENTS
   */
  // These are separate from Spike but allow run-time changes to simulations
  // The defaults are that simulations are for 20s and the delay is 0.8ms
  std::stringstream ss;
  float simtime = 20.0;
  bool fast = false;
  bool NOTG = false;
  int num_timesteps_min_delay = 8;
  int num_timesteps_max_delay = 8;
  const char* const short_opts = "";
  const option long_opts[] = {
    {"simtime", 1, nullptr, 0},
    {"fast", 0, nullptr, 1},
    {"num_timesteps_min_delay", 1, nullptr, 2},
    {"num_timesteps_max_delay", 1, nullptr, 3},
    {"NOTG", 0, nullptr, 4}
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
      case 2:
        printf("Running with minimum delay: %s timesteps\n", optarg);
        ss << optarg;
        ss >> num_timesteps_min_delay;
        ss.clear();
        if (num_timesteps_max_delay < num_timesteps_min_delay)
          num_timesteps_max_delay = num_timesteps_min_delay;
        break;
      case 3:
        printf("Running with maximum delay: %s timesteps\n", optarg);
        ss << optarg;
        ss >> num_timesteps_max_delay;
        ss.clear();
        if (num_timesteps_max_delay < num_timesteps_min_delay){
          std::cerr << "ERROR: Max timestep shouldn't be smaller than min!" << endl;
          exit(1);
        } 
        break;
      case 4:
        printf("No timestep grouping \n");
        NOTG = true;
        break;
      default:
        break;
    }
  };
  
  /*
   *    CREATING A SPIKINGMODEL
   */
  // The SpikingModel object manages neurons, synapses and plasticity rules
  SpikingModel * BenchModel = new SpikingModel();
  // Any changes to the timestep should be done before anything else
  float timestep = 0.0001f; // in seconds
  BenchModel->SetTimestep(timestep);

  /*
   *    NEURON AND SYNAPSE TYPE CHOICE
   */
  // Create neuron, synapse and plasticity components to this model
  LIFSpikingNeurons * lif_spiking_neurons = new LIFSpikingNeurons();
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
  // Add component choices to the model
  BenchModel->spiking_neurons = lif_spiking_neurons;
  BenchModel->spiking_synapses = conductance_spiking_synapses;

  // Add a monitor for Neuron Spiking
  SpikingActivityMonitor* spike_monitor = new SpikingActivityMonitor(lif_spiking_neurons);
  if (!fast)
    BenchModel->AddActivityMonitor(spike_monitor);

  /*
   *    NEURON PARAMETER SETUP
   */
  lif_spiking_neuron_parameters_struct * EXC_NEURON_PARAMS = new lif_spiking_neuron_parameters_struct();
  lif_spiking_neuron_parameters_struct * INH_NEURON_PARAMS = new lif_spiking_neuron_parameters_struct();

  EXC_NEURON_PARAMS->somatic_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF
  INH_NEURON_PARAMS->somatic_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF

  EXC_NEURON_PARAMS->somatic_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS
  INH_NEURON_PARAMS->somatic_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS

  EXC_NEURON_PARAMS->resting_potential_v0 = -60.0f*pow(10.0, -3);
  INH_NEURON_PARAMS->resting_potential_v0 = -60.0f*pow(10.0, -3);
  
  EXC_NEURON_PARAMS->after_spike_reset_potential_vreset = -60.0f*pow(10.0, -3);
  INH_NEURON_PARAMS->after_spike_reset_potential_vreset = -60.0f*pow(10.0, -3);

  EXC_NEURON_PARAMS->absolute_refractory_period = 5.0f*pow(10, -3);  // ms
  INH_NEURON_PARAMS->absolute_refractory_period = 5.0f*pow(10, -3);  // ms

  EXC_NEURON_PARAMS->threshold_for_action_potential_spike = -50.0f*pow(10.0, -3); // -53mV threshold
  INH_NEURON_PARAMS->threshold_for_action_potential_spike = -50.0f*pow(10.0, -3); // -53mV threshold

  EXC_NEURON_PARAMS->background_current = 2.0f*pow(10.0, -2); //
  INH_NEURON_PARAMS->background_current = 2.0f*pow(10.0, -2); //

  /*
   *    CREATE NEURON POPULATIONS
   */
  vector<int> EXCITATORY_NEURONS;
  vector<int> INHIBITORY_NEURONS;
  // Creating a single exc and inh population for now
  EXC_NEURON_PARAMS->group_shape[0] = 1;
  EXC_NEURON_PARAMS->group_shape[1] = 3200;
  INH_NEURON_PARAMS->group_shape[0] = 1;
  INH_NEURON_PARAMS->group_shape[1] = 800;
  EXCITATORY_NEURONS.push_back(BenchModel->AddNeuronGroup(EXC_NEURON_PARAMS));
  INHIBITORY_NEURONS.push_back(BenchModel->AddNeuronGroup(INH_NEURON_PARAMS));

  /*
   *    SYNAPSE PARAMETER SETUP
   */
  conductance_spiking_synapse_parameters_struct * EXC_OUT_SYN_PARAMS = new conductance_spiking_synapse_parameters_struct();
  conductance_spiking_synapse_parameters_struct * INH_OUT_SYN_PARAMS = new conductance_spiking_synapse_parameters_struct();
  // Setting delays
  EXC_OUT_SYN_PARAMS->delay_range[0] = num_timesteps_min_delay*timestep;
  EXC_OUT_SYN_PARAMS->delay_range[1] = num_timesteps_max_delay*timestep;
  INH_OUT_SYN_PARAMS->delay_range[0] = num_timesteps_min_delay*timestep;
  INH_OUT_SYN_PARAMS->delay_range[1] = num_timesteps_max_delay*timestep;
  // Setting Reversal Potentials for excitatory vs inhibitory synapses
  EXC_OUT_SYN_PARAMS->reversal_potential_Vhat = 0.0f*pow(10.0, -3);
  INH_OUT_SYN_PARAMS->reversal_potential_Vhat = -80.0f*pow(10.0, -3);
  // Set Weight Range?
  EXC_OUT_SYN_PARAMS->weight_range[0] = 0.4f;
  EXC_OUT_SYN_PARAMS->weight_range[1] = 0.4f;
  INH_OUT_SYN_PARAMS->weight_range[0] = 5.1f;
  INH_OUT_SYN_PARAMS->weight_range[1] = 5.1f;
  // Set timescales
  EXC_OUT_SYN_PARAMS->decay_term_tau_g = 5.0f*pow(10.0, -3);  // 5ms
  INH_OUT_SYN_PARAMS->decay_term_tau_g = 10.0f*pow(10.0, -3);  // 10ms

  // Biological Scaling factors -> This sets the scale of the weights
  EXC_OUT_SYN_PARAMS->weight_scaling_constant = 10.0f*pow(10.0,-9);
  INH_OUT_SYN_PARAMS->weight_scaling_constant = 10.0f*pow(10.0,-9);

  // Connect neurons randomly with a 2% probability
  EXC_OUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
  INH_OUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
  EXC_OUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);
  INH_OUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);
  EXC_OUT_SYN_PARAMS->random_connectivity_probability = 0.02; // 2%
  INH_OUT_SYN_PARAMS->random_connectivity_probability = 0.02; // 2%

  /*
   *    CREATE SYNAPSES
   */
  BenchModel->AddSynapseGroup(EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0], EXC_OUT_SYN_PARAMS);
  BenchModel->AddSynapseGroup(EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0], EXC_OUT_SYN_PARAMS);
  BenchModel->AddSynapseGroup(INHIBITORY_NEURONS[0], EXCITATORY_NEURONS[0], INH_OUT_SYN_PARAMS);
  BenchModel->AddSynapseGroup(INHIBITORY_NEURONS[0], INHIBITORY_NEURONS[0], INH_OUT_SYN_PARAMS);
  
  /*
  // Adding connections based upon matrices given
  connect_from_mat(
    EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0],
    EXC_OUT_SYN_PARAMS, 
    "../../ee.wmat",
    BenchModel);
  connect_from_mat(
    EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0],
    EXC_OUT_SYN_PARAMS, 
    "../../ei.wmat",
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

  /*
   *    RUN THIS SIMULATION
   */
  BenchModel->finalise_model();
  if (NOTG)
    BenchModel->timestep_grouping = 1;

  clock_t starttime = clock();
  BenchModel->run(simtime);
  clock_t totaltime = clock() - starttime;
  if ( fast ){
    std::ofstream timefile;
    timefile.open("timefile.dat");
    timefile << std::setprecision(10) << ((float)totaltime / CLOCKS_PER_SEC);
    timefile.close();
  } else {
    //spike_monitor->save_spikes_as_txt("./");
    spike_monitor->save_spikes_as_binary("./", "VA");
  }

  //BenchModel->spiking_synapses->save_connectivity_as_binary("./");
  //BenchModel->spiking_synapses->save_connectivity_as_txt("./");
  return(0);
}
