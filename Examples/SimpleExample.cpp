/*

  An Example Model for running the SPIKE simulator

  To create the executable for this network:
  - Run cmake from the build directory: "cmake ../"
  - Make this example: "make ExampleExperiment"
  - Finally, execute the binary: "./ExampleExperiment"


*/


#include "Spike/Spike.hpp"

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

  /*
      CHOOSE THE COMPONENTS OF YOUR SIMULATION
  */

  // Create an instance of the Model
  SpikingModel* ExampleModel = new SpikingModel();
    

  // Set up the simulator with a timestep at which the neuron, synapse and STDP properties will be calculated 
  float timestep = 0.0001;  // In seconds
  ExampleModel->SetTimestep(timestep);


  // Choose an input neuron type
  GeneratorInputSpikingNeurons* generator_input_neurons = new GeneratorInputSpikingNeurons();
  // PoissonInputSpikingNeurons* input neurons = new PoissonInputSpikingNeurons();

  // Choose your neuron type
  LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();

  // Choose your synapse type
  ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
  // VoltageSpikingSynapses * voltage_spiking_synapses = new VoltageSpikingSynapses();
  // CurrentSpikingSynapses * current_spiking_synapses = new CurrentSpikingSynapses();

  // Allocate your chosen components to the simulator
  ExampleModel->input_spiking_neurons = generator_input_neurons;
  ExampleModel->spiking_neurons = lif_spiking_neurons;
  ExampleModel->spiking_synapses = conductance_spiking_synapses;

  /*
      ADD ANY ACTIVITY MONITORS OR PLASTICITY RULES YOU WISH FOR 
  */
  SpikingActivityMonitor* spike_monitor = new SpikingActivityMonitor(lif_spiking_neurons);
  SpikingActivityMonitor* input_spike_monitor = new SpikingActivityMonitor(generator_input_neurons);
  ExampleModel->AddActivityMonitor(spike_monitor);
  ExampleModel->AddActivityMonitor(input_spike_monitor);

  /*
      SETUP PROPERTIES AND CREATE NETWORK:
    
    Note: 
    All Neuron, Synapse and STDP types have associated parameters structures.
    These structures are defined in the header file for that class and allow us to set properties.
  */

  // SETTING UP INPUT NEURONS
  // Creating an input neuron parameter structure
  generator_input_spiking_neuron_parameters_struct* input_neuron_params = new generator_input_spiking_neuron_parameters_struct();
  // Setting the dimensions of the input neuron layer
  input_neuron_params->group_shape[0] = 1;    // x-dimension of the input neuron layer
  input_neuron_params->group_shape[1] = 10;   // y-dimension of the input neuron layer
  // Create a group of input neurons. This function returns the ID of the input neuron group
  int input_layer_ID = ExampleModel->AddInputNeuronGroup(input_neuron_params);

  // SETTING UP NEURON GROUPS
  // Creating an LIF parameter structure for an excitatory neuron population and an inhibitory
  // 1 x 100 Layer
  lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
  excitatory_population_params->group_shape[0] = 1;
  excitatory_population_params->group_shape[1] = 100;
  excitatory_population_params->resting_potential_v0 = -0.074f;
  excitatory_population_params->threshold_for_action_potential_spike = -0.053f;
  excitatory_population_params->somatic_capacitance_Cm = 500.0*pow(10, -12);
  excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);

  lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
  inhibitory_population_params->group_shape[0] = 1;
  inhibitory_population_params->group_shape[1] = 100;
  inhibitory_population_params->resting_potential_v0 = -0.082f;
  inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
  inhibitory_population_params->somatic_capacitance_Cm = 214.0*pow(10, -12);
  inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

  // Create populations of excitatory and inhibitory neurons
  int excitatory_neuron_layer_ID = ExampleModel->AddNeuronGroup(excitatory_population_params);
  int inhibitory_neuron_layer_ID = ExampleModel->AddNeuronGroup(inhibitory_population_params);


  // SETTING UP SYNAPSES
  // Creating a synapses parameter structure for connections from the input neurons to the excitatory neurons
  conductance_spiking_synapse_parameters_struct* input_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
  input_to_excitatory_parameters->weight_range[0] = 0.5f;   // Create uniform distributions of weights [0.5, 10.0]
  input_to_excitatory_parameters->weight_range[1] = 10.0f;
  input_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
  input_to_excitatory_parameters->delay_range[0] = 8*timestep;    // Create uniform distributions of delays [1 timestep, 5 timesteps]
  input_to_excitatory_parameters->delay_range[1] = 8*timestep;
  // The connectivity types for synapses include:
    // CONNECTIVITY_TYPE_ALL_TO_ALL
    // CONNECTIVITY_TYPE_ONE_TO_ONE
    // CONNECTIVITY_TYPE_RANDOM
    // CONNECTIVITY_TYPE_PAIRWISE
  input_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
  //input_to_excitatory_parameters->plasticity_vec.push_back(STDP_RULE);

  // Creating a set of synapse parameters for connections from the excitatory neurons to the inhibitory neurons
  conductance_spiking_synapse_parameters_struct * excitatory_to_inhibitory_parameters = new conductance_spiking_synapse_parameters_struct();
  excitatory_to_inhibitory_parameters->weight_range[0] = 10.0f;
  excitatory_to_inhibitory_parameters->weight_range[1] = 10.0f;
  excitatory_to_inhibitory_parameters->weight_scaling_constant = inhibitory_population_params->somatic_leakage_conductance_g0;
  excitatory_to_inhibitory_parameters->delay_range[0] = 5.0*timestep;
  excitatory_to_inhibitory_parameters->delay_range[1] = 3.0f*pow(10, -3);
  excitatory_to_inhibitory_parameters->connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;

  // Creating a set of synapse parameters from the inhibitory neurons to the excitatory neurons
  conductance_spiking_synapse_parameters_struct * inhibitory_to_excitatory_parameters = new conductance_spiking_synapse_parameters_struct();
  inhibitory_to_excitatory_parameters->weight_range[0] = -5.0f;
  inhibitory_to_excitatory_parameters->weight_range[1] = -2.5f;
  inhibitory_to_excitatory_parameters->weight_scaling_constant = excitatory_population_params->somatic_leakage_conductance_g0;
  inhibitory_to_excitatory_parameters->delay_range[0] = 5.0*timestep;
  inhibitory_to_excitatory_parameters->delay_range[1] = 3.0f*pow(10, -3);
  inhibitory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
  

  // CREATING SYNAPSES
  // When creating synapses, the ids of the presynaptic and postsynaptic populations are all that are required
  // Note: Input neuron populations cannot be post-synaptic on any synapse
  ExampleModel->AddSynapseGroup(input_layer_ID, excitatory_neuron_layer_ID, input_to_excitatory_parameters);
  ExampleModel->AddSynapseGroup(excitatory_neuron_layer_ID, inhibitory_neuron_layer_ID, excitatory_to_inhibitory_parameters);
  ExampleModel->AddSynapseGroup(inhibitory_neuron_layer_ID, excitatory_neuron_layer_ID, inhibitory_to_excitatory_parameters);


  /*
      ADD INPUT STIMULI TO THE GENERATOR NEURONS CLASS
  */
  // We can now assign a set of spike times to neurons in the input layer
  int s1_num_spikes = 5;
  int s1_neuron_ids[5] = {0, 1, 3, 6, 7};
  float s1_spike_times[5] = {0.1f, 0.3f, 0.2f, 0.5f, 0.9f};
  // Adding this stimulus to the input neurons
  int first_stimulus = generator_input_neurons->add_stimulus(s1_num_spikes, s1_neuron_ids, s1_spike_times);
  // Creating a second stimulus
  int s2_num_spikes = 5;
  int s2_neuron_ids[5] = {2, 5, 9, 8, 0};
  float s2_spike_times[5] = {5.01f, 6.9f, 7.2f, 8.5f, 9.9f};
  int second_stimulus = generator_input_neurons->add_stimulus(s2_num_spikes, s2_neuron_ids, s2_spike_times);
  


  /*
      RUN THE SIMULATION
  */

  // The only argument to run is the number of seconds
  ExampleModel->finalise_model();
  float simtime = 50.0f;
  generator_input_neurons->select_stimulus(first_stimulus);
  ExampleModel->run(simtime);

  generator_input_neurons->select_stimulus(second_stimulus);
  ExampleModel->run(simtime);
  

  //spike_monitor->save_spikes_as_txt("./");
  input_spike_monitor->save_spikes_as_txt("./");
  ExampleModel->spiking_synapses->save_connectivity_as_txt("./");

  return 0;
}

