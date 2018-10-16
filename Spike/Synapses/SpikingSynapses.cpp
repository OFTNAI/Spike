#include "SpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

SpikingSynapses::~SpikingSynapses() {
#ifdef CRAZY_DEBUG
  std::cout << "SpikingSynapses::~SpikingSynapses\n";
#endif
  free(delays);

}

void SpikingSynapses::prepare_backend_early() {
  Synapses::prepare_backend_early();
  Synapses::sort_synapses(model->input_spiking_neurons, model->spiking_neurons);
  SpikingSynapses::sort_synapses();
  
  // Setting Neuron and InputNeuron start indices
  set_synapse_start(presynaptic_neuron_indices[0], 0);
  for (int s=1; s < total_number_of_synapses; s++){
    if (presynaptic_neuron_indices[s-1] != presynaptic_neuron_indices[s])
      set_synapse_start(presynaptic_neuron_indices[s], s); 
  }
}

void SpikingSynapses::set_synapse_start(int pre_index, int syn_start){
  bool is_presynaptic = PRESYNAPTIC_IS_INPUT(pre_index);
  int corr_pre_index = CORRECTED_PRESYNAPTIC_ID(pre_index, is_presynaptic);

  int* neuron_start_indices = is_presynaptic ? model->input_spiking_neurons->per_neuron_efferent_synapse_start : model->spiking_neurons->per_neuron_efferent_synapse_start;

  neuron_start_indices[corr_pre_index] = syn_start;
}


void SpikingSynapses::sort_synapses(){
  
  int* temp_delay_array = (int*)malloc(total_number_of_synapses * sizeof(int));
  int* temp_synlabel_array = (int*)malloc(total_number_of_synapses * sizeof(int));
  // Re-ordering arrays
  for (int s=0; s < total_number_of_synapses; s++){
    temp_delay_array[s] = delays[synapse_sort_indices[s]];
    temp_synlabel_array[s] = syn_labels[synapse_sort_indices[s]];
  }

  free(delays);
  free(syn_labels);

  delays = temp_delay_array;
  syn_labels = temp_synlabel_array;
}

// Connection Detail implementation
//  INPUT:
//    Pre-neuron population ID
//    Post-neuron population ID
//    An array of the exclusive sum of neuron populations
//    CONNECTIVITY_TYPE (Constants.h)
//    2 number float array for weight range
//    2 number float array for delay range
//    Boolean value to indicate if population is STDP based
//    Parameter = either probability for random synapses or S.D. for Gaussian
int SpikingSynapses::AddGroup(int presynaptic_group_id, 
            int postsynaptic_group_id, 
            Neurons * neurons,
            Neurons * input_neurons,
            float timestep,
            synapse_parameters_struct * synapse_params) {
  
  
  int groupID = Synapses::AddGroup(presynaptic_group_id, 
              postsynaptic_group_id, 
              neurons,
              input_neurons,
              timestep,
              synapse_params);

  // First incrementing the synapses
  SpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

  spiking_synapse_parameters_struct * spiking_synapse_group_params = (spiking_synapse_parameters_struct*)synapse_params;

  // Convert delay range from time to number of timesteps
  int delay_range_in_timesteps[2] = {int(round(spiking_synapse_group_params->delay_range[0]/timestep)), int(round(spiking_synapse_group_params->delay_range[1]/timestep))};

  // Check delay range bounds greater than timestep
  if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
    printf("%d\n", delay_range_in_timesteps[0]);
    printf("%d\n", delay_range_in_timesteps[1]);
#ifdef CRAZY_DEBUG
                // spiking_synapse_group_params->delay_range[0] = timestep;
                // spiking_synapse_group_params->delay_range[1] = timestep;
    printf("################### Delay range must be at least one timestep\n");
#else

        
    print_message_and_exit("Delay range must be at least one timestep.");
#endif
  }
  
  for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++){
    // Setup Delays
    float delayval = delay_range_in_timesteps[0];
    if (delay_range_in_timesteps[0] != delay_range_in_timesteps[1])
      delayval = delay_range_in_timesteps[0] + (delay_range_in_timesteps[1] - delay_range_in_timesteps[0]) * ((float)rand() / (RAND_MAX));
    delays[i] = round(delayval);
    syn_labels[i] = 0; // Conductance or other systems can now use this if they wish
    if (spiking_synapse_group_params->connectivity_type == CONNECTIVITY_TYPE_PAIRWISE){
      if (spiking_synapse_group_params->pairwise_connect_delay.size() == temp_number_of_synapses_in_last_group){
        delays[i] = (int)round(spiking_synapse_group_params->pairwise_connect_delay[i + temp_number_of_synapses_in_last_group - total_number_of_synapses] / timestep);
      } else if (spiking_synapse_group_params->pairwise_connect_delay.size() != 0) {
        print_message_and_exit("PAIRWISE CONNECTION ISSUE: Delay vector length not as expected. Should be the same length as pre/post vecs.");
      }
    }
    
    // Ensure max/min delays are set correctly
    if (delays[i] > maximum_axonal_delay_in_timesteps) maximum_axonal_delay_in_timesteps = delays[i];
    if (delays[i] < minimum_axonal_delay_in_timesteps) minimum_axonal_delay_in_timesteps = delays[i];
  }
  if (neurons->total_number_of_neurons > neuron_pop_size)
    neuron_pop_size = neurons->total_number_of_neurons; 

  return groupID;

}

void SpikingSynapses::increment_number_of_synapses(int increment) {
  delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
  syn_labels = (int*)realloc(syn_labels, total_number_of_synapses * sizeof(int));
}


void SpikingSynapses::state_update(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
  backend()->state_update(neurons, input_neurons, current_time_in_seconds, timestep);
}

void SpikingSynapses::save_connectivity_as_txt(std::string path, std::string prefix, int synapsegroupid){
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }
  Synapses::save_connectivity_as_txt(path, prefix, synapsegroupid);
  std::ofstream delayfile;

  // Open output files
  delayfile.open((path + "/" + prefix + "SynapticDelays.txt"), std::ios::out | std::ios::binary);

  // Ensure weight data has been copied to frontend
  if (_backend)
    backend()->copy_to_frontend();

  // Send data to file
  for (int i = startid; i < endid; i++){
    delayfile << delays[synapse_reversesort_indices[i]] << std::endl;
  }

  // Close files
  delayfile.close();

};
// Ensure copied from device, then send
void SpikingSynapses::save_connectivity_as_binary(std::string path, std::string prefix, int synapsegroupid){
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }
  Synapses::save_connectivity_as_binary(path, prefix, synapsegroupid);
  std::ofstream delayfile;

  // Open output files
  delayfile.open((path + "/" + prefix + "SynapticDelays.bin"), std::ios::out | std::ios::binary);

  // Ensure weight data has been copied to frontend
  if (_backend)
    backend()->copy_to_frontend();

  // Send data to file
  for (int i = startid; i < endid; i++){
    delayfile.write((char *)&delays[synapse_reversesort_indices[i]], sizeof(int));
  }

  // Close files
  delayfile.close();
}

SPIKE_MAKE_INIT_BACKEND(SpikingSynapses);
