#pragma once


__device__ void my_activate_synapses(
  spiking_synapses_data_struct* synaptic_data,
  spiking_neurons_data_struct* neuron_data,
  int timestep_group_index,
  int preneuron_idx,
  bool is_input)
{
  int synapse_count = neuron_data->per_neuron_efferent_synapse_count[preneuron_idx];
  atomicMax(synaptic_data->num_active_synapses, synapse_count);
  int pos = atomicAdd(synaptic_data->num_activated_neurons, 1);
  synaptic_data->active_synapse_counts[pos] = synapse_count;
  synaptic_data->active_presynaptic_neuron_indices[pos] = CORRECTED_PRESYNAPTIC_ID(preneuron_idx, is_input);
  synaptic_data->group_indices[pos] = timestep_group_index;
};



__device__ float my_conductance_spiking_injection_kernel(
    spiking_synapses_data_struct* in_synaptic_data,
    spiking_neurons_data_struct* neuron_data,
    float multiplication_to_volts,
    float current_membrane_voltage,
    float current_time_in_seconds,
    float timestep,
    int timestep_grouping,
    int idx,
    int g){
  
  conductance_spiking_synapses_data_struct* synaptic_data = (conductance_spiking_synapses_data_struct*) in_synaptic_data;
    
  int bufferloc = ((synaptic_data->neuron_inputs.bufferloc[0] + g) % synaptic_data->neuron_inputs.temporal_buffersize)*synaptic_data->neuron_inputs.input_buffersize;
    
  float total_current = 0.0f;
  for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
    float decay_factor = synaptic_data->decay_factors_g[syn_label];
    float reversal_value = synaptic_data->reversal_potentials_Vhat[syn_label];
    float synaptic_conductance_g = synaptic_data->neuron_wise_conductance_trace[syn_label + idx*synaptic_data->num_syn_labels];
    // Update the synaptic conductance
    synaptic_conductance_g = decay_factor*synaptic_conductance_g + synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_syn_labels];
    // Reset the conductance update
    synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_syn_labels] = 0.0f;
    total_current += synaptic_conductance_g*(reversal_value - current_membrane_voltage);

    synaptic_data->neuron_wise_conductance_trace[syn_label + idx*synaptic_data->num_syn_labels] = synaptic_conductance_g;

  }
  return total_current*multiplication_to_volts;
};

__device__ float my_current_spiking_injection_kernel(
    spiking_synapses_data_struct* in_synaptic_data,
    spiking_neurons_data_struct* neuron_data,
    float multiplication_to_volts,
    float current_membrane_voltage,
    float current_time_in_seconds,
    float timestep,
    int timestep_grouping,
    int idx,
    int g){
  
  current_spiking_synapses_data_struct* synaptic_data = (current_spiking_synapses_data_struct*) in_synaptic_data;
    
  int total_number_of_neurons =  neuron_data->total_number_of_neurons;
  int bufferloc = ((synaptic_data->neuron_inputs.bufferloc[0] + g) % synaptic_data->neuron_inputs.temporal_buffersize)*synaptic_data->neuron_inputs.input_buffersize;
  float total_current = 0.0f;
    for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
      float decay_term_value = synaptic_data->decay_terms_tau[syn_label];
      float decay_factor = expf(- timestep / decay_term_value);
      float synaptic_current = synaptic_data->neuron_wise_current_trace[total_number_of_neurons*syn_label + idx];
      // Update the synaptic conductance
      synaptic_current *= decay_factor;
      synaptic_current += synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label*total_number_of_neurons + idx];
      // Reset the conductance update
      synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label*total_number_of_neurons + idx] = 0.0f;
      total_current += synaptic_current;
      synaptic_data->neuron_wise_current_trace[total_number_of_neurons*syn_label + idx] = synaptic_current;

    }
    
    return total_current*multiplication_to_volts;
};


__device__ float my_voltage_spiking_injection_kernel(
    spiking_synapses_data_struct* in_synaptic_data,
    spiking_neurons_data_struct* neuron_data,
    float multiplication_to_volts,
    float current_membrane_voltage,
    float current_time_in_seconds,
    float timestep,
    int timestep_grouping,
    int idx,
    int g){
  
  spiking_synapses_data_struct* synaptic_data = (spiking_synapses_data_struct*) in_synaptic_data;
    
  int bufferloc = ((synaptic_data->neuron_inputs.bufferloc[0] + g) % synaptic_data->neuron_inputs.temporal_buffersize)*synaptic_data->neuron_inputs.input_buffersize;

  float total_current = 0.0f;
  for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
    total_current += synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_syn_labels];
    
    synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_syn_labels] = 0.0f;
    
  }


  // This is already in volts, no conversion necessary
  return total_current;
}
