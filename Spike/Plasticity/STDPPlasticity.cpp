#include "STDPPlasticity.hpp"

STDPPlasticity::~STDPPlasticity(){
  for (int pre_index = 0; pre_index < pre_neuron_set.size(); pre_index++)
    free(pre_neuron_efferent_ids[pre_index]);
}

void STDPPlasticity::AddSynapse(int presynaptic_neuron_id, int postsynaptic_neuron_id, int synapse_id){
  plastic_synapses.push_back(synapse_id);
  // Store the post-synaptic neuron ID if it has not yet already been stored
  bool post_exists = false;
  int post_loc = -1;
  if (post_neuron_conversion.size() > postsynaptic_neuron_id){
    if (post_neuron_conversion[postsynaptic_neuron_id] >= 0){
      post_exists = true;
      post_loc = post_neuron_conversion[postsynaptic_neuron_id];
    }
  }
  if (!post_exists){
    while (post_neuron_conversion.size() <= postsynaptic_neuron_id)
      post_neuron_conversion.push_back(-1);
    post_neuron_set.push_back(postsynaptic_neuron_id);
    post_loc = post_neuron_set.size() - 1;
    post_neuron_conversion[postsynaptic_neuron_id] = post_loc; 
  }
  
  /*
  // If the presynaptic_neuron exists, find it and add the synapse to the correct location, otherwise create a location
  // Also add the location in the post_neuron_set of the postsynaptic neuron (to be used later)
  bool pre_exists = false;
  for (int neuron_id = 0; neuron_id < pre_neuron_set.size(); neuron_id++){
    if (pre_neuron_set[neuron_id] == presynaptic_neuron_id){
      pre_neuron_efferent_count[neuron_id]++;

      pre_neuron_efferent_ids[neuron_id] = (int*)realloc(pre_neuron_efferent_ids[neuron_id], pre_neuron_efferent_count[neuron_id] * sizeof(int));

      pre_neuron_efferent_ids[neuron_id][pre_neuron_efferent_count[neuron_id] - 1] = synapse_id;
      pre_exists = true;
      break;
    }
  }
  if (!pre_exists){
    int* efferent_list = (int*)malloc(sizeof(int));
    pre_neuron_set.push_back(presynaptic_neuron_id);
    pre_neuron_efferent_count.push_back(1);
    efferent_list[0] = synapse_id;
    pre_neuron_efferent_ids.push_back(efferent_list);

  }
  */

}

void STDPPlasticity::reset_state() {
  backend()->reset_state();
}
