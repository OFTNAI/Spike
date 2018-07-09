#include "STDPPlasticity.hpp"

STDPPlasticity::~STDPPlasticity(){
}

void STDPPlasticity::AddSynapseIndices(int synapse_start_id, int num_synapses_to_add){
  for (int id = synapse_start_id; id < (synapse_start_id + num_synapses_to_add); id++)
    plastic_synapses.push_back(id);

  total_number_of_plastic_synapses = plastic_synapses.size();
}


void STDPPlasticity::reset_state() {
  backend()->reset_state();
}
