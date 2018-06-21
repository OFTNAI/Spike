#ifndef UTILITY_FUNCS
#define UTILITY_FUNCS
/*
 *  A set of utility functions used primarily to connect using .mat files or with very specific connectivities
 */

#include "Spike/Spike.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
void connect_from_mat(
    int layer1,
    int layer2,
    conductance_spiking_synapse_parameters_struct* SYN_PARAMS, 
    std::string filename,
    SpikingModel* Model){

  ifstream weightfile;
  string line;
  stringstream ss;
  std::vector<int> prevec, postvec;
  int pre, post;
  float weight;
  int linecount = 0;
  weightfile.open(filename.c_str());

  if (weightfile.is_open()){
    printf("Loading weights from mat file: %s\n", filename.c_str());
    while (getline(weightfile, line)){
      if (line.c_str()[0] == '%'){
        continue;
      } else {
        linecount++;
        if (linecount == 1) continue;
        //printf("%s\n", line.c_str());
        ss.clear();
        ss << line;
        ss >> pre >> post >> weight;
        prevec.push_back(pre - 1);
        postvec.push_back(post - 1);
        //printf("%d, %d\n", pre, post);
      }
    }
    SYN_PARAMS->pairwise_connect_presynaptic = prevec;
    SYN_PARAMS->pairwise_connect_postsynaptic = postvec;
    SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;
    Model->AddSynapseGroup(layer1, layer2, SYN_PARAMS);
  }
}
void connect_with_sparsity(
    int input_layer,
    int output_layer,
    spiking_neuron_parameters_struct* input_layer_params,
    spiking_neuron_parameters_struct* output_layer_params,
    voltage_spiking_synapse_parameters_struct* SYN_PARAMS,
    float sparseness,
    SpikingModel* Model
    ){
  // Change the connectivity type
  int num_post_neurons = 
    output_layer_params->group_shape[0]*output_layer_params->group_shape[1];
  int num_pre_neurons = 
    input_layer_params->group_shape[0]*input_layer_params->group_shape[1];
  int num_syns_per_post = 
    sparseness*num_pre_neurons;
  
  std::vector<int> prevec, postvec;
  for (int outid = 0; outid < num_post_neurons; outid++){
    for (int inid = 0; inid < num_syns_per_post; inid++){
      postvec.push_back(outid);
      prevec.push_back(rand() % num_pre_neurons);
    }
  }

  SYN_PARAMS->pairwise_connect_presynaptic = prevec;
  SYN_PARAMS->pairwise_connect_postsynaptic = postvec;
  SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;

  Model->AddSynapseGroup(input_layer, output_layer, SYN_PARAMS);

}

#endif // UTILITY_FUNCS
