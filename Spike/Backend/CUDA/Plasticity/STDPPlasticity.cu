// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/STDPPlasticity.hpp"
#include <iostream>

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, STDPPlasticity);

namespace Backend {
  namespace CUDA {
    STDPPlasticity::~STDPPlasticity() {
      CudaSafeCall(cudaFree(stdp_synapse_indices));
    }

    void STDPPlasticity::prepare() {

      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());

      // Get the correct ID
      int plasticity_id = frontend()->plasticity_rule_id;
      if (plasticity_id >= 0){
        total_number_of_stdp_synapses = frontend()->syns->plasticity_synapse_number_per_rule[plasticity_id];
      } else {
        total_number_of_stdp_synapses = 0;
      }

      allocate_device_pointers();
    }

    void STDPPlasticity::allocate_device_pointers(){
      if (total_number_of_stdp_synapses > 0){
        CudaSafeCall(cudaMalloc((void **)&stdp_synapse_indices, sizeof(int)*total_number_of_stdp_synapses));
        CudaSafeCall(cudaMemcpy((void*)stdp_synapse_indices,
                                (void*)frontend()->syns->plasticity_synapse_indices_per_rule[frontend()->plasticity_rule_id],
                                sizeof(int)*total_number_of_stdp_synapses,
                                cudaMemcpyHostToDevice));
      }
    }

    void STDPPlasticity::reset_state() {
    }
  }
}
