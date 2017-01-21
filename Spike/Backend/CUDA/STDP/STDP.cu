// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/STDP/STDP.hpp"
#include <iostream>

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, STDP);

namespace Backend {
  namespace CUDA {
    STDP::~STDP() {
      CudaSafeCall(cudaFree(stdp_synapse_indices));
    }

    void STDP::prepare() {

      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());

      // Get the correct ID
      int stdp_id = frontend()->stdp_rule_id;
      if (stdp_id >= 0){
        total_number_of_stdp_synapses = frontend()->syns->stdp_synapse_number_per_rule[stdp_id];
      } else {
        total_number_of_stdp_synapses = 0;
      }

      allocate_device_pointers();
    }

    void STDP::allocate_device_pointers(){
      if (total_number_of_stdp_synapses > 0){
        CudaSafeCall(cudaMalloc((void **)&stdp_synapse_indices, sizeof(int)*total_number_of_stdp_synapses));
        CudaSafeCall(cudaMemcpy((void*)stdp_synapse_indices,
                                (void*)frontend()->syns->stdp_synapse_indices_per_rule[frontend()->stdp_rule_id],
                                sizeof(int)*total_number_of_stdp_synapses,
                                cudaMemcpyHostToDevice));
      }
    }

    void STDP::reset_state() {
    }
  }
}
