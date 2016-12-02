#ifndef ConductanceSPIKINGSYNAPSES_H
#define ConductanceSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

struct conductance_spiking_synapse_parameters_struct : spiking_synapse_parameters_struct {
	conductance_spiking_synapse_parameters_struct(): biological_conductance_scaling_constant_lambda(1.0), reversal_potential_Vhat(0.0f), decay_term_tau_g(0.001f) { spiking_synapse_parameters_struct(); }

	float biological_conductance_scaling_constant_lambda;
	float reversal_potential_Vhat;
	float decay_term_tau_g;
};

class ConductanceSpikingSynapses; // forward definition

namespace Backend {
  class ConductanceSpikingSynapses : public virtual SpikingSynapses {
  public:
    ADD_FRONTEND_GETTER(ConductanceSpikingSynapses);
    virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds) = 0;
  };
}

#include "Spike/Backend/Dummy/Synapses/ConductanceSpikingSynapses.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"
#endif


class ConductanceSpikingSynapses : public SpikingSynapses {

public:
  ~ConductanceSpikingSynapses();

  ADD_BACKEND_GETTER(ConductanceSpikingSynapses);

  float * synaptic_conductances_g = nullptr;
  float * biological_conductance_scaling_constants_lambda = nullptr;
  float * reversal_potentials_Vhat = nullptr;
  float * decay_terms_tau_g = nullptr;

  // Synapse Functions
  virtual void AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params);

  virtual void increment_number_of_synapses(int increment);
  virtual void shuffle_synapses();

  virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep);
  virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);

  virtual void prepare_backend(Context* ctx = _global_ctx);
  virtual void reset_state();
};

#endif
