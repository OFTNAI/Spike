add_library (SpikeDummyNeurons STATIC
  Neurons.cpp
  SpikingNeurons.cpp

  AdExSpikingNeurons.cpp
  IzhikevichSpikingNeurons.cpp
  LIFSpikingNeurons.cpp

  InputSpikingNeurons.cpp
  GeneratorInputSpikingNeurons.cpp
  PoissonInputSpikingNeurons.cpp
  GeneralPoissonInputSpikingNeurons.cpp
  ImagePoissonInputSpikingNeurons.cpp
  )
