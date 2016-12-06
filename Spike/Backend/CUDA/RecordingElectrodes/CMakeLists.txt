CUDA_ADD_LIBRARY (SpikeCUDARecordingElectrodes
  RecordingElectrodes.cu
  CountNeuronSpikesRecordingElectrodes.cu
  CollectNeuronSpikesRecordingElectrodes.cu
  NetworkStateArchiveRecordingElectrodes.cu
  STATIC)
