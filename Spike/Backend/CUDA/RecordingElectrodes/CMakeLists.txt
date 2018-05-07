set (CUDARecordingElectrodes
  ${CMAKE_CURRENT_SOURCE_DIR}/RecordingElectrodes.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/CountNeuronSpikesRecordingElectrodes.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/CollectNeuronSpikesRecordingElectrodes.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/NetworkStateArchiveRecordingElectrodes.cu
  PARENT_SCOPE)
