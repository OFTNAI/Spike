foreach(example
    VogelsAbbottNet
    )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example}
    Spike
  )
  #[[
  if (BUILD_WITH_CUDA)
    target_link_libraries(${example}
      ${CUDA_LIBRARIES}
    -L/usr/local/cuda/lib64 -lcudadevrt
      )
    set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_CUDA")
  endif()
  target_link_libraries(${example} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  ]]
endforeach()
