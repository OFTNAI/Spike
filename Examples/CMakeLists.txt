foreach(example
    VogelsAbbottNet
    Brunel10K
    SimpleExample
    )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example}
    Spike
  )
if (BUILD_WITH_CUDA)
    target_link_libraries(${example}
      ${CUDA_LIBRARIES}
      )
    set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_CUDA -std=c++11")
  endif()
  #target_link_libraries(${example} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endforeach()
