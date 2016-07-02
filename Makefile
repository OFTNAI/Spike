#--------------------------------------------------------------------------------
#
# Simple makefile for compiling the Spike Simulator
#
#--------------------------------------------------------------------------------

# Compiler = Nvidia version for CUDA code
CC = nvcc
# Flags
# -c = ignore the fact that there is no main
# --compiler-options -Wall = Warnings All. Give them to me.
# Wall flag is inefficient
CFLAGS = -c
# CFLAGS += -lineinfo

# Mac OS X 10.9+ uses libc++, which is an implementation of c++11 standard library. 
# We must therefore specify c++11 as standard for out of the box compilation on Linux. 
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS += --std=c++11 
endif


# Include all of the folders from which we want to include files
# CU
SIM_FILES := $(wildcard Simulator/*.cu)
NEUR_FILES := $(wildcard Neurons/*.cu)
STDP_FILES := $(wildcard STDP/*.cu)
HELP_FILES := $(wildcard Helpers/*.cu)
SYNS_FILES := $(wildcard Synapses/*.cu)
REC_FILES := $(wildcard RecordingElectrodes/*.cu)
ANALY_FILES := $(wildcard SpikeAnalyser/*.cu)
# CPP
HELP_CPP_FILES := $(wildcard Helpers/*.cpp)

# COMBINE LISTS
CU_FILES := $(SIM_FILES) $(NEUR_FILES) $(STDP_FILES) $(HELP_FILES) $(SYNS_FILES) $(REC_FILES) $(ANALY_FILES)
CPP_FILES := $(HELP_CPP_FILES)

# Create Objects
CU_OBJ_FILES := $(addprefix ObjectFiles/,$(notdir $(CU_FILES:.cu=.o)))
CPP_OBJ_FILES := $(addprefix ObjectFiles/,$(notdir $(CPP_FILES:.cpp=.o)))


# Default
model: ${FILE}
directory: ${EXPERIMENT_DIRECTORY}

${FILE}: ObjectFiles/${FILE}.o $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(CC) -lineinfo ObjectFiles/${FILE}.o $(CU_OBJ_FILES) $(CPP_OBJ_FILES) -o ${EXPERIMENT_DIRECTORY}/bin/${FILE}

# Compiling the Model file
ObjectFiles/${FILE}.o: ${EXPERIMENT_DIRECTORY}/${FILE}.cpp
	$(CC) $(CFLAGS) ${EXPERIMENT_DIRECTORY}/${FILE}.cpp -o $@

# CUDA
ObjectFiles/%.o: */%.cu
	$(CC) $(CFLAGS) -o $@ $<
# CPP
ObjectFiles/%.o: Helpers/%.cpp
	$(CC) $(CFLAGS) -o $@ $<




# Test Files
TEST_CPP_FILES := $(wildcard Tests/*.cu)
TEST_OBJ_FILES := $(addprefix Tests/obj/,$(notdir $(TEST_CPP_FILES:.cu=.o)))

test: ${TEST_OBJ_FILES} $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(CC)  -lineinfo ${TEST_OBJ_FILES} $(CU_OBJ_FILES) $(CPP_OBJ_FILES) -o Tests/unittests

# Test Compilation
Tests/obj/%.o: Tests/%.cu
	$(CC) $(CFLAGS) -o $@ $<

# Cleaning tests
cleantest:
	rm Tests/obj/* Tests/unittests



# Removing all created files
clean:
	rm ObjectFiles/*.o Tests/obj/*
