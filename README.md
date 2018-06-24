# Spike

A GPGPU based Spiking Neural Network (SNN) designed to provide users flexibility in the creation of simulations and high speed in execution. Written in C++/CUDA. (Distributed under the MIT License, See LICENSE)

## Getting Started

[Click here for a getting started guide and a description of the main components of spike.](https://sites.google.com/view/spike-simulator/home)

Please see the requirements for this simulator below. The comparison of Spike to other simulators can be see in the [SNNSimulatorComparison](https://github.com/nasiryahm/SNNSimulatorComparison) repository.

Executing the install.sh file in this directory will create a Build folder and compile the example networks in the Examples folder.

## Requirements
  - NVIDIA GPU capable of CUDA code execution
  - NVIDIA CUDA Toolkit v7.5 (or greater)
  - C++11 compiler
  - CMAKE v3.1 (or greater)

This simulator is under constant development. We recommend users to regularly update any codebase which they use. This software is regularly tested on Ubuntu 16.04 with NVIDIA GPUs (Geforce GTX 980, Geforce GTX 980ti, Geforce GTX 1080, Geforce GTX 1070++) The software has been tested on CUDA Toolkit 7.5 and above.

This tool should be capable of running on any Linux/Mac OS system with an NVIDIA GPU of Compute Capability >= 5.2

## Recent Updates
  - Addition of VogelsAbbott and Brunel Examples
  - High speed synapse management
  - Relocatable device code for neuron current injections


## Contact Us
If you have any questions or wish to contribute, please contact: 

Development Team:
  - Nasir Ahmad (nasir.ahmad@psy.ox.ac.uk)
  - James Isbister (james.isbister@psy.ox.ac.uk)
  - Toby St Clere Smithe (toby.smithe@psy.ox.ac.uk)

Contributors:
  - Akihiro Eguchi
