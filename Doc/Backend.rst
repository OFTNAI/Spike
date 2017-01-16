Backend Structure
=================

Spike employs a separation of code into front and back "ends":
frontend classes are used to specify the computations to be done,
while backend classes implement those computations. The consequence of
this is that Spike can support different hardware types, and it is
possible to experiment with different simulation methods without
having to rewrite the code specifying the model of interest.

This document describes the structure of a Spike backend. For a simple
reference implementation, see the `Dummy` backend (in `Spike/Backends/Dummy`),
which provides stub implementations of all the frontend classes, and which is
used as a fallback in case a requested backend is not available at runtime.

Example
-------

The following example illustrates all the relevant features of the
Spike backend system.

...


Defining a backend class
------------------------

The backend interface is defined by pure virtual classes in the
`Backend` namespace, which must derive from the base class
`Backend::SpikeBackendBase`. Each backend class is associated with a
corresponding class of the same name in the top-level namespace, and
different backends are themselves grouped by namespace. For example,
the class `Backend::Dummy::LIFSpikingNeurons` (defined in
`Spike/Backend/Dummy/Neurons/LIFSpikingNeurons.hpp`) is the `Dummy`
backend for `LIFSpikingNeurons`, and its interface is defined by
`Backend::LIFSpikingNeurons` (defined with `LIFSpikingNeurons` in
`Spike/Neurons/LIFSpikingNeurons.hpp`).

Each backend class must override two virtual methods -- `prepare()`,
and `reset_state()` -- as well as provide a constructor and a default
destructor override. `reset_state()` is a convenience function that is
expected to reset the state of the object to its initial state (for
example, at time 0). `prepare()` is called on initializing the
backend, and should prepare the backend class for performing
computations. Typically, this might involve allocating memory, or
copying constants from the frontend.

Importantly, `reset_state()` and `prepare()` are both virtual methods,
and so a call to either will reach the most derived
class. Consequently, it is important that implementations of derived
classes should call the corresponding functions back up the
inheritance hierarchy.

The constructor is provided in `Spike/Backend/Macros.hpp`, by the
macro `SPIKE_MAKE_BACKEND_CONSTRUCTOR(TYPE)`, and no other constructor
should be defined; other initialization should be performed in
`prepare()`, when the frontend pointer is available.

The frontend pointer is accessed by the method `frontend()`, which
returns a pointer to the corresponding type in the top-level
namespace. This function is defined by adding the macro
`SPIKE_ADD_BACKEND_FACTORY(TYPE)` to the backend interface definition,
which also sets up the backend 'factory' for the given type.


Connecting to the frontend
--------------------------

...


Registering a backend
---------------------

...

