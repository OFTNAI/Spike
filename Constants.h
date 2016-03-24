//
//  Constants.h
//  
//
//  Created by James Isbister on 23/03/2016.
//
//

#ifndef _Constants_h
#define _Constants_h

enum NEURON_TYPE
{
    NEURON_TYPE_IZHIKEVICH,
    NEURON_TYPE_POISSON,
    NEURON_TYPE_GEN //?? In Spike, what are they for? Generator?
    
};

enum CONNECTIVITY_TYPE
{
    CONNECTIVITY_TYPE_ALL_TO_ALL,
    CONNECTIVITY_TYPE_ONE_TO_ONE,
    CONNECTIVITY_TYPE_RANDOM,
    CONNECTIVITY_TYPE_GAUSSIAN,
    CONNECTIVITY_TYPE_IRINA_GAUSSIAN,
    CONNECTIVITY_TYPE_SINGLE
};




#endif
