#pragma once

//	CUDA/ErrorCheckHelpers
//
//
//	Original Author: Nasir Ahmad in CUDAcode.h
//	Date: 31/3/2016

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

// The two functions that we can use
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

// When we wish to check for errors in functions such as malloc directly
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
// If we want to do error-checking
#ifdef CUDA_ERROR_CHECK
	// Check for success
    if ( cudaSuccess != err )
    {
    	// Output the issue if there is one
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// When we wish to check that functions did not introduce errors
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	// Get last error (i.e. in the function that has just run)
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}
