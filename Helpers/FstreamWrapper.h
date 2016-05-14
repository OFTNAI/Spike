/*
 *  Network.h
 *  VisBack
 *
 *  Created by Bedeho Mender on 11/29/10.
 *  Copyright 2010 OFTNAI. All rights reserved.
 *
 */

#ifndef FSTREAMWRAPPER_H
#define FSTREAMWRAPPER_H

// Forward declarations

// Includes
#include <fstream>
#include <string>

using std::fstream;
using std::string;

/*
* Was originally a fstream subclass,
* but to be able to force setting
* exception mask prior to any file opening attempt
* (which otherwise would have been able with constructor).
* Disadvantage is that we have to keep exporting fstream
* methods like open/close.
*/
class fstreamWrapper : public fstream {

	//private:
	//	fstream stream;

    public: 
        
        // Constructors
        fstreamWrapper();
		fstreamWrapper(const char * inputWeightFile);
		fstreamWrapper(const string & inputWeightFile);

		void setupExceptions();

		/*
		// Open file, throws exception
		void open(const char * inputWeightFile);
		void open(const string & inputWeightFile);

		void close()
		*/
        
        // Overloaded output/input ops. resp.
        template <class T> fstreamWrapper & operator>>(T& val);
        template <class T> fstreamWrapper & operator<<(T val);
};

#endif // FSTREAMWRAPPER_H
