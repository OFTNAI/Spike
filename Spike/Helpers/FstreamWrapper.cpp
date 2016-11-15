 #include "FstreamWrapper.hpp"
#include "Utilities.hpp"

fstreamWrapper::fstreamWrapper() : fstream() { setupExceptions(); }

fstreamWrapper::fstreamWrapper(const char * inputWeightFile) : fstream(inputWeightFile, std::ios::binary) { setupExceptions(); }

fstreamWrapper::fstreamWrapper(const string & inputWeightFile) : fstream(inputWeightFile.c_str(), std::ios::binary) { setupExceptions(); }

void fstreamWrapper::setupExceptions() 
{ 
	exceptions ( fstream::failbit | fstream::badbit); //
}

/*
fstreamWrapper::fstreamWrapper() : stream() { stream.exceptions (fstream::failbit | fstream::badbit); }

void fstreamWrapper::open(const char * inputWeightFile) { stream.open(inputWeightFile); }
void fstreamWrapper::open(const string & inputWeightFile) { stream.open(inputWeightFile.c_str()); }
void fstreamWrapper::close() { stream.close(); }
*/

template <class T>
fstreamWrapper & fstreamWrapper::operator>>(T& val) {
    read(reinterpret_cast<char*>(&val), sizeof(T));
    return *this;
}

template <class T>
fstreamWrapper & fstreamWrapper::operator<<(T val) {
    write(reinterpret_cast<char*>(&val), sizeof(T));
    return *this;
}

// Template madness
// an alternative solution would have been to inject the above functions into the class
// code, apparently that does not cause linker confusion since the
// code is regarded as part of the declaration.. jeeeez C++
//template fstreamWrapper & fstreamWrapper::operator<<(unsigned val);
template fstreamWrapper & fstreamWrapper::operator<<(u_short val);
template fstreamWrapper & fstreamWrapper::operator<<(float val);
//template fstreamWrapper & fstreamWrapper::operator<<(bool val);

//template fstreamWrapper & fstreamWrapper::operator>>(unsigned & val);
template fstreamWrapper & fstreamWrapper::operator>>(u_short & val);
template fstreamWrapper & fstreamWrapper::operator>>(float & val);
//template fstreamWrapper & fstreamWrapper::operator>>(bool & val);
