
#ifndef UTILITIES_H
#define UTILITIES_H

#ifndef DEBUG
#define OMP_ENABLE
#endif

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
	#define OS_WIN
#endif

typedef unsigned short u_short;

inline u_short wrap(int x, u_short d) {
	
	// One cannot trust result of (x % b) with negative
	// input x for various compilers:
	// http://www.learncpp.com/cpp-tutorial/32-arithmetic-operators/
	// Hence we take abs(x) first
	
	if(x > 0)
		return x % d;
	else if((-x) % d == 0)
		return 0;
	else
		return d - ((-x) % d);
}

/*
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/types.h>

using namespace std;

vector<string> directoryListing(const char * directory);

vector<string> split(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);
vector<string> split(const char * s, char delim);

template <class T>
vector<T> splitAndConvert(char * s);

template <class T>
T convert(char * s);
*/
#endif // UTILITIES_H
