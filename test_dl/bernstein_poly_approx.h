#ifndef bernstein_poly_approx
#define bernstein_poly_approx

#include <python3.6/Python.h>
#include <algorithm>
#include <iostream>
#include <list>

using namespace std;

string bernsteinPolyGeneration(char const *module_name, char const *function_name, char const *degree_bound, char const *box);

#endif


