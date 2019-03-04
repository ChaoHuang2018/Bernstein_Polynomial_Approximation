#include <iostream>
#include "bernstein_poly_approx.h"

//extern "C" {
//	#include "./bernstein_poly_approx.h"
//}

//using namespace bernstein_poly_approx;

int main(void)
{
	char const *module_name = "dubins_controller_poly_approx";
	char const *function_name = "dubins_poly_controller";
	char const *degree_bound = "[5, 8]";
	char const *box = "[[1, 3], [2, 7]]";
	cout << "Result of call: " << bernsteinPolyGeneration(module_name, function_name, degree_bound, box) << endl;
}