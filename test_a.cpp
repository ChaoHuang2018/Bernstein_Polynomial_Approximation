#include <iostream>
#include "bernstein_poly_approx.h"

//extern "C" {
//	#include "./bernstein_poly_approx.h"
//}

//using namespace bernstein_poly_approx;

int main(void)
{
	char const *module_name = "dubins_controller_poly_approx";
	char const *function_name1 = "dubins_poly_controller";
	char const *function_name2 = "poly_approx_error";
	char const *degree_bound = "[5, 8]";
	char const *box = "[[1, 3], [2, 7]]";
	char const *lips = "0.42349497952858162";
	cout << "Result of call polynomial generation function: " << bernsteinPolyApproximation(module_name, function_name1, degree_bound, box, lips) << endl;
	cout << "Result of call error bound function: " << stod(bernsteinPolyApproximation(module_name, function_name2, degree_bound, box, lips)) << endl;
}