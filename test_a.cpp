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
	char const *function_name3 = "network_lips";
	char const *degree_bound = "[3, 3]";
	char const *box = "[[ 0, 0.01 ],[ 0, 0.01 ]]";
	char const *lips = "71.9486";
	char const *activation = "ReLU";
	cout << "Result of call polynomial generation function: " << bernsteinPolyApproximation(module_name, function_name1, degree_bound, box, lips, activation) << endl;
	cout << "Result of call error bound function: " << stod(bernsteinPolyApproximation(module_name, function_name2, degree_bound, box, lips, activation)) << endl;
	cout << "Result of estimating Lipschitz constant: " << stod(bernsteinPolyApproximation(module_name, function_name3, degree_bound, box, lips, activation)) << endl;
}