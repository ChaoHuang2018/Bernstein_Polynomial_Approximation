#include <iostream>
#include "bernstein_poly_approx.h"
#include <string>

//extern "C" {
//	#include "./bernstein_poly_approx.h"
//}

//using namespace bernstein_poly_approx;

int main(void)
{
	char const *module_name = "controller_approximation_lib";
	char const *function_name1 = "poly_approx_controller";
	char const *function_name2 = "poly_approx_error";
	char const *function_name3 = "network_lips";
	char const *degree_bound = "[3, 3, 3, 3]";
	char const *box = "[[0.5, 1.5],[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]]";
	char const *lips = "71.9486";
	char const *activation = "ReLU";
	char const *output_index = "0";
	char const *neural_network = "nn_19_relu";
	cout << "Result of call polynomial generation function: " << bernsteinPolyApproximation(module_name, function_name1, degree_bound, box, activation, output_index, neural_network) << endl;
	//string new_lips = bernsteinPolyApproximation(module_name, function_name3, degree_bound, box, activation, output_index);
	//cout << "Result of estimating Lipschitz constant: " << new_lips << endl;
	double a = stod(bernsteinPolyApproximation(module_name, function_name2, degree_bound, box, activation, output_index, neural_network));
	cout << "Result of call error bound function: " << a << endl;
	
}
