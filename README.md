# Bernstein_Polynomial_Approximation

Command for testing python/c++ hybrid programming:  
g++ try_bernstein.cpp -o try_bernstein -I/usr/include/python3.6m/ -lpython3.6m  
./try_bernstein dubins_controller_poly_approx dubins_poly_controller [2,2] [[1,3],[2,5]]  

Command for testing dynamic library:  
g++ -shared -fPIC bernstein_poly_approx.cpp -o libbernstein_poly_approx.so -I/usr/include/python3.6m/ -lpython3.6m  
g++ test_so.cpp -o test_so -L. -lbernstein_poly_approx  

Command for testing static library:  
g++ -c bernstein_poly_approx.cpp -I/usr/include/python3.6m/ -lpython3.6m  
ar -cr libbernstein_poly_approx.a bernstein_poly_approx.o  
g++ test_a.cpp -o test_a libbernstein_poly_approx.a -I/usr/include/python3.6m/ -lpython3.6m

## Running example
'''
python error_analysis.py --filename nn_13_sigmoid
--error_bound 1e-3
'''
