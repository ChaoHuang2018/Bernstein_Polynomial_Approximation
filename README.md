# Bernstein_Polynomial_Approximation

Testing command 1:  
g++ try_bernstein.cpp -o try_bernstein -I/usr/include/python3.6m/ -lpython3.6m  
./try_bernstein dubins_controller_poly_approx dubins_poly_controller [2,2] [[1,3],[2,5]]  

Testing command 2:  
g++ -shared -fPIC bernstein_poly_approx.cpp -o libbernstein_poly_approx.so -I/usr/include/python3.6m/ -lpython3.6m  
g++ test_so.cpp -o test_so -L. -lbernstein_poly_approx  