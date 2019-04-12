CXX = g++-8
HOME= /usr/local/include
LIB_HOME = ../flowstar
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpython3.6m
CFLAGS = -I . -I $(HOME) -I /usr/include/python3.6m/ -g -O3 -std=c++11
LINK_FLAGS = -g -L../Bernstein_Polynomial_Approximation -L$(LIB_HOME) -L/usr/local/lib

all: nn_12_relu nn_12_tanh nn_12_sigmoid nn_13_relu nn_13_tanh nn_13_sigmoid nn_14_relu nn_14_tanh nn_14_sigmoid nn_15_relu nn_15_tanh nn_15_sigmoid nn_16_relu nn_16_tanh nn_16_sigmoid nn_17_relu nn_17_tanh nn_17_sigmoid nn_18_relu nn_18_tanh nn_18_sigmoid nn_19_relu nn_19_tanh nn_19_sigmoid interval nn_tora nn_inv_pen

nn_12_relu: nn_12_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_12_tanh: nn_12_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_12_sigmoid: nn_12_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_13_relu: nn_13_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_13_tanh: nn_13_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_13_sigmoid: nn_13_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_14_relu: nn_14_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_14_tanh: nn_14_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_14_sigmoid: nn_14_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_15_relu: nn_15_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_15_tanh: nn_15_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_15_sigmoid: nn_15_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_16_relu: nn_16_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_16_tanh: nn_16_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_16_sigmoid: nn_16_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_17_relu: nn_17_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_17_tanh: nn_17_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_17_sigmoid: nn_17_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_18_relu: nn_18_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_18_tanh: nn_18_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_18_sigmoid: nn_18_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_19_relu: nn_19_relu.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_19_tanh: nn_19_tanh.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_19_sigmoid: nn_19_sigmoid.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

interval: interval.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_tora: nn_tora.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_inv_pen: nn_inv_pen.o ../Bernstein_Polynomial_Approximation/libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean: 
	rm -f *.o test
