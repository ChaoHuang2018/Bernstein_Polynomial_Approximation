CXX = g++-8
HOME= /usr/local/include
LIB_HOME = ../flowstar
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpython3.6m
CFLAGS = -I . -I $(HOME) -I /usr/include/python3.6m/ -g -O3 -std=c++11
LINK_FLAGS = -g -L. -L$(LIB_HOME) -L/usr/local/lib

all: nn_12_relu

nn_12_relu: nn_12_relu.o ../libbernstein_poly_approx.a
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean: 
	rm -f *.o test
