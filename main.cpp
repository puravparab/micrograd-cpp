#include <iostream>
#include "engine.h"

int main(){
	Value A(2.0f);
	Value B(2.0f);

	Value C = B.tanh();
	Value D = C.pow(A);

	D.backward();
	
	std::cout << "A: " << A.data << " -> " << A.grad << std::endl;
	std::cout << "B: " << B.data << " -> " << B.grad << std::endl;
	std::cout << "C: " << C.data << " -> " << C.grad << std::endl;
	std::cout << "D: " << D.data << " -> " << D.grad << std::endl;
	return 0;
};