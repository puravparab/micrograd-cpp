#include <iostream>
#include "engine.h"

int main(){
	Value A(1.0f);
	Value B(2.0f);
	Value C = A * B;
	C.grad = 1.0f;
	C._backward();
	A._backward();
	B._backward();
	std::cout << "A: " << A.grad << std::endl;
	std::cout << "B: " << B.grad << std::endl;
	std::cout << "C: " << C.grad << std::endl;
	return 0;
};