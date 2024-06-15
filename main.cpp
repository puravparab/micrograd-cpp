#include <iostream>
#include "engine.h"

int main(){
	Value A(1.0f);
	Value B(2.0f);
	Value C = A + B;
	std::cout << C.data << std::endl;
	return 0;
};