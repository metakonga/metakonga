#include "cu_particleSystem.h"

int main(int argc, char** argv)
{
	cu_particleSystem psys;
	psys.initialize();
	psys.run();
	std::cout << "over" << std::endl;
	return 1;
}