#ifndef COUPLINGSYSTEM_H
#define COUPLINGSYSTEM_H

namespace parSIM
{
	class Demsimulation;
	class Mbdsimulation;

	class CouplingSystem
	{
	public:
		CouplingSystem(Demsimulation* dem = 0, Mbdsimulation* mbd = 0);
		~CouplingSystem();

		bool Run();
		bool ModifyRun(unsigned int tframe);


	private:
		Demsimulation *Dem;
		Mbdsimulation *Mbd;
	};
}

#endif