#ifndef MBD_FORCE_H
#define MBD_FORCE_H

#include "algebra.h"

using namespace algebra;

namespace parSIM
{
	class Simulation;
	class mbd_force
	{
	public:
		mbd_force(Simulation *_sim);
		~mbd_force();
		void setGavity(double x, double y, double z) { gravity = vector3<double>(x, y, z); }

		vector3<double>& Gravity() { return gravity; }

	private:
		vector3<double> gravity;

		Simulation *sim;
	};
}

#endif