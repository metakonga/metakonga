#ifndef RIGID_BODY_H
#define RIGID_BODY_H

#include "pointmass.h"

namespace parSIM
{
	namespace mass
	{
		class rigid_body : public pointmass
		{
		public:
			rigid_body(Simulation *sim, std::string _name, geometry* _Geo = NULL, mass_type mt=RIGID_BODY);
			virtual ~rigid_body();

			void define(unsigned int Id, double m, vector3<double>& diagIner, vector3<double>& symIner, vector3<double>& pos, euler_parameter<double>& epara);
			virtual void save2file(std::fstream& of);
			virtual void define_mass();
		};
	}	
}

#endif