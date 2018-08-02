#ifndef SIMULATION_H
#define SIMULATION_H

#include <iostream>
#include <list>
#include <map>
#include <ctime>
#include "algebra.h"
#include "types.h"
#include "writer.h"

static std::string make_date_form(tm& d)
{
	char date[256];
	sprintf_s(date, sizeof(char)*256, "%d.%d.%d-%d.%d.%d", 1900+d.tm_year, d.tm_mon+1, d.tm_mday, d.tm_hour, d.tm_min, d.tm_sec);
	return std::string(date);
}

namespace parSIM
{
	class particles;
	class Integrator;
	class geometry;
	class pointmass;
	class kinematicConstraint;
	class drivingConstraint;
	class force;
	class cell_grid;
	class Mbdsimulation;
	class Demsimulation;

	class Simulation
	{
	public:
		Simulation(std::string name);
		virtual ~Simulation();

		void clear();
		// set member function
		//void setSubSimulation(Simulation* subSim) { sub_sim = subSim; }
		void setSpecificDataFileName(std::string& sdata) { specific_data = sdata; }
 		void setName(std::string& name) { Name = name; }
		void setDevice(device_type _device) { device = _device; }		
		void setSpecificData(std::string spath);

		device_type& Device() { return device; }

		std::string& getSpecificDataFileName() { return specific_data; }
		std::string getBasePath() { return base_path; }
		std::string getName() { return Name; }

		unsigned int getMassSize();

		bool isExistSpecificData() { return !specific_data.empty(); }

		geometry* getGeometry(std::string);

		//Simulation* getSubSimulation() { return sub_sim; }
		particles* getParticles() 
		{ 
			return ps; 
		}
		std::map<std::string, geometry*> *getGeometries() 
		{ 
			return &geometries; 
		}
		std::map<std::string, pointmass*> *getMasses() { return &masses; }
		std::map<std::string, kinematicConstraint*> *getKinematicConstraint() { return &kinConsts; }
		std::map<std::string, drivingConstraint*> *getDrivingconstraint() { return &driConsts; }
		//Integrator** getIntegrators() { return itor; }
		//Integrator* getIntegrator() { return itor[Itor]; }		
		void insert_geometry(std::string, geometry*);
		void insert_pointmass(std::string, pointmass*);
		
		void add_pair_material_condition(int bm, int pm=0);

		static std::string specific_data;
		static std::string base_path;
		static std::string caseName;
		static dimension_type dimension;
		static precision_type float_type;
		static solver_type solver;
		static unsigned int save_step;
		static double sim_time;
		static double time;
		static double dt;
		static vector3<double> gravity;
		static unsigned int cStep;

	protected:
		std::string Name;
		device_type device;
		particles *ps;
		force *cforce;
		cell_grid *cdetect;
		std::map<std::string, geometry*> geometries;
		std::map<std::string, pointmass*> masses;
		std::map<std::string, kinematicConstraint*> kinConsts;
		std::map<std::string, drivingConstraint*> driConsts;

		typedef std::map<std::string, kinematicConstraint*>::iterator KConstraintIterator;
		typedef std::map<std::string, drivingConstraint*>::iterator DConstraintIterator;
		typedef std::map<std::string, pointmass*>::iterator MassIterator;
	};
}

#endif