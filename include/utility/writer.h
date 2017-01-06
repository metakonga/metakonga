#ifndef WRITER_H
#define WRITER_H

#include <iostream>
#include <fstream>
#include <string>
#include "../algebra/vector3.hpp"

class Simulation;
namespace utility
{
	class writer
	{
	public:
		writer();
		~writer();

		static void SetSimulation(Simulation *baseSimulation);
		static void EndSimulation();
		static void SaveGeometry();
		static bool Save(unsigned int step);
		static void SetFileSystem(std::string filename);
		static void CloseFileSystem();

		static unsigned int part;
		static char solverType;
		static char fileFormat;
		static std::string subDirectory;
		static std::string directory;
		static algebra::vector3<double> pick_force;

	private:
		static std::fstream of;
		static std::fstream pf_of;
		
		static Simulation* sim;
	};
}

#endif