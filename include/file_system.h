#ifndef FILE_SYSTEM_H
#define FILE_SYSTEM_H

#include <fstream>
#include <string>
#include "types.h"

namespace parSIM
{
	class Simulation;

	class file_system
	{
	public:
		file_system();
		virtual ~file_system() {}

		bool save_geometry();
		void close();

		virtual bool run(double time) = 0;
		virtual bool cu_run(double time) = 0;

	protected:
		//std::fstream of;
		std::string caseName;
		std::string path;
		solver_type svt;
		fileFormat fmt;

		Simulation* dem_sim;
		Simulation* mbd_sim;
	};
}

#endif