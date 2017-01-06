#ifndef WRITER_H
#define WRITER_H

#include "types.h"
#include "file_system.h"
#include "Simulation.h"

namespace parSIM{
	class writer : public file_system
	{
	public:
		writer(std::string _path, std::string _case);
		~writer();

		void set(Simulation* demSim = 0, Simulation* mbdSim = 0, fileFormat _ftype = BINARY);
		void start_particle_data();
		virtual bool run(double time);
		virtual bool cu_run(double time);

		int part;
		/*std::fstream of;*/
	};
}


#endif