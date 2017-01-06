#ifndef ISPH_LOADER_H
#define ISPH_LOADER_H

#include <string>
#include <vector>
#include "Simulation.h"

namespace parSIM 
{
	class Loader
	{
	public:

		Loader() {}
		~Loader() {}

		void SetInput(const std::string& basePath, const std::string& casePath)	
		{ 
			base_path = basePath;
			path = basePath + casePath; 
			int s_pos = path.find_last_of("/")+1;
			int e_pos = path.find_last_of(".");
			case_name = path.substr(s_pos, e_pos-s_pos);
		}

		virtual Simulation* Read() = 0;

	protected:
		std::string base_path;
		std::string path;
		std::string case_name;
	};
}

#endif
