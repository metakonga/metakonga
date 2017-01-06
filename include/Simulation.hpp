#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "algebra.h"
#include "types.h"
#include "Object/Particles.hpp"
#include "Object/Geometry.hpp"

template<typename base_type>
class Simulation
{
public:
	Simulation(std::string bpath, std::string cname)
		: work_directory(bpath)
		, case_name(cname)
	{}
	virtual ~Simulation()
	{
		for (std::map<std::string, Geometry<base_type>*>::iterator it = geometries.begin(); it != geometries.end(); it++){
			delete it->second;
		}
	}

	Geometry<base_type>* CreateGeometry(geometry_shape gs, std::string name, geometry_type gt)
	{
		Geometry<base_type> *geometry = NULL;
		switch (gs){
		case RECTANGLE:
			geometry = new geo::Rectangle<base_type>(name, gt);
			geometries[name] = geometry;
			break;
		}
		return geometry;
	}

	void SaveGeometries(char ft)
	{
		std::fstream of;
		if (ft == 'b'){
			std::string filename = work_directory + case_name + "/boundary";
			if (!of.is_open()){
				if (ft == 'b'){
					filename += ".bin";
					of.open(filename, std::ios::binary | std::ios::out);
				}
			}
			for (std::map<std::string, Geometry<base_type>*>::iterator it = geometries.begin(); it != geometries.end(); it++){
				if (it->second->Type() != GEO_PARTICLE)
					it->second->save2file(of, ft);
			}
		}
		int iMin = INT_MIN;
		of.write((char*)&iMin, sizeof(int));
		of.close();
	}

	void SaveCycle(unsigned int step)
	{
		base_type tme = step * dt;
		char partName[256] = { 0, };
		sprintf_s(partName, sizeof(char) * 256, "%spart%04d.bin", (work_directory + case_name).c_str(), part);
		std::fstream of;
		of.open(partName, std::ios::out | std::ios::binary);
		if (of.is_open()){
			of.write((char*)&tme, sizeof(base_type));
			of.write((char*)&balls->Np(), sizeof(unsigned int));
			of.write((char*)balls->Position(), sizeof(base_type) * 4 * balls->Np());
			of.write((char*)balls->Velocity(), sizeof(base_type) * 4 * balls->Np());
		}

	}

	virtual bool Initialize() = 0;
	//char& FloatingDataType() { return floatingDataType; }
	//std::string& WorkDirectory() { return work_directory; }
	//std::string& CaseName() { return case_name; }
	//unsigned int& NBall() { return nball; }
	//particles<base_type>* Balls() { return balls; }
	//std::map<std::string, Geometry*>& Geometries() { return geometries; }

	/*static double dt;
	static double times;*/

protected:
	//char floatingDataType;
	//vector3<base_type> gravity;
	base_type dt;
	std::string work_directory;
	std::string case_name;

	//device_type device;
	//dimension_type dim;

	particles<base_type> *balls;
	//unsigned int nball;
	std::map<std::string, Geometry<base_type>*> geometries;
};

//template<typename base_type> Simulation<base_type>::dt = 0.0;
//template<typename base_type> Simulation<base_type>::times = 0.0;

#endif