#include "file_system.h"
#include "Simulation.h"
#include "geometry.h"
#include <list>

using namespace parSIM;

file_system::file_system()
{

}

bool file_system::save_geometry()
{
	std::fstream pf;
	pf.open(path + "/boundary.bin", std::ios::out | std::ios::binary);
	std::map<std::string, geometry*> *geos;
	std::map<std::string, geometry*>::iterator geo;
	if(dem_sim){
		geos = dem_sim->getGeometries();
		for(geo = geos->begin(); geo != geos->end(); geo++){
			if(geo->second->GeometryUse() != BOUNDARY) continue;
			geo->second->save2file(pf);
		}
	}
	if(mbd_sim){
		geos = mbd_sim->getGeometries();
		for(geo = geos->begin(); geo != geos->end(); geo++){
			geo->second->save2file(pf);
		}
	}
	int iMin = INT_MIN;
	pf.write((char*)&iMin, sizeof(int));
	pf.close();
	return true;
}

void file_system::close()
{
// 	int CLOSE_FILESYSTME = -1;
// 	of.write((char*)&CLOSE_FILESYSTME, sizeof(int));
// 	of.close();
}
