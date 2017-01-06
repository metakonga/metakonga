#include "parSIM.h"
#include "timer.h"
#include <ctime>
#include <direct.h>
#include <fstream>

using namespace parSIM;

std::string Simulation::specific_data = "";
std::string Simulation::base_path = "";
std::string Simulation::caseName = "";
dimension_type Simulation::dimension = DIM_2;
precision_type Simulation::float_type = DOUBLE_PRECISION;
solver_type Simulation::solver = DEM;
unsigned int Simulation::save_step = 0;
double Simulation::sim_time = 0.0;
double Simulation::time = 0.0;
double Simulation::dt = 1e-5;
vector3<double> Simulation::gravity = vector3<double>(0.0, -9.80665, 0.0);
unsigned int Simulation::cStep = 0;

Simulation::Simulation(std::string name)
	: Name(name)
	, cforce(NULL)
	, cdetect(NULL)
{

}

Simulation::~Simulation()
{
	clear();
}

void parSIM::Simulation::clear()
{
// 	for(int i = 0; i < NUM_INTEGRATOR; i++){
// 		if(itor[i]){
// 			delete itor[i];
// 			itor[i] = NULL;
// 		}
// 	}
	if(cforce) delete cforce; cforce = NULL;
	if(cdetect) delete cdetect; cdetect = NULL;

	if(geometries.size()){
		for(std::map<std::string, geometry*>::iterator Geo = geometries.begin(); Geo != geometries.end(); Geo++){
			delete Geo->second;
		}
	}
	if(masses.size()){
		for(std::map<std::string, pointmass*>::iterator Mass = masses.begin(); Mass != masses.end(); Mass++){
			delete Mass->second;
		}
	}
	if(kinConsts.size()){
		for(std::map<std::string, kinematicConstraint*>::iterator kinc = kinConsts.begin(); kinc != kinConsts.end(); kinc++){
			delete kinc->second;
		}
	}
	if(driConsts.size()){
		for(std::map<std::string, drivingConstraint*>::iterator dric = driConsts.begin(); dric != driConsts.end(); dric++){
			delete dric->second;
		}
	}
}

unsigned int parSIM::Simulation::getMassSize()
{
// 	if(masses){
// 		return masses->size();
// 	}
// 	else if(sub_sim->getMasses()){
// 		return sub_sim->getMasses()->size();
// 	}
	return 0;
}

geometry* parSIM::Simulation::getGeometry(std::string val)
{
	std::map<std::string, geometry*>::iterator geo = geometries.find(val);
	if(geo == geometries.end()){
		Log::Send(Log::Error, "Function : geometry* parSIM::Simulation::getGeometry(std::string val) - No exist geometry for " + val);
		checkErrors( ReturnNULL );
	}
	return geo->second;
}

void parSIM::Simulation::insert_geometry(std::string _name, geometry* _geo)
{
	geometries[_name] = _geo;
	//delete _geo;
}

void parSIM::Simulation::insert_pointmass(std::string _name, pointmass* _mass)
{
	masses[_name] = _mass;
}

// void parSIM::Simulation::setIntegrator(integrator_type integrator)
// {
// // 	Itor = integrator;
// // 	//Euler_integrator* e = new Euler_integrator;
// // 	switch(Itor){
// // 	case EULER_METHOD: itor[Itor] = new Euler_integrator(this); break;
// // 	case VELOCITY_VERLET: itor[Itor] = new Verlet_integrator(this); break;
// // 	}
// }

void parSIM::Simulation::add_pair_material_condition(int m1, int m2/* =0 */)
{
// 	if(!m2)
// 		m2 = ps->getMaterial();
// 	/*pair_material_type[m1] = m2;*/
// 
// 	Log::Send(Log::Info, "add_pair_material_condition : [" + material_enum2str(m1) + ", " + material_enum2str(m2) + "]");
}

void parSIM::Simulation::setSpecificData(std::string spath)
{
	std::fstream pf;
	pf.open(spath + specific_data, std::ios::in | std::ios::binary);
	if(pf.is_open()){
		while(1){
			int type;
			pf.read((char*)&type, sizeof(int));
			switch (type)
			{
			case -1:
				return;
			case PARTICLE:
				ps->setSpecificDataFromFile(pf);	
				break;
			case SHAPE:{
				unsigned int name_size = 0;
				pf.read((char*)&name_size, sizeof(unsigned int));
				char cname[256] = {0, };
				pf.read(cname, sizeof(char)*name_size);
				std::string stdname = cname;
				std::map<std::string, geometry*>::iterator Geo = geometries.find(stdname);
				if(Geo->second->GeometryUse() != BOUNDARY)
					break;
				Geo->second->setSpecificDataFromFile(pf);
				//delete [] cname;
				}
				break;
			default:
				break;
			}
		}
	}
	else
	{
		Log::Send(Log::Error, "No exist specific_data. The path is " + spath + specific_data);
		exit(0);
	}
}

// bool Simulation::Run(Demsimulation* dem /* = 0 */, Mbdsimulation* mbd /* = 0 */)
//{
//	if(!dem && !mbd){
//		std::cout << "ERROR : No exist simulation object. ---- Demsimulation* dem = " << dem << ", Mbdsimulation* mbd = " << mbd << std::endl;
//		return false;
//	}
//
//	if(dem && mbd)
//		this->RunCoupledAnalysis(dem, mbd);
//
//// 	unsigned int nStep = static_cast<unsigned int>((sim_time / dt) + 1);
//// 	timer tmer;
//// 	time_t t;
//// 	tm date;
//// 	std::time(&t);
//// 	localtime_s(&date, &t);
//// 	std::string rpath = base_path + "result/" + caseName;
//// 	_mkdir(rpath.c_str());
//}
//
//bool Simulation::RunCoupledAnalysis(Demsimulation* dem, Mbdsimulation* mbd)
//{
//
//}