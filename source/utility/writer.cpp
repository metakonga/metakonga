#include "writer.h"
#include "Geometry.h"
#include "Simulation.h"
#include "ball.h"
#include <direct.h>

using namespace utility;

unsigned int writer::part = 0;
char writer::solverType = '0';
char writer::fileFormat = '0';
std::string writer::subDirectory = "";
std::string writer::directory = "";
std::fstream writer::of;
std::fstream writer::pf_of;
vector3<double> writer::pick_force = 0.0;
Simulation* writer::sim = NULL;

writer::writer()
{
	
}

writer::~writer()
{
	
}

void writer::SetSimulation(Simulation* baseSimulation)
{
	pf_of.open("C:/C++/pforce.txt", std::ios::out);
	sim = baseSimulation;
	directory = sim->BasePath() + sim->CaseName() + "/";
	_mkdir(directory.c_str());
}

void writer::EndSimulation()
{
	pf_of.close();
}

void writer::SaveGeometry()
{
	std::string filename = directory + "/boundary";
	if (!of.is_open()){
		if (fileFormat == 'b'){
			filename += ".bin";
			of.open(filename, std::ios::binary | std::ios::out);
		}
		else if (fileFormat == 'a'){
			filename += ".txt";
			of.open(filename, std::ios::out);
		}
		else{
			filename += ".txt";
			of.open(filename, std::ios::out);
		}
	}
	std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin();
	for (; it != sim->Geometries().end(); it++){
		if (it->second->Type() != GEO_PARTICLE)
			it->second->save2file(of, fileFormat);
	}
	for (std::map<std::string, Object*>::iterator obj = sim->Objects().begin(); obj != sim->Objects().end(); obj++){
		obj->second->save2file(of, fileFormat);
	}
	int iMin = INT_MIN;
	of.write((char*)&iMin, sizeof(int));
	of.close();
}

bool writer::Save(unsigned int step)
{
	if (fileFormat == 'b'){
		int vecSize = 3;
		double tme = step * Simulation::dt;
		char partName[256] = { 0, };
		sprintf_s(partName, sizeof(char) * 256, "%spart%04d.bin", (directory + subDirectory).c_str(), part);
		of.open(partName, std::ios::out | std::ios::binary);
		if (of.is_open()){
			of.write((char*)&vecSize, sizeof(int));
			of.write((char*)&tme, sizeof(double));
			of.write((char*)&ball::nballs, sizeof(unsigned int));
			for (unsigned int i = 0; i < ball::nballs; i++){
				ball* b = &sim->Balls()[i];
				of.write((char*)&b->Color(), sizeof(int));
				of.write((char*)&b->Radius(), sizeof(double));
				of.write((char*)&b->Position(), sizeof(vector3<double>));
				of.write((char*)&b->Velocity(), sizeof(vector3<double>));
			}
			for (std::map<std::string, Object*>::iterator obj = sim->Objects().begin(); obj != sim->Objects().end(); obj++){
				vector3<double> center = obj->second->CenterOfGravity();
				int type = MASS;
				vector3<double> v(0.0);
				of.write((char*)&type, sizeof(int));
				int name_size = obj->second->Name().size();
				of.write((char*)&name_size, sizeof(int));
				of.write((char*)obj->second->Name().c_str(), sizeof(char) * name_size);
				of.write((char*)&center, sizeof(vector3<double>));
				of.write((char*)&v.x, sizeof(vector3<double>));
			}
			for (std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin(); it != sim->Geometries().end(); it++){
				if (it->second->Shape() == SHAPE){
					vector3<double> vel = 0.0;
					vector3<double> force = 0.0;
					int type = MASS;
					of.write((char*)&type, sizeof(int));
					geo::Shape* sh = dynamic_cast<geo::Shape*>(it->second);
					int name_size = sh->Name().size();
					of.write((char*)&name_size, sizeof(int));
					of.write((char*)sh->Name().c_str(), sizeof(char) * name_size);
					of.write((char*)&sh->Position(), sizeof(vector3<double>));
					of.write((char*)&vel, sizeof(vector3<double>));
					of.write((char*)&force, sizeof(vector3<double>));
				}
			}
		}
		else{
			std::cout << "ERROR : bool writer::Save(unsigned int " << step << ") - File system is not opened." << std::endl;
			return false;
		}	
	}
	else
	{
		
	}
	pf_of << pick_force.x << " " << pick_force.y << " " << pick_force.z << std::endl;
	part++;
	of.close();
	return true;
}

void writer::SetFileSystem(std::string subdir)
{
	subDirectory = subdir + "/";
	std::string tdir = directory + subDirectory;
	_mkdir(tdir.c_str());
	if (of.is_open()){
		of.close();
	}	
	part = 0;
}

void writer::CloseFileSystem()
{
	if (of.is_open())
		of.close();
}