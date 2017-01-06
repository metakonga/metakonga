#include "s_writer.h"
#include "s_geometry.h"
#include "s_particle.h"
#include <stdlib.h>
#include <stdio.h>
#include <direct.h>

using namespace parsph;

writer::writer(sphydrodynamics* _sph, std::string directory)
	: sph(_sph)
	, dir(directory)
	, partCount(0)
{

}

writer::~writer()
{

}

bool writer::initialize()
{	
	std::string path = dir + sph->Name();
	std::fstream pf;
	pf.open(path + "/check.bin", std::ios::in | std::ios::binary);
	if(!pf.is_open()){
		_mkdir(path.c_str());
		pf.close();
		pf.open(path + "/check.bin", std::ios::out | std::ios::binary);
	}
	char v;
	switch(tdata){
	case ASCII: v = 't'; pf.write(&v, sizeof(char)); extension = "txt"; break;
	case BINARY: v = 'b'; pf.write(&v, sizeof(char)); extension = "bin"; break;
	case VTK: v = 'v'; pf.write(&v, sizeof(char)); extension = "vtk"; break;
	default:
		return false;
	}
	int tcount = 0;
	if(exportVariable[POSITION]){
		v = 'o';
		pf.write(&v, sizeof(char));
	}
	if(exportVariable[PRESSURE]){
		char r = 'r';
		pf.write(&r, sizeof(char));
	}
	if(exportVariable[VELOCITY]){
		char e = 'e';
		pf.write(&e, sizeof(char));
	}
	pf.close();

	return true;
}

bool writer::exportBoundaryData()
{
	std::fstream pf;
	pf.open(dir + sph->Name() + "/Boundary.bin", std::ios::out | std::ios::binary);
	for(std::multimap<std::string, Geometry*>::iterator it = sph->models.begin(); it != sph->models.end(); it++){
		if(it->second->Type() == BOUNDARY){
			it->second->Export(pf);
		}
	}
	return true;
}

bool writer::exportVariableData(t_device dev/*, double time*/)
{
	char v;
	std::fstream pf;
	char path[256] = {0, };
	sprintf_s(path, 256, "%s/part%04d.%s", (dir + sph->Name()).c_str(), partCount, extension.c_str());
	if(tdata != ASCII)
		pf.open(path, std::ios::out | std::ios::binary);
	else
		pf.open(path, std::ios::out);
	bool *h_fsurface = new bool[sph->ParticleCount()];
	if(dev == GPU){
		vector3<double> *h_pos = new vector3<double>[sph->ParticleCount()];
		vector3<double> *h_vel = new vector3<double>[sph->ParticleCount()];
		double *h_pressure = new double[sph->ParticleCount()];
		double *h_divP = new double[sph->ParticleCount()];
		

		checkCudaErrors( cudaMemcpy(h_pos, sph->d_pos, sizeof(vector3<double>)*sph->ParticleCount(), cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_vel, sph->d_vel, sizeof(vector3<double>)*sph->ParticleCount(), cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_pressure, sph->d_pressure, sizeof(double)*sph->ParticleCount(), cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_fsurface, sph->d_free_surface, sizeof(bool)*sph->ParticleCount(), cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_divP, sph->d_divP, sizeof(double)*sph->ParticleCount(), cudaMemcpyDeviceToHost) );

		for(unsigned int i = 0; i < sph->ParticleCount(); i++){
			s_particle *parI = sph->getParticle(i);
			parI->setPosition(vector3<double>(h_pos[i].x, h_pos[i].y, h_pos[i].z));
			parI->setVelocity(vector3<double>(h_vel[i].x, h_vel[i].y, h_vel[i].z));
			parI->setPressure(h_pressure[i]);
			parI->setFreeSurface(h_fsurface[i]);
			parI->setDivP(h_divP[i]);
		}
		
		delete [] h_pos;
		delete [] h_vel;
		delete [] h_pressure;
	}

	if(tdata == BINARY){
		s_particle* par;
		std::fstream ppf;
		//ppf.open("C:/C++/dam_dry_bed.txt", std::ios::out);
		unsigned int np = sph->ParticleCount();
		//pf.write((char*)&time, sizeof(double));
		pf.write((char*)&np, sizeof(unsigned int));
		//pf.write((char*)h_fsurface, sizeof(bool) * np);
		for(unsigned int i = 0; i < sph->ParticleCount(); i++){
			par = sph->getParticle(i);
			switch(par->Type()){
			case FLUID: v = 'f'; pf.write(&v, sizeof(char)); break;
			case BOUNDARY: v = 'b'; pf.write(&v, sizeof(char)); break;
			case DUMMY: v = 'd'; pf.write(&v, sizeof(char)); break;
			}
			if(v == 'd'){
				bool pause = true;
			}
			pf.write((char*)&par->FreeSurface(), sizeof(bool));
			pf.write((char*)&par->DivP(), sizeof(double));
			if(exportVariable[POSITION])
				pf.write((char*)&par->Position(), sizeof(double)*3);
			if(exportVariable[PRESSURE])
				pf.write((char*)&par->Pressure(), sizeof(double));
			if(exportVariable[VELOCITY])
				pf.write((char*)&par->Velocity(), sizeof(double)*3);
		}
		//ppf.close();
	}
	pf.close();
	
	partCount++;

	return true;
}
