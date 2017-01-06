#include "parSIM.h"

using namespace parSIM;

writer::writer(std::string _path, std::string _case) : part(0)
{
	file_system::caseName = _case;
	file_system::path = _path;
}

writer::~writer()
{
	file_system::close();
// 	if(of.is_open()){
// 		of.close();
// 	}
}

void writer::set(Simulation* demSim /* = 0 */, Simulation* mbdSim /* = 0 */, fileFormat _ftype /* = BINARY */)
{
	file_system::dem_sim = demSim;
	file_system::mbd_sim = mbdSim;
	file_system::fmt = _ftype;
}

void writer::start_particle_data()
{
	int type = (int)PARTICLE;
	//o/f.write((char*)&type, sizeof(int));
}

bool writer::run(double time)
{
// 	int vecSize = 4;
// 	std::fstream of;
// 	if (file_system::fmt == BINARY){
// 		char partName[256] = { 0, };
// 		color_type clr = BLUE;
// 		double radius = 0.0;
// 		sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", path.c_str(), part);
// 		of.open(partName, std::ios::out | std::ios::binary);
// 		unsigned int np = sim->getParticles()->Size();
// 		if (of.is_open()){
// 			of.write((char*)&vecSize, sizeof(int));
// 			of.write((char*)&time, sizeof(double));
// 			of.write((char*)&np, sizeof(unsigned int));
// 			algebra::vector4<double> *p = sim->getParticles()->Position();
// 			algebra::vector4<double> *v = sim->getParticles()->Velocity();
// 
// 			of.write((char*)&clr, sizeof(int));
// 			of.write((char*)&radius, sizeof(double));
// 			of.write((char*)p, sizeof(double) * 4 * np);
// 			of.write((char*)v, sizeof(double) * 4 * np);
// 		}

// 		Simulation *subSim = sim->getSubSimulation();
// 		if(subSim)
// 		{
// 			switch(subSim->getSolver()){
// 			case MBD:
// 				{
// 					std::map<std::string, pointmass*>::iterator pm = subSim->getMasses()->begin();
// 					if(pm != subSim->getMasses()->end()){
// 						for(; pm != subSim->getMasses()->end(); pm++){
// 							int type = MASS;
// 							of.write((char*)&type, sizeof(int));
// 						//of.write((char*)&time, sizeof(double));
// 							pm->second->save2file(of);
// 						}
// 					}
// 				}
// 				break;
// 			}
// 		}
// 	}
// 	part++;
// 	of.close();
// 	else
// 	{
// 
// 	}
// 	part++;
// 	of.close();
// 	return true;
// 	int START_STEP_DATA = INT_MIN;
// 	of.write((char*)&START_STEP_DATA, sizeof(int));
// 	of.write((char*)&time, sizeof(double));	
// 	char partName[256] = {0, };
// 	sprintf_s(partName, sizeof(char)*256, "part%04d", part);
// 	if(fmt==ASCII){
// 		std::string file_path = path + "/" + caseName + ".txt"; 
// 		of.open(file_path, std::ios::out);
// 	}
// 	else
// 	{
// 		particles *ps = sim->getParticles();
// 		particle* pars = sim->getParticle();
// 
// 		if(ps){
// 			int type = (int)PARTICLE;
// 			of.write((char*)&type, sizeof(int));
// 			unsigned int np = ps->Size();
// 			of.write((char*)&np, sizeof(unsigned int));
// 			for(unsigned int i = 0; i < np; i++){
// 				vector4<double> tp = vector4<double>(pars[i].Position().x, pars[i].Position().y, pars[i].Position().z, pars[i].Radius());
// 				of.write((char*)&tp, sizeof(double) * 4);
// 			}
// 			for(unsigned int i = 0; i < np; i++){
// 				vector4<double> tv = vector4<double>(pars[i].Velocity().x, pars[i].Velocity().y, pars[i].Velocity().z, -1.0);
// 				of.write((char*)&tv, sizeof(double) * 4);
// 			}
// // 			algebra::vector4<double> *p = ps->Position();
// // 			algebra::vector4<double> *v = ps->Velocity();
// // 			//std::string file_path = path + "/" + partName + ".bin"; 
// // 			//of.open(file_path, std::ios::out | std::ios::binary);
// // 			of.write((char*)&np, sizeof(unsigned int));
// // 				
// // 			of.write((char*)p, sizeof(double) * 4 * np);
// // 			of.write((char*)v, sizeof(double) * 4 * np);
// 		}
// 		
// 		Simulation *subSim = sim->getSubSimulation();
// 		if(subSim)
// 		{
// 			switch(subSim->getSolver()){
// 			case MBD:{
// 				std::map<std::string, pointmass*>::iterator pm = subSim->getMasses()->begin();
// 				if(pm != subSim->getMasses()->end()){
// 					for(; pm != subSim->getMasses()->end(); pm++){
// 						int type = MASS;
// 						of.write((char*)&type, sizeof(int));
// 						//of.write((char*)&time, sizeof(double));
// 						pm->second->save2file(of);
// 					}
// 				}
// 					 }
// 				break;
// 			}
// 		}
// 		
// 	}
// 	int END_STEP_DATA = INT_MAX;
// 	of.write((char*)&END_STEP_DATA, sizeof(int));
// 	part++;
	return true;
}

bool writer::cu_run(double time)
{
	int vecSize = 4;
	std::fstream of;
	if (file_system::fmt == BINARY){
		char partName[256] = { 0, };
		color_type clr = BLUE;
		double radius = 0.0;
		sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", path.c_str(), part);
		of.open(partName, std::ios::out | std::ios::binary);
		unsigned int fsize = sizeof(double);
		of.write((char*)&fsize, sizeof(unsigned int));
		if(dem_sim){
			unsigned int np = dem_sim->getParticles()->Size();
			if(of.is_open()){
				of.write((char*)&vecSize, sizeof(int));
				of.write((char*)&time, sizeof(double));
				of.write((char*)&np, sizeof(unsigned int));
				algebra::vector4<double> *p = dem_sim->getParticles()->Position();
				algebra::vector4<double> *v = dem_sim->getParticles()->Velocity();

				checkCudaErrors( cudaMemcpy(p, dem_sim->getParticles()->cu_Position(), sizeof(double) * 4 * np, cudaMemcpyDeviceToHost) );
				checkCudaErrors( cudaMemcpy(v, dem_sim->getParticles()->cu_Velocity(), sizeof(double) * 4 * np, cudaMemcpyDeviceToHost) );

				of.write((char*)&clr, sizeof(int));
				of.write((char*)&radius, sizeof(double));
				of.write((char*)p, sizeof(double) * 4 * np);
				of.write((char*)v, sizeof(double) * 4 * np);
				if(dem_sim->getParticles()->cu_IsLineContact())
				{
					bool* isShapContact = new bool[np];
					checkCudaErrors( cudaMemcpy(isShapContact, dem_sim->getParticles()->cu_IsLineContact(), sizeof(bool)*np, cudaMemcpyDeviceToHost));
					of.write((char*)isShapContact, sizeof(bool)*np);
				}
			}
		}
		
		if(mbd_sim){
			std::map<std::string, pointmass*>::iterator mass = mbd_sim->getMasses()->begin();
			if(mass != mbd_sim->getMasses()->end()){
				for(; mass != mbd_sim->getMasses()->end(); mass++){
					if(!mass->second->ID()) continue;
					int type = MASS;
					of.write((char*)&type, sizeof(int));
					mass->second->save2file(of);
				}
			}
		}

		//Simulation *subSim = sim->getSubSimulation();
// 		if(subSim)
// 		{
// 			switch(subSim->getSolver()){
// 			case MBD:
// 				{
// 					std::map<std::string, pointmass*>::iterator pm = subSim->getMasses()->begin();
// 					if(pm != subSim->getMasses()->end()){
// 						for(; pm != subSim->getMasses()->end(); pm++){
// 							int type = MASS;
// 							of.write((char*)&type, sizeof(int));
// 							//of.write((char*)&time, sizeof(double));
// 							pm->second->save2file(of);
// 						}
// 					}
// 				}
// 				break;
// 			}
// 		}
	}
	part++;
	of.close();
// 	int START_STEP_DATA = INT_MIN;
// 	of.write((char*)&START_STEP_DATA, sizeof(int));
// 	of.write((char*)&time, sizeof(double));	
// 	if(fmt==ASCII){
// 
// 	}
// 	else
// 	{
// 		int type = (int)PARTICLE;
// 		of.write((char*)&type, sizeof(int));
// 
// 		particles *ps = sim->getParticles();
// 		unsigned int np = ps->Size();
// 		algebra::vector4<double> *p = ps->Position();
// 		algebra::vector4<double> *v = ps->Velocity();
// 		checkCudaErrors( cudaMemcpy(p, ps->cu_Position(), sizeof(double) * 4 * np, cudaMemcpyDeviceToHost) );
// 		checkCudaErrors( cudaMemcpy(v, ps->cu_Velocity(), sizeof(double) * 4 * np, cudaMemcpyDeviceToHost) );
// 		of.write((char*)&np, sizeof(unsigned int));
// 			
// 		of.write((char*)p, sizeof(double) * 4 * np);
// 		of.write((char*)v, sizeof(double) * 4 * np);
// 
// 		Simulation *subSim = sim->getSubSimulation();
// 		if(subSim)
// 		{
// 			switch(subSim->getSolver()){
// 			case MBD:{
// 				std::map<std::string, pointmass*>::iterator pm = subSim->getMasses()->begin();
// 				if(pm != subSim->getMasses()->end()){
// 					for(; pm != subSim->getMasses()->end(); pm++){
// 						int type = MASS;
// 						of.write((char*)&type, sizeof(int));
// 						//of.write((char*)&time, sizeof(double));
// 						pm->second->save2file(of);
// 					}
// 				}
// 					 }
// 					 break;
// 			}
// 		}
// 	}
// 	int END_STEP_DATA = INT_MAX;
// 	of.write((char*)&END_STEP_DATA, sizeof(int));
// 	part++;
 	return true;
}