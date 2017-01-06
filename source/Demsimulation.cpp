#include "parSIM.h"
#include "Demsimulation.h"
#include "timer.h"
#include <ctime>
#include <direct.h>
#include <iomanip>

#include "cu_dem_dec.cuh"

using namespace parSIM;

Demsimulation::Demsimulation(std::string name)
	: Simulation(name)
{

}

Demsimulation::~Demsimulation()
{

}

bool Demsimulation::RunSim()
{
// 	std::cout << "Start simulation !!" << std::endl;
// 
// 	initialize_simulation();
// 	Device == CPU ? cpu_run() : gpu_run();
// 
 	return true;
}

bool Demsimulation::initialize()
{
	std::cout << "Define Particles" << std::endl;
	ps = new particles(this);
	if(!ps->initialize()){
		std::cout << "ERROR : The particle initialize is failed." << std::endl;
		return false;
	}
	std::cout << "Done" << std::endl;

	std::cout << "Define Geometries" << std::endl;	
	for(std::map<std::string, geometry*>::iterator Geo = geometries.begin(); Geo != geometries.end(); Geo++){
		if(Geo->second->get_sim() == this && Geo->second->GeometryUse() == BOUNDARY)
			Geo->second->define_geometry();
	}
	std::cout << "Done" << std::endl;

	if(device == GPU)
 		ps->define_device_info();

	switch(integrator){
	case EULER_METHOD: itor[integrator] = new Euler_integrator(this); break;
	case VELOCITY_VERLET: itor[integrator] = new Verlet_integrator(this); break;
	}

	itor[integrator]->setNp(ps->Size());
 
	return true;
}

bool Demsimulation::initialize_simulation()
{
// 	sub_sim = ssim;
// 		std::cout << "----- Initializing -----" << std::endl;
// 	
// 		initialize_general();
 		return true;
}

void Demsimulation::cpu_run()
{
// 	timer tmer;
// 	time_t t;
// 	tm date;
// 	time(&t);
// 	localtime_s(&date, &t);
// 	std::string rpath = base_path + "result/" + CaseName;
// 	
// 	std::string spath = base_path + "result/" + specific_data;
// 	_mkdir(rpath.c_str());
// 	writer wr(this, rpath, CaseName);
// 	wr.set(DEM, BINARY);
// 	wr.save_geometry();
// 
// 	unsigned int curRun = 0;
// 	unsigned int eachRun = 0;
// 	double times = curRun * Dt;
// 	double elapsed_time = 0;
// 	//wr.start_particle_data();
// 	std::cout << "---------------------------------------------------------------------------------" << std::endl
// 		      << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |        Date       |" << std::endl
// 			  << "---------------------------------------------------------------------------------" << std::endl;
// 	std::ios::right;
// 	std::setprecision(6);// std::fixed;
// 	if(wr.run(times)){
// 		std::cout << "| " << std::setw(9) << wr.part-1 << std::setw(12) << times << std::setw(10) << eachRun <<  std::setw(11) << cRun << std::setw(15) << 0 << std::setw(21) << make_date_form(date) << std::endl;
// 	}
// 	wr.close();
// // 	if(Platform == ROCK_PLATFORM){
// // 		ps->rearrangement(cdetect);
// // 	}
// 	unsigned int nRun = static_cast<unsigned int>((sim_time / Dt) + 1);
// 	unsigned int save_iter = (nRun - 1) / static_cast<unsigned int>((sim_time / save_dt));
// 	Integrator* sub_itor = NULL;
// 	if(sub_sim){
// 		switch(sub_sim->getSolver()){
// 		case MBD:
// 			mbd_sim = dynamic_cast<Mbdsimulation*>(sub_sim);
// 		}
// 		sub_itor = sub_sim->getIntegrator();
// 	}
// 	
// 	curRun++;
// 	//itor[Solver]->integration();
// 	tmer.Start();
// 	while(curRun < nRun){	
// 		double ct = curRun * Dt;
// 		//std::cout << cRun << std::endl;
// 		itor[Itor]->integration();
// 		if(sub_itor/* && ct > 1.0*/){
// 			sub_itor->integration();
// 		}
// 		cdetect->detection(ps->Position());
// 		cforce->collision(cdetect, ps->Position(), ps->Velocity(), ps->Acceleration(), ps->AngularVelocity(), ps->AngularAcceleration());
// 
// 		if(mbd_sim/* && ct > 1.0*/){
// 			mbd_sim->oneStepAnalysis();
// 		}
// 		if(Itor==VELOCITY_VERLET)
// 			itor[Itor]->integration();
// 		if(!((curRun) % save_iter)){
// 			time(&t);
// 			localtime_s(&date, &t);
// 			tmer.Stop();
// 			
// 			if(wr.run(ct)){
// 				std::cout << "| " << std::setw(9) << wr.part-1 << std::setw(12) << std::fixed << ct << std::setw(10) << eachRun <<  std::setw(11) << curRun << std::setw(15) << tmer.GetElapsedTimeF() << std::setw(23) << make_date_form(date) << std::endl;
// 			}
// 			eachRun = 0;
// 			tmer.Start();
// 		}
//  		curRun++;
//  		eachRun++;
// 	}
// 	wr.close();
}

void Demsimulation::gpu_run()
{
// 	timer tmer;
// 	time_t t;
// 	tm date;
// 	time(&t);
// 	localtime_s(&date, &t);
// 	std::string rpath = base_path + "result/" + CaseName;
// 	_mkdir(rpath.c_str());
// 	writer wr(this, rpath, CaseName);
// 	wr.set(DEM, BINARY);
// 	wr.save_geometry();
// 	
// 	double* pos = ps->cu_Position();
// 	double* vel = ps->cu_Velocity();
// 	double* acc = ps->cu_Acceleration();
// 	double* omega = ps->cu_AngularVelocity();
// 	double* alpha = ps->cu_AngularAcceleration();
// 
// 	unsigned int cRun = 0;
// 	unsigned int eachRun = 0;
// 	double times = cRun * Dt;
// 	double elapsed_time = 0;
// 	//wr.start_particle_data();
// 	std::cout << "---------------------------------------------------------------------------------" << std::endl
// 			  << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |        Date       |" << std::endl
// 			  << "---------------------------------------------------------------------------------" << std::endl;
// 	std::ios::right;
// 	std::setprecision(6);// std::fixed;
// 	if(wr.cu_run(times)){
// 		std::cout << "| " << std::setw(9) << wr.part-1 << std::setw(12) << times << std::setw(10) << eachRun <<  std::setw(11) << cRun << std::setw(15) << 0 << std::setw(21) << make_date_form(date) << std::endl;
// 	}
// 	//wr.close();
// 	unsigned int nRun = static_cast<unsigned int>((sim_time / Dt) + 1);
// 	unsigned int save_iter = (nRun - 1) / static_cast<unsigned int>((sim_time / save_dt));
// 	Integrator* sub_itor = NULL;
// 	if(sub_sim){
// 		switch(sub_sim->getSolver()){
// 		case MBD:
// 			mbd_sim = dynamic_cast<Mbdsimulation*>(sub_sim);
// 		}
// 		sub_itor = sub_sim->getIntegrator();
// 	}
// 	cRun++;
// 	//itor[Solver]->integration();
// 	tmer.Start();
// 	while(cRun < nRun){	
// 		double ct = cRun * Dt;
// 		itor[Itor]->cu_integration();
// 		if(sub_itor/* && ct > 1.0*/){
// 			sub_itor->cu_integration();
// 		}
// 		cdetect->cu_detection(pos);
// 		cforce->cu_collision(cdetect, pos, vel, acc, omega, alpha);
// 		if(mbd_sim/* && ct > 1.0*/){
// 			mbd_sim->oneStepAnalysis();
// 		}
// 		if(Itor==VELOCITY_VERLET)
// 			itor[Itor]->cu_integration();
//  		if(!((cRun) % save_iter)){
//  			time(&t);
//  			localtime_s(&date, &t);
//  			tmer.Stop();
//  			double ct = cRun * Dt;
//  			if(wr.cu_run(ct)){
//  				std::cout << "| " << std::setw(9) << wr.part-1 << std::setw(12) << std::fixed << ct << std::setw(10) << eachRun <<  std::setw(11) << cRun << std::setw(15) << tmer.GetElapsedTimeF() << std::setw(23) << make_date_form(date) << std::endl;
//  			}
//  			eachRun = 0;
//  			tmer.Start();
//  		}
// 		cRun++;
//  		eachRun++;
// 	}
// 	wr.close();
}

void Demsimulation::Integration()
{
	device == GPU ? itor[integrator]->cu_integration() : itor[integrator]->integration();
}

void Demsimulation::cu_integrator_binding_data(double* p, double *v, double* a, double* omg, double* aph, double* f, double* m)
{
	itor[integrator]->cu_binding_data(p,v,a,omg,aph,f,m);
}
