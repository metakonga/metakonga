#include "CouplingSystem.h"
#include "parSIM.h"
#include "cu_dem_dec.cuh"
#include <direct.h>

using namespace parSIM;

CouplingSystem::CouplingSystem(Demsimulation* dem, Mbdsimulation* mbd)
	: Dem(dem)
	, Mbd(mbd)
{
	
}

CouplingSystem::~CouplingSystem()
{

}

bool CouplingSystem::ModifyRun(unsigned int nf)
{
	//cell_grid cdetect(Dem, Mbd);
	//cdetect.setWorldOrigin(0.0, 0.0, 0.0);
	//cdetect.setGridSize(128, 128, 128);
	//cdetect.initialize();

	//force cforce(Dem, Mbd);
	//cforce.initialize();

	//if(!Simulation::dt){
	//	//Simulation::dt = 1;
	//	contact_coefficient coe = cforce.getCoefficient("particle");//coefficients.find(NULL);
	//	double ks = coe.ks;
	//	algebra::vector3<double> gravity = cforce.Gravity();
	//	for(unsigned int i = 0; i < Dem->getParticles()->Size(); i++){
	//		double rad = Dem->getParticles()->Position()[i].w;
	//		double temp = 2 * PI * sqrt(particles::inertia / (12 * ks * rad * rad));
	//		if(Simulation::dt > temp) 
	//			Simulation::dt = temp;
	//		Dem->getParticles()->Acceleration()[i] = algebra::vector4<double>(gravity.x, gravity.y, gravity.z, Dem->getParticles()->Acceleration()[i].w);
	//	}
	//}
	//std::cout << "Simulation time step is " << Simulation::dt << std::endl;
	//Mbd->getPredictor().getTimeStep() = Simulation::dt;
	//Mbd->setIntParameter();

	//if(Dem->Device() == GPU){
	//	device_parameters paras;
	//	paras.np = Dem->getParticles()->Size();
	//	paras.nsp = cdetect.getNumShapeVertex();
	//	paras.dt = Simulation::dt;
	//	paras.half2dt = 0.5*paras.dt*paras.dt;
	//	paras.gravity = make_double3(cforce.Gravity().x, cforce.Gravity().y, cforce.Gravity().z);
	//	paras.cell_size = cdetect.getCellSize();
	//	paras.ncell = cdetect.getNumCell();
	//	paras.grid_size = make_uint3(cdetect.GridSize().x, cdetect.GridSize().y, cdetect.GridSize().z);
	//	paras.world_origin = make_double3(cdetect.WorldOrigin().x, cdetect.WorldOrigin().y, cdetect.WorldOrigin().z);
	//	contact_coefficient ppc = cforce.getCoefficient("particle");
	//	paras.kn = ppc.kn;
	//	paras.vn = ppc.vn;
	//	paras.ks = ppc.ks;
	//	paras.vs = ppc.vs;
	//	paras.mu = ppc.mu;
	//	paras.cohesive = force::cohesive;
	//	setSymbolicParameter(&paras);
	//	Dem->cu_integrator_binding_data(Dem->getParticles()->cu_Position(), Dem->getParticles()->cu_Velocity(), Dem->getParticles()->cu_Acceleration(), Dem->getParticles()->cu_AngularVelocity(), Dem->getParticles()->cu_AngularAcceleration(), cforce.cu_Force(), cforce.cu_Moment());
	//	for(std::map<std::string, geometry*>::iterator Geo = Dem->getGeometries()->begin(); Geo != Dem->getGeometries()->end(); Geo++){
	//		if(Geo->second->GeometryUse() != BOUNDARY) continue;
	//		Geo->second->define_device_info();
	//	}
	//	if(Mbd)
	//		for(std::map<std::string, geometry*>::iterator Geo = Mbd->getGeometries()->begin(); Geo != Mbd->getGeometries()->end(); Geo++)
	//			Geo->second->define_device_info();
	//}

	//std::string pfile = Simulation::base_path + Simulation::caseName;
	//vector4<double>* pos = new vector4<double>(Dem->getParticles()->Size());
	//vector4<double>* vel = new vector4<double>(Dem->getParticles()->Size());
	//std::fstream of;
	//for(unsigned int i = 1; i < nf; i++){
	//	char partName[256] = { 0, };
	//	color_type clr = BLUE;
	//	double radius = 0.0;
	//	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", pfile.c_str(), i);
	//	of.open(partName, std::ios::in | std::ios::binary);
	//	unsigned int fdtype = 0;
	//	int vecsize = 0;
	//	double time = 0;
	//	double radius = 0;
	//	unsigned int tnp = 0;
	//	of.read((char*)&vecsize, sizeof(int));
	//	of.read((char*)&time, sizeof(double));
	//	of.read((char*)&tnp, sizeof(unsigned int));
	//	if(Dem->getParticles()->Size() != tnp)
	//	{
	//		std::cout << "Different particle size." << std::endl;
	//		delete [] pos;
	//		delete [] vel;
	//		return false;
	//	}
	//	of.read((char*)&vecsize, sizeof(int));
	//	of.read((char*)&radius, sizeof(double));
	//	of.read((char*)pos, sizeof(vector4<double>)*tnp);
	//	of.read((char*)vel, sizeof(vector4<double>)*tnp);
	//	checkCudaErrors( cudaMemcpy(Dem->getParticles()->cu_Position(), pos, sizeof(vector4<double>)*tnp, cudaMemcpyHostToDevice) );
	//	checkCudaErrors( cudaMemcpy(Dem->getParticles()->cu_Velocity(), vel, sizeof(vector4<double>)*tnp, cudaMemcpyHostToDevice) );

	//	cdetect.detection();					// 입자-입자, 입자-벽, 입자-전개판 접촉 판별
	//	cforce.collision(&cdetect);				// 접촉력 계산


	//	of.close();
	//}

	return true;
}

bool CouplingSystem::Run()
{
	if(!Dem && !Mbd){
		std::cout << "ERROR : No exist simulation object. ---- Demsimulation* dem = " << Dem << ", Mbdsimulation* mbd = " << Mbd << std::endl;
		return false;
	}

	cell_grid cdetect(Dem, Mbd);
	cdetect.setWorldOrigin(0.0, 0.0, 0.0);
	cdetect.setGridSize(128, 128, 128);
	cdetect.initialize();

	force cforce(Dem, Mbd);
	cforce.initialize();

	if(!Simulation::dt){
		//Simulation::dt = 1;
		contact_coefficient coe = cforce.getCoefficient("particle");//coefficients.find(NULL);
		double ks = coe.ks;
		algebra::vector3<double> gravity = cforce.Gravity();
		for(unsigned int i = 0; i < Dem->getParticles()->Size(); i++){
			double rad = Dem->getParticles()->Position()[i].w;
			double temp = 2 * PI * sqrt(particles::inertia / (12 * ks * rad * rad));
			if(Simulation::dt > temp) 
				Simulation::dt = temp;
			Dem->getParticles()->Acceleration()[i] = algebra::vector4<double>(gravity.x, gravity.y, gravity.z, Dem->getParticles()->Acceleration()[i].w);
		}
	}
	std::cout << "Simulation time step is " << Simulation::dt << std::endl;
	Mbd->getPredictor().getTimeStep() = Simulation::dt;
	Mbd->setIntParameter();

	if(Dem->Device() == GPU){
		device_parameters paras;
		paras.np = Dem->getParticles()->Size();
		paras.nsp = cdetect.getNumShapeVertex();
		paras.dt = Simulation::dt;
		paras.half2dt = 0.5*paras.dt*paras.dt;
		paras.gravity = make_double3(cforce.Gravity().x, cforce.Gravity().y, cforce.Gravity().z);
		paras.cell_size = cdetect.getCellSize();
		paras.ncell = cdetect.getNumCell();
		paras.grid_size = make_uint3(cdetect.GridSize().x, cdetect.GridSize().y, cdetect.GridSize().z);
		paras.world_origin = make_double3(cdetect.WorldOrigin().x, cdetect.WorldOrigin().y, cdetect.WorldOrigin().z);
		contact_coefficient ppc = cforce.getCoefficient("particle");
		paras.kn = ppc.kn;
		paras.vn = ppc.vn;
		paras.ks = ppc.ks;
		paras.vs = ppc.vs;
		paras.mu = ppc.mu;
		paras.cohesive = force::cohesive;
		setSymbolicParameter(&paras);
		Dem->cu_integrator_binding_data(Dem->getParticles()->cu_Position(), Dem->getParticles()->cu_Velocity(), Dem->getParticles()->cu_Acceleration(), Dem->getParticles()->cu_AngularVelocity(), Dem->getParticles()->cu_AngularAcceleration(), cforce.cu_Force(), cforce.cu_Moment());
		for(std::map<std::string, geometry*>::iterator Geo = Dem->getGeometries()->begin(); Geo != Dem->getGeometries()->end(); Geo++){
			if(Geo->second->GeometryUse() != BOUNDARY) continue;
			Geo->second->define_device_info();
		}
		if(Mbd)
			for(std::map<std::string, geometry*>::iterator Geo = Mbd->getGeometries()->begin(); Geo != Mbd->getGeometries()->end(); Geo++)
				Geo->second->define_device_info();
	}
	if(Simulation::specific_data != ""){
		Mbd->ShapeDataUpdate();
	}
	unsigned int cStep = 0;
	unsigned int eachStep = 0;
	unsigned int nStep = static_cast<unsigned int>((Simulation::sim_time / Simulation::dt) + 1);
	timer tmer;
	time_t t;
	tm date;
	std::time(&t);
	localtime_s(&date, &t);
	std::string rpath = Simulation::base_path + "result/" + Simulation::caseName;
	_mkdir(rpath.c_str());
	writer wr(rpath, Simulation::caseName);
	wr.set(Dem, Mbd, BINARY);
	wr.save_geometry();
	double times = cStep * Simulation::dt;
	double elapsed_time = 0;
	std::cout << "---------------------------------------------------------------------------------" << std::endl
			  << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |        Date       |" << std::endl
		      << "---------------------------------------------------------------------------------" << std::endl;
	std::ios::right;
	std::setprecision(6);
 	if(wr.cu_run(times)){
 		std::cout << "| " << std::setw(9) << wr.part-1 << std::setw(12) << times << std::setw(10) << eachStep <<  std::setw(11) << cStep << std::setw(15) << 0 << std::setw(21) << make_date_form(date) << std::endl;
 	}
	cStep++;
	tmer.Start();
	while(nStep > cStep){
		times = cStep * Simulation::dt;			// 시간 갱신
		Dem->Integration();						// DEM 입자들의 위치 계산
		cdetect.detection();					// 입자-입자, 입자-벽, 입자-전개판 접촉 판별
		cforce.collision(&cdetect);				// 접촉력 계산
		Mbd->Prediction(cStep);					// MBD 예측 계산 수행
		Dem->Integration();						// DEM 입자들의 속도 계산
		Mbd->Correction(cStep);					// MBD 수정 계산 수행(Newton-Raphson 알고리즘)
		if(!((cStep) % Simulation::save_step)){
			time(&t);
			localtime_s(&date, &t);
			tmer.Stop();
			double ct = cStep * Simulation::dt;
			if(wr.cu_run(ct)){					// 결과 저장
				std::cout << "| " << std::setw(9) << wr.part-1 << std::setw(12) << std::fixed << ct << std::setw(10) << eachStep <<  std::setw(11) << cStep << std::setw(15) << tmer.GetElapsedTimeF() << std::setw(23) << make_date_form(date) << std::endl;
			}
			eachStep = 0;
			tmer.Start();
		}
		cStep++;
		eachStep++;
	}
	wr.close();
	return true;
}