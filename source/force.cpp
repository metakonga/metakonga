#include "parSIM.h"
#include "cu_dem_dec.cuh"

using namespace parSIM;

double force::cohesive = 0.0;

force::force(Simulation* Sim)
	: sim(Sim)
	, m_force(NULL), d_force(NULL)
	, m_moment(NULL), d_moment(NULL)
{
	gravity = Simulation::gravity;
	if(!gravity.x && !gravity.y && !gravity.z)
		std::cout << "...............................WARNING : Not define the gravity. " << std::endl;
}

force::force(Simulation* demSim, Simulation *mbdSim)
	: dem_sim(demSim)
	, mbd_sim(mbdSim)
	, m_force(NULL), d_force(NULL)
	, m_moment(NULL), d_moment(NULL)
{
	gravity = Simulation::gravity;
	if(!gravity.x && !gravity.y && !gravity.z)
		std::cout << "...............................WARNING : Not define the gravity. " << std::endl;
}

force::~force()
{
	if(m_force) delete [] m_force; m_force = NULL;
	if(m_moment) delete [] m_moment; m_moment = NULL;
	if(d_force) checkCudaErrors( cudaFree(d_force) ); d_force = NULL;
	if(d_moment) checkCudaErrors( cudaFree(d_moment) ); d_moment = NULL;
}

void force::initialize()
{
	if(dem_sim){
		particles* ps = dem_sim->getParticles();
		double cor = 0.93;
		double pmass = particles::mass;
		double pradius =particles::radius;
		cmaterialType pm = ps->getMaterialParameters();
		double eym = pm.youngs / (2 * (1 - pm.poisson*pm.poisson));
		double er = (pradius * pradius) / (pradius + pradius);
		double em = (pmass * pmass) / (pmass + pmass);
		double beta=(PI/log(cor));

		contact_coefficient cc;
		cc.kn = (4/3)*sqrt(er)*eym;
		cc.vn = sqrt((4 * em * cc.kn) / (1 + beta * beta));
		cc.ks = cc.kn;
		cc.vs = cc.vn;
		cc.mu = 0.35;
		/*contact_coefficient pc = cc;*/
		coefficients["particle"] = cc;

		for(std::map<std::string, geometry*>::iterator Geo = dem_sim->getGeometries()->begin(); Geo != dem_sim->getGeometries()->end(); Geo++){
			if(Geo->second->GeometryUse() != BOUNDARY) continue;
			double w_cor = 0.1;
			double w_beta=(PI/log(w_cor));
			cmaterialType bm = Geo->second->getMaterialParameters();
			double ewpy = (pm.youngs * bm.youngs)/(pm.youngs*(1 - bm.poisson*bm.poisson) + bm.youngs*(1 - pm.poisson*pm.poisson));
			cc.kn = (4/3)*sqrt(pradius)*ewpy;
			cc.vn = sqrt((4 * pmass * cc.kn) / (1 + w_beta * w_beta));
			cc.ks = cc.kn * 0.8;
			cc.vs = cc.vn;
			cc.mu = 0.3;
			coefficients[Geo->second->get_name()] = cc;	
			/*if(Geo->second->Geometry()==SHAPE){
				geo::shape *sh = dynamic_cast<geo::shape*>(Geo->second);
				shapes[geo::shape::get_id()] = Geo->second;
			}*/
		}
		if(mbd_sim){
			for(std::map<std::string, geometry*>::iterator Geo2 = mbd_sim->getGeometries()->begin(); Geo2 != mbd_sim->getGeometries()->end(); Geo2++){
				if(Geo2->second->Geometry()==SHAPE){
					double w_cor = 0.98;
					double w_beta=(PI/log(w_cor));
					cmaterialType bm = Geo2->second->getMaterialParameters();
					double ewpy = (pm.youngs * bm.youngs)/(pm.youngs*(1 - bm.poisson*bm.poisson) + bm.youngs*(1 - pm.poisson*pm.poisson));
					cc.kn = (4/3)*sqrt(pradius)*ewpy;
					cc.vn = sqrt((4 * pmass * cc.kn) / (1 + w_beta * w_beta));
					cc.ks = cc.kn * 0.8;
					cc.vs = cc.vn;
					cc.mu = 0.3;
					coefficients[Geo2->second->get_name()] = cc;	
// 					geo::shape *sh = dynamic_cast<geo::shape*>(Geo2->second);
// 										shapes[geo::shape::get_id()] = Geo->second;
				}
			}	
		}

		m_np = dem_sim->getParticles()->Size();
		m_force = new vector3<double>[m_np]; memset(m_force, 0, sizeof(vector3<double>) * m_np);
		m_moment = new vector3<double>[m_np]; memset(m_moment, 0, sizeof(vector3<double>) * m_np);

		if(dem_sim->Device()==GPU)
		{
			checkCudaErrors( cudaMalloc((void**)&d_force, sizeof(double) * (m_np) * 3) );
			checkCudaErrors( cudaMalloc((void**)&d_moment, sizeof(double) * (m_np) * 3) );

			checkCudaErrors( cudaMemcpy(d_force, m_force, sizeof(double) * m_np * 3, cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(d_moment, m_moment, sizeof(double) * m_np * 3, cudaMemcpyHostToDevice) );
		}

	}
	
}

void force::collision(cell_grid *detector)
{

	particles *ps = dem_sim->getParticles();
 	std::map<std::string, geometry*>::iterator Geo;
	if(mbd_sim){
		for(Geo = mbd_sim->getGeometries()->begin(); Geo != mbd_sim->getGeometries()->end(); Geo++){
		//for(std::map<int, geometry*>::iterator Shape = shapes.begin(); Shape != shapes.end(); Shape++){
			if(Geo->second->Geometry() == SHAPE){
				//Geo = mbd_sim->getGeometries()->begin();
				//geo::shape *sh = dynamic_cast<geo::shape*>(Geo->second);
				//geometry* 
				std::string gname = Geo->second->get_name();
				std::map<std::string, contact_coefficient>::iterator coefs = coefficients.find(gname);
				contact_coefficient coe = coefs->second;
				//std::cout << "aafttte3.5" << std::endl;
				Geo->second->cu_hertzian_contact_force(
					coe, 
					ps->cu_IsLineContact(),
					ps->cu_Position(),//detector->getCuMergedData(), 
					ps->cu_Velocity(), 
					ps->cu_AngularVelocity(), 
					d_force, 
					d_moment,
					m_np, 
					detector->cu_getSortedID(),
					detector->cu_getCellStart(),
					detector->cu_getCellEnd());
			}
		}
	}
	//std::cout << "aafttte4" << std::endl;
	cu_calculate_p2p(
		ps->cu_Position(),
		ps->cu_Velocity(),
		ps->cu_Acceleration(),
		ps->cu_AngularVelocity(),
		ps->cu_AngularAcceleration(),
		d_force,
		d_moment,
		detector->cu_getSortedID(),
		detector->cu_getCellStart(),
		detector->cu_getCellEnd(),
		m_np,
		0);
	//std::cout << "aafttte5" << std::endl;
	for(Geo = dem_sim->getGeometries()->begin(); Geo != dem_sim->getGeometries()->end(); Geo++){
		if(Geo->second->GeometryUse() != BOUNDARY)
			continue;
		contact_coefficient coe = coefficients.find(Geo->second->get_name())->second;
		switch(Geo->second->Geometry()){
		case CUBE:
			Geo->second->cu_hertzian_contact_force(
				coe, 
				NULL,
				ps->cu_Position(), 
				ps->cu_Velocity(),
				ps->cu_AngularVelocity(),
				d_force,
				d_moment,
				m_np);
			break;	  
		}
	}
}

void force::collision(
	cell_grid *detector, 
	vector4<double>* pos, 
	vector4<double>* vel, 
	vector4<double>* acc, 
	vector4<double>* omega, 
	vector4<double>* alpha)
{

}

void force::cu_collision(
	cell_grid* detector, 
	double* pos, 
	double* vel, 
	double* acc, 
	double* omega, 
	double* alpha,
	unsigned int cRun)
{

}