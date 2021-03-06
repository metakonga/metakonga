#include "parSIM.h"
#include "cu_dem_dec.cuh"

using namespace parSIM;

dem_force::dem_force(Simulation *Sim)
	: force(Sim)
{
	
}

dem_force::dem_force(Simulation *demSim, Simulation *mbdSim)
	: force(demSim, mbdSim)
{

}

dem_force::~dem_force()
{
// 	if(m_force) delete [] m_force; m_force = NULL;
// 	if(m_moment) delete [] m_moment; m_moment = NULL;
// 	if(d_force) checkCudaErrors( cudaFree(d_force) ); d_force = NULL;
// 	if(d_moment) checkCudaErrors( cudaFree(d_moment) ); d_moment = NULL;
}

void dem_force::initialize()
{
	particles* ps = sim->getParticles();
	double cor = 0.7;
	double pmass = particles::mass;
	double pradius =particles::radius;
	cmaterialType pm = ps->getMaterialParameters();
	double eym = pm.youngs / (2 * (1 - pm.poisson*pm.poisson));
	double er = (pradius * pradius) / (pradius + pradius);
	double em = (pmass * pmass) / (pmass + pmass);
	double beta=(PI/log(cor));

	//	double kn = (16/15) * sqrt(er) * eym * pow((15 * em * 1.0) / (16 * sqrt(er) * eym), 1/5);
	contact_coefficient cc;
	cc.kn = (4/3)*sqrt(er)*eym;
	cc.vn = sqrt((4 * em * cc.kn) / (1 + beta * beta));
	cc.ks = cc.kn * 0.8;
	cc.vs = cc.vn;
	cc.mu = 0.4;
	/*contact_coefficient pc = cc;*/
	coefficients[ps->Name()] = cc;
	for(std::map<std::string, geometry*>::iterator Geo = sim->getGeometries()->begin(); Geo != sim->getGeometries()->end(); Geo++){
		double w_cor = 0.98;
		double w_beta=(PI/log(w_cor));
		cmaterialType bm = Geo->second->getMaterialParameters();
		double ewpy = (pm.youngs * bm.youngs)/(pm.youngs*(1 - bm.poisson*bm.poisson) + bm.youngs*(1 - pm.poisson*pm.poisson));
		cc.kn = (4/3)*sqrt(pradius)*ewpy;
		cc.vn = sqrt((4 * pmass * cc.kn) / (1 + w_beta * w_beta));
		cc.ks = cc.kn * 0.8;
		cc.vs = cc.vn;
		cc.mu = 0.3;
		coefficients[Geo->second->get_name()] = cc;	

		if(Geo->second->Geometry()==SHAPE){
			geo::shape *sh = dynamic_cast<geo::shape*>(Geo->second);
			shapes[geo::shape::get_id()] = Geo->second;
		}
	}	

	m_np = sim->getParticles()->Size();
	m_force = new vector3<double>[m_np]; memset(m_force, 0, sizeof(vector3<double>) * m_np);
	m_moment = new vector3<double>[m_np];

	if(sim->Device()==GPU)
	{
		checkCudaErrors( cudaMalloc((void**)&d_force, sizeof(double) * (m_np/**factor*/) * 3) );
		//checkCudaErrors( cudaMemset(d_force, 0, sizeof(double) * m_np * 3) );
		checkCudaErrors( cudaMalloc((void**)&d_moment, sizeof(double) * (m_np/**factor*/) * 3) );

		checkCudaErrors( cudaMemcpy(d_force, m_force, sizeof(double) * m_np * 3, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_moment, m_moment, sizeof(double) * m_np * 3, cudaMemcpyHostToDevice) );
	}
}

bool dem_force::calForce(
	double ri, 
	double rj, 
	vector3<double>& posi, 
	vector3<double>& posj, 
	vector3<double>& veli,
	vector3<double>& velj, 
	vector3<double>& omegai,
	vector3<double>& omegaj, 
	vector3<double>& force, 
	vector3<double>& moment)
{
	contact_coefficient coe = coefficients.find("particle")->second;
	
	vector3<double> relative_pos(posj.x - posi.x, posj.y - posi.y, posj.z - posi.z);
	double dist = relative_pos.length();
	double collid_dist = (ri + rj) - dist;
	if(collid_dist <= 0)
		return false;
	vector3<double> unit = relative_pos / dist;     
	vector3<double> relative_vel = velj + omegaj.cross(-rj * unit) - (veli + omegai.cross(ri * unit));
	vector3<double> single_force = (-coe.kn * pow(collid_dist, 1.5) + coe.vn * relative_vel.dot(unit)) * unit;
	vector3<double> single_moment;
	vector3<double> e = relative_vel - relative_vel.dot(unit) * unit;
	double mag_e = e.length();
	if(mag_e)
	{
		vector3<double> s_hat = e / mag_e;
		double ds = mag_e * m_dt;
		vector3<double> shear_force = MIN(coe.ks * ds + coe.vs * (relative_vel.dot(s_hat)), coe.mu * single_force.length()) * s_hat;
		single_moment = (ri * unit).cross(shear_force);
	}
	force += single_force;
	moment += single_moment;
	return true;
}

void dem_force::collision(
	cell_grid *detector, 
	vector4<double>* pos, 
	vector4<double>* vel, 
	vector4<double>* acc, 
	vector4<double>* omega, 
	vector4<double>* alpha)
{
	vector3<int> grid_pos;
	vector3<int> neighbour_pos;
	vector3<double> force, moment;
	vector3<double> posi, posj;
	vector3<double> veli, velj, omegai, omegaj;
	unsigned int grid_hash, start_index, end_index;
	double ri, rj, inv_mass, inv_inertia;
	m_dt = Simulation::dt;
	bool shape_contact_state = false;
	for(unsigned int i = 0; i < m_np; i++){
		vector3<double> line_force;
		vector3<double> line_moment;
		ri = pos[i].w;
		shape_contact_state = false;
		posi = vector3<double>(pos[i].x, pos[i].y, pos[i].z);
		veli = vector3<double>(vel[i].x, vel[i].y, vel[i].z);
		omegai = vector3<double>(omega[i].x, omega[i].y, omega[i].z);
		inv_mass = 1 / acc[i].w;
		inv_inertia = 1 / alpha[i].w;
		grid_pos = detector->get_triplet(posi.x, posi.y, posi.z);
		force = acc[i].w * gravity;
		moment.zeros();
		for(int z = -1; z <= 1; z++){
			for(int y = -1; y <= 1; y++){
				for(int x = -1; x <= 1; x++){
					neighbour_pos = vector3<int>(grid_pos.x + x, grid_pos.y + y, grid_pos.z + z);
					grid_hash = detector->calcGridHash(neighbour_pos);
					start_index = detector->getCellStart(grid_hash);
					if(start_index != 0xffffffff){
						end_index = detector->getCellEnd(grid_hash);
						for(size_t j = start_index; j < end_index; j++){
							size_t k = detector->getSortedIndex(j);
							if(i == k)
								continue;
							rj = pos[k].w;							
							posj = vector3<double>(pos[k].x, pos[k].y, pos[k].z);
							if(rj > 0){								
								velj = vector3<double>(vel[k].x, vel[k].y, vel[k].z);
								omegaj = vector3<double>(omega[k].x, omega[k].y, omega[k].z);
								calForce(ri, rj, posi, posj, veli, velj, omegai, omegaj, force, moment);
							}
							else if(rj < 0 && !shape_contact_state){								
// 								std::map<int, geo::shape*>::iterator sh = shapes.find(int(rj));
// 								contact_coefficient coe = coefficients.find(sh->second)->second;
// 								shape_contact_state = sh->second->hertzian_contact_force(k - m_np, ri, coe, posi, veli, omegai, force, moment, line_force, line_moment);
							}

						}
					}
				}
			}
		}

		if(!shape_contact_state){
// 			for(std::map<int, geo::shape*>::iterator sh = shapes.begin(); sh != shapes.end(); sh++){
// 				if(sh->second->isLineContact){
// 					sh->second->body_force += sh->second->line_contact_force;
// 					sh->second->isLineContact = false;
// 					sh->second->line_contact_force = 0.0;
// 				}
// 			}
			force += line_force;
			moment += line_moment;
		}

		for(std::map<std::string, geometry*>::iterator Geo = sim->getGeometries()->begin(); Geo != sim->getGeometries()->end(); Geo++){
			contact_coefficient coe = coefficients.find(Geo->second->get_name())->second;
			switch(Geo->second->Geometry()){
			case CUBE:{
				geo::cube *Cube = dynamic_cast<geo::cube*>(Geo->second);	
				Cube->hertzian_contact_force(ri, m_dt, coe, posi, veli, omegai, force, moment);
					  } break;

			default: break;
			}
		}
		m_force[i] = force;
		m_moment[i] = moment;
	}
}

void dem_force::cu_collision(
	cell_grid* detector,
	bool* isLineContact,
	double* pos, 
	double* vel, 
	double* acc, 
	double* omega, 
	double* alpha,
	unsigned int cRun)
{
	for(std::map<std::string, geometry*>::iterator Geo = sim->getGeometries()->begin(); Geo != sim->getGeometries()->end(); Geo++){
		contact_coefficient coe = coefficients.find(Geo->second->get_name())->second;
		if(Geo->second->Geometry() == SHAPE){
			geo::shape *sh = dynamic_cast<geo::shape*>(Geo->second);
			Geo->second->cu_hertzian_contact_force(coe, isLineContact, pos, vel, omega, d_force, d_moment, m_np, detector->cu_getSortedID(), detector->cu_getCellStart(), detector->cu_getCellEnd());
		}
	}
	cu_calculate_p2p(
		pos, 
		vel, 
		acc, 
		omega, 
		alpha, 
		d_force, 
		d_moment, 
		detector->cu_getSortedID(), 
		detector->cu_getCellStart(), 
		detector->cu_getCellEnd(), m_np, cRun);

	for(std::map<std::string, geometry*>::iterator Geo = sim->getGeometries()->begin(); Geo != sim->getGeometries()->end(); Geo++){
		contact_coefficient coe = coefficients.find(Geo->second->get_name())->second;
		switch(Geo->second->Geometry()){
		case CUBE:{
			//geo::cube *Cube = dynamic_cast<geo::cube*>(Geo->second);	
			Geo->second->cu_hertzian_contact_force(coe, isLineContact, pos, vel, omega, d_force, d_moment, m_np);
				  } break;
// 		case SHAPE:{
// 			//geo::shape *Shape = dynamic_cast<geo::shape*>(Geo->second);
// 			Geo->second->cu_hertzian_contact_force(coe, pos, vel, omega, d_force, d_moment, m_np, detector->cu_getSortedID(), detector->cu_getCellStart(), detector->cu_getCellEnd());
// 				   }break;
		default: break;
		}
	}

}