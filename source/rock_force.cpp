#include "parSIM.h"

using namespace parSIM;

rock_force::rock_force(Simulation* Sim)
	: force(Sim)
{
	
}

rock_force::~rock_force()
{
	
}

void rock_force::initialize(particle* pars)
{
	particles *ps = sim->getParticles();
	mu = ps->RockProperties().friction;
	//kn = new double[ps->Size()];
	//ks = new double[ps->Size()];
	//Fs = new vector3<double>[ps->Size()];
	//memset(Fs, 0, sizeof(vector3<double>) * ps->Size());
	rock_properties prock = sim->getParticles()->RockProperties();
	cement_properties pcement = sim->getParticles()->CementProperties();
	if(sim->getDim() == DIM_2){
		for(unsigned int i = 0; i < ps->Size(); i++){
			pars[i].setKn( 2 * 1 * prock.youngs);
			pars[i].setKs(pars[i].Kn() / prock.stiffness_ratio);
		}
		
	}
	else{
		vector4<double>* pos = ps->Position();
		for(unsigned int i = 0; i < ps->Size(); i++){
// 			kn[i] = 4 * pos[i].w * prock.youngs;
// 			ks[i] = kn[i] / prock.stiffness_ratio;
		}
	}
	m_np = sim->getParticles()->Size();
	m_force = new vector3<double>[m_np]; memset(m_force, 0, sizeof(vector3<double>) * m_np);
	m_moment = new vector3<double>[m_np];
}

bool rock_force::calForce(
	unsigned int i, 
	unsigned int j,
	particle* pars) 
// 	double ir, 
// 	double jr, 
// 	vector3<double>& ip, 
// 	vector3<double>& jp,
// 	vector3<double>& iv,
// 	vector3<double>& jv,
// 	vector3<double>& iw,
// 	vector3<double>& jw,
// 	vector3<double>& f, 
// 	vector3<double>& m)
{
// 	vector3<double> relative_pos = pars[j].Position() - pars[i].Position();
// 	double dist = relative_pos.length();
// 	double collid_dist = (pars[i].Radius() + pars[j].Radius()) - dist;
// 	if(collid_dist <= 0)
// 		return false;
// 	particle::contact_data cdata;
// 	cdata.j = j;
// 	cdata.cdist = collid_dist;
// 	//double Kn = kn[i] * kn[j] / (kn[i] + kn[j]);
// 	//double Ks = ks[i] * ks[j] / (ks[i] + ks[j]);
// 	vector3<double> unit = relative_pos / dist;
// 	vector3<double> x_c = pars[i].Position() + (pars[i].Radius() - 0.5*collid_dist) * unit;
// 	vector3<double> CV = (pars[j].Velocity() + pars[j].Omega().cross(x_c - pars[j].Position())) - (pars[i].Velocity() + pars[i].Omega().cross(x_c - pars[i].Position()));
// 	vector3<double> d_Us = CV - CV.dot(unit) * unit;
// 	double mag_d_Us = d_Us.length();
// 
// 	double _Fn = Kn * collid_dist;
// 	vector3<double> dFs = -Ks * d_Us;
// 	Fs[i] += dFs;
// 	if(Fs[i].length() > mu*_Fn) {
// 		Fs[i] = mu * _Fn * unit;
// 	}
// 	vector3<double> pf = _Fn * unit;// + Fs[i];
// 	m -= (x_c - ip).cross(pf);
// 	f -= pf;
 	return true;
}

void rock_force::collision(cell_grid *detector, particle *pars)
{
	vector3<int> grid_pos;
	vector3<int> neighbour_pos;

	particle::contact_data cdata;
	unsigned int grid_hash, start_index, end_index;
	//double ri, rj, inv_mass, inv_inertia;
	m_dt = sim->getDt();
	//bool shape_contact_state = false;
	bool isContact = false;
	for(unsigned int i = 0; i < m_np; i++){
		isContact = false;
		pars[i].contacts.clear();
		pars[i].Force() += gravity;
		//pars[i].Moment() = 0;
		grid_pos = detector->get_triplet(pars[i].Position());
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
							vector3<double> relative_pos = pars[k].Position() - pars[i].Position();
							double dist = relative_pos.length();
							double collid_dist = (pars[i].Radius() + pars[k].Radius()) - dist;
							if(collid_dist > 0)
							{
								cdata.geo = NULL;
								cdata.j = pars + k;
								cdata.cdist = collid_dist;
								cdata.normal = relative_pos / dist;
								pars[i].AppendContactRelation(cdata);
							}
						}
					}
				}
			}
		}

		for(std::map<std::string, geometry*>::iterator Geo = sim->getGeometries()->begin(); Geo != sim->getGeometries()->end(); Geo++){
			switch(Geo->second->Geometry()){
			case RECTANGLE:{
				geo::rectangle *rec = dynamic_cast<geo::rectangle*>(Geo->second);
				rec->collision(pars[i]);
					  } break;
			default: break;
			}
		}
		pars[i].CalculateRockForce(m_dt);
	}
}

void rock_force::cu_collision(cell_grid *detector, double* pos, double* vel, double* acc, double* omega, double* alpha, unsigned int cRun /* = 0 */)
{

}

