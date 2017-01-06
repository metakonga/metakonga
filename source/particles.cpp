#include "parSIM.h"

#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <string>

using namespace parSIM;

double particles::radius = 0.0;
double particles::mass = 0.0;
double particles::inertia = 0.0;

particles::particles(Simulation *_sim)
	: saveCount(0)
	, precision_size(0)
	, sim(_sim)
	, np(0)
	, added_np(0)
	, d_isLineContact(NULL)
	, pos(NULL), d_pos(NULL)
	, vel(NULL), d_vel(NULL)
	, acc(NULL), d_acc(NULL)
	, omega(NULL), d_omega(NULL)
	, alpha(NULL), d_alpha(NULL)
{

}

particles::~particles()
{
	if(pos)	delete [] pos; pos = NULL;
	if(vel) delete [] vel; vel = NULL;
	if(acc) delete [] acc; acc = NULL;
	if(omega) delete [] omega; omega = NULL;
	if(alpha) delete [] alpha; alpha = NULL;

	if(d_isLineContact) checkCudaErrors( cudaFree(d_isLineContact) ); d_isLineContact = NULL;
	if(d_pos) checkCudaErrors( cudaFree(d_pos) ); d_pos = NULL;
	if(d_vel) checkCudaErrors( cudaFree(d_vel) ); d_vel = NULL;
	if(d_acc) checkCudaErrors( cudaFree(d_acc) ); d_acc = NULL;
	if(d_omega) checkCudaErrors( cudaFree(d_omega) ); d_omega = NULL;
	if(d_alpha) checkCudaErrors( cudaFree(d_alpha) ); d_alpha = NULL;
}

void particles::setMaterials(material_type mtype)
{
	switch (mtype)
	{
	case STEEL:
		mat_type = STEEL;
		material.density = STEEL_DENSITY;
		material.youngs = STEEL_YOUNGS_MODULUS;
		material.poisson = STEEL_POISSON_RATIO;
		break;
	case ACRYLIC:
		
		break;
// 	case POLYETHYLENE:
// 		mat_type = POLYSTYRENE;
// 		material.density = POLYSTYRENE_DENSITY;
// 		material.youngs = POLYSTYRENE_YOUNGS_MODULUS;
// 		material.poisson = POLYSTYRENE_POISSON_RATIO;
// 		break;
	case POLYETHYLENE:
		mat_type = POLYETHYLENE;
		material.density = POLYETHYLENE_DENSITY;
		material.youngs = POLYETHYLENE_YOUNGS_MODULUS;
		material.poisson = POLYETHYLENE_POISSON_RATIO;
		break;
	case MEDIUM_CLAY:
		mat_type = MEDIUM_CLAY;
		material.density = MEDIUM_CLAY_DENSITY;
		material.youngs = MEDIUM_CLAY_YOUNGS_MODULUS;
		material.poisson = MEDIUM_CLAY_POISSON_RATIO;
	default:
		break;
	}
}

// void particles::calculate_num_particle()
// {
// 	if(arrange_shape == "cube"){
// 		dim3np = algebra::vector3<unsigned int>(static_cast<unsigned int>(abs(arrange_size.x / (Radius*2)))
// 		 	                                   ,static_cast<unsigned int>(abs(arrange_size.y / (Radius*2)))
// 											   ,static_cast<unsigned int>(abs(arrange_size.z / (Radius*2))));
// 		if(dim3np.x == 0) dim3np.x = 1;
// 		if(dim3np.y == 0) dim3np.y = 1;
// 		if(dim3np.z == 0) dim3np.z = 1;
// 		np = added_np = dim3np.x * dim3np.y * dim3np.z;
// 	}
// 	std::cout << "The number of particle : " << np << "(ea)" << std::endl;
// 
// 	
// }
// 
// void particles::AllocMemory()
// {
// 	std::cout << "Allocation of particle memory ";
// 	pos = new vector4<double>[np]; 
// 	std::cout << ".";
// 	vel = new vector4<double>[np];
// 	std::cout << ".";
// 	acc = new vector4<double>[np];
// 	std::cout << ".";
// 	omega = new vector4<double>[np]; 
// 	std::cout << ".";
// 	alpha = new vector4<double>[np]; 
// 	std::cout << " done" << std::endl;
// }

void particles::resize_pos(unsigned int tnp)
{
	if(tnp <= np) 
		return;
	added_np = tnp;
	vector4<double> *temp = new vector4<double>[np];
	memcpy(temp, pos, sizeof(vector4<double>)*np);
	delete [] pos; pos = NULL;
	pos = new algebra::vector4<double>[tnp];
	memcpy(pos, temp, sizeof(vector4<double>)*np);
	delete [] temp;
}

vector4<double>* particles::add_pos_v3data(vector3<double>* v3, double w, unsigned int an)
{
	for(unsigned int i = 0; i < an; i++){
		pos[np + i] = vector4<double>(v3[i].x, v3[i].y, v3[i].z, w);
	}
	return &pos[np];
}

void particles::CreateParticlesByCube()
{
	geo::cube *gc = dynamic_cast<geo::cube*>(Geo);
	
	dim3np = algebra::vector3<unsigned int>(static_cast<unsigned int>(abs(gc->cube_size().x / (radius*2)))
										   ,static_cast<unsigned int>(abs(gc->cube_size().y / (radius*2)))
										   ,static_cast<unsigned int>(abs(gc->cube_size().z / (radius*2))));
	if(dim3np.x == 0) dim3np.x = 1;
	if(dim3np.y == 0) dim3np.y = 1;
	if(dim3np.z == 0) dim3np.z = 1;
// 
// 	dim3np.x = 1;
// 	dim3np.y = 2;
// 	dim3np.z = 1;
	//np = added_np = dim3np.x * dim3np.y * dim3np.z;
	bool bcond = true;
	vector3<double> p;
	double spacing = radius * 2.1;
	while(bcond){
		bcond = false;
		p = vector3<double>(gc->Position().x + radius + (dim3np.x - 1) * spacing
			,gc->Position().y + radius + (dim3np.y - 1) * spacing
			,gc->Position().z + radius + (dim3np.z - 1) * spacing);
		if(p.x + radius > gc->cube_size().x && dim3np.x > 1){
			dim3np.x--;
			bcond = true;
		}
		if(p.y + radius > gc->cube_size().y && dim3np.y > 1){
			dim3np.y--;
			bcond = true;
		}
		if(p.z + radius > gc->cube_size().z && dim3np.z > 1){
			dim3np.z--;
			bcond = true;
		}
		if(!bcond)
			break;
	}

	np = added_np = dim3np.x * dim3np.y * dim3np.z;
	//np = 1;
	std::cout << "The number of particle : " << np << "(ea)" << std::endl;

	std::cout << "Allocation of particle memory ";
	pos = new vector4<double>[np]; 
	std::cout << ".";
	vel = new vector4<double>[np];
	std::cout << ".";
	acc = new vector4<double>[np];
	std::cout << ".";
	omega = new vector4<double>[np]; 
	std::cout << ".";
	alpha = new vector4<double>[np]; 
	std::cout << " done" << std::endl;

	std::cout << "Creating and Arrangement of particle " << std::endl;
	unsigned int p_id = 0;
	
	std::cout << "    Mode : cube" << std::endl;
	srand(1973);
	double jitter = radius * 0.001;
	for(unsigned int z = 0; z < dim3np.z; z++){
		for(unsigned int y = 0; y < dim3np.y; y++){
			for(unsigned int x = 0; x < dim3np.x; x++){
				//double dd = frand();
				pos[p_id].x = (gc->Position().x + radius + x*spacing) + frand()*jitter;
				pos[p_id].y = (gc->Position().y + radius + y*spacing) + frand()*jitter;
				pos[p_id].z = (gc->Position().z + radius + z*spacing) + frand()*jitter;
				pos[p_id].w = radius;
				vel[p_id].w = -1.0;
				//vel[p_id].x = 0.1;]
				acc[p_id].y = 0.0;//-9.80665;
				acc[p_id].w = mass;
				alpha[p_id].w = inertia;
				p_id++;
			}
		}
	}

//  	pos[0].x += 2.01;
// // 	pos[0].y = 0.030;
//  	pos[0].z += 1.51;
// 
// 	pos[1].x += 2.015;
// 	// 	pos[0].y = 0.030;
// 	pos[1].z += 1.51;

	std::cout << "done" << std::endl;
	
}

void particles::CreateParticles(geometry_type g_type)
{
	switch(g_type){
	case CUBE: CreateParticlesByCube(); break;
	}
}

// void particles::arrangement()
// {
// 	std::cout << "Creating and Arrangement of particle " << std::endl;
// 	unsigned int p_id = 0;
// 	double spacing = radius * 2.1;
// 
// 	if(arrange_shape == "cube"){
// 		std::cout << "    Mode : cube" << std::endl;
// 		srand(1973);
// 		double jitter = radius * 0.001;
// 		for(unsigned int z = 0; z < dim3np.z; z++){
// 			for(unsigned int y = 0; y < dim3np.y; y++){
// 				for(unsigned int x = 0; x < dim3np.x; x++){
// 					//double dd = frand();
// 					pos[p_id].x = (arrange_position.x + radius + x*spacing) + frand()*jitter;
// 					pos[p_id].y = (arrange_position.y + radius + y*spacing) + frand()*jitter;
// 					pos[p_id].z = (arrange_position.z + radius + z*spacing) + frand()*jitter;
// 					pos[p_id].w = radius;
// 					vel[p_id].w = -1.0;
// 					//vel[p_id].x = 0.1;]
// 					acc[p_id].y = -9.80665;
// 					acc[p_id].w = mass;
// 					alpha[p_id].w = inertia;
// 					p_id++;
// 				}
// 			}
// 		}
// 		std::cout << "done" << std::endl;
// 	}
// }

bool particles::initialize()
{
	if(radius <= 0){
		std::cout << "ERROR : The radius of particle is zero." << std::endl;
		return false;
	}

	std::map<std::string, geometry*> *geos = sim->getGeometries();
	for(std::map<std::string, geometry*>::iterator g = geos->begin(); g != geos->end(); g++)
		if(g->second->GeometryUse() == PARTICLE)
			Geo = g->second;

	setMaterials(Geo->Material());

	if(mass == 0.0)
		mass = material.density * 4.0 * PI * radius * radius * radius / 3.0;

	if(inertia == 0.0)
		inertia = 2.0 * mass * radius * radius / 5.0;

	CreateParticles(Geo->Geometry());

	if(Simulation::specific_data != ""){
		std::fstream pf;
		pf.open(Simulation::specific_data, std::ios::in | std::ios::binary);
		if(pf.is_open()){
			while(!pf.eof()){
				int type;
				pf.read((char*)&type, sizeof(int));
				switch (type)
				{
				case -1:
					return true;
				case PARTICLE:
					setSpecificDataFromFile(pf);	
					break;
				default:
					break;
				}
			}
		}
		else
		{
			Log::Send(Log::Error, "No exist specific_data. The path is " + Simulation::specific_data);
			return false;
		}
	}
	return true;
}

void particles::define_device_info()
{
	checkCudaErrors( cudaMalloc((void**)&d_isLineContact, sizeof(bool)*added_np) );
	checkCudaErrors( cudaMalloc((void**)&d_pos, sizeof(double)*added_np*4) );
	checkCudaErrors( cudaMalloc((void**)&d_vel, sizeof(double)*added_np*4) );
	checkCudaErrors( cudaMalloc((void**)&d_acc, sizeof(double)*added_np*4) );
	checkCudaErrors( cudaMalloc((void**)&d_omega, sizeof(double)*added_np*4) );
	checkCudaErrors( cudaMalloc((void**)&d_alpha, sizeof(double)*added_np*4) );

	checkCudaErrors( cudaMemset(d_isLineContact, 0, sizeof(bool)*added_np) );
	checkCudaErrors( cudaMemcpy(d_pos, pos, sizeof(double)*added_np*4, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_vel, vel, sizeof(double)*np*4, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_acc, acc, sizeof(double)*np*4, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_omega, omega, sizeof(double)*np*4, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_alpha, alpha, sizeof(double)*np*4, cudaMemcpyHostToDevice) );

}

void particles::setSpecificDataFromFile(std::fstream& pf)
{
	float *tpos = new float[np * 4];
	float *tvel = new float[np * 4];

	pf.read((char*)tpos, sizeof(float)*np*4);
	pf.read((char*)tvel, sizeof(float)*np*4);
	
	for(unsigned int i = 0; i < np; i++){
		pos[i] = vector4<double>(
			static_cast<double>(tpos[i*4+0]),
			static_cast<double>(tpos[i*4+1]),
			static_cast<double>(tpos[i*4+2]),
			static_cast<double>(tpos[i*4+3]));
		vel[i] = vector4<double>(
			static_cast<double>(tvel[i*4+0]),
			static_cast<double>(tvel[i*4+1]),
			static_cast<double>(tvel[i*4+2]),
			static_cast<double>(tvel[i*4+3]));
	}

	delete [] tpos;
	delete [] tvel;
}

double particles::calMaxRadius()
{
	double max_radius = 0;
	for(unsigned int i = 0; i < np; i++){
		if(max_radius < pos[i].w)
			max_radius = pos[i].w;
	}
	return max_radius;
}

// void particles::rearrangement(cell_grid* detector)
//{
// 	detector->detection(pos);
// 	vector3<int> grid_pos;
// 	vector3<int> neighbour_pos;
// 	vector3<double> posi, posj;
// 	unsigned int grid_hash, start_index, end_index;
// 	for(unsigned int i = 0; i < np; i++){
// 		posi = vector3<double>(pos[i].x, pos[i].y, pos[i].z);
// 		grid_pos = detector->get_triplet(posi.x, posi.y, posi.z);
// 		for(int z = -1; z <= 1; z++){
// 			for(int y = -1; y <= 1; y++){
// 				for(int x = -1; x <= 1; x++){
// 					neighbour_pos = vector3<int>(grid_pos.x + x, grid_pos.y + y, grid_pos.z + z);
// 					grid_hash = detector->calcGridHash(neighbour_pos);
// 					start_index = detector->getCellStart(grid_hash);
// 					if(start_index != 0xffffffff){
// 						end_index = detector->getCellEnd(grid_hash);
// 						for(size_t j = start_index; j < end_index; j++){
// 							size_t k = detector->getSortedIndex(j);
// 							posj = vector3<double>(pos[k].x, pos[k].y, pos[k].z);
// 							pos
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
//}