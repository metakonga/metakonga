#include "particle_system.h"
#include "modeler.h"
#include "object.h"
#include "collision.h"
#include "collision_particles_particles.h"
#include "grid_base.h"
#include "mphysics_cuda_dec.cuh"

particle_system::particle_system(QString& _name, modeler* _md)
	: nm(_name)
	, md(_md)
	, max_r(0)
	, isc(0)
	, _isMemoryAlloc(false)
	, c_p2p(NULL)
	, ot(PARTICLES)
	, coh(0)
{
	md->setParticleSystem(this);
}

particle_system::~particle_system()
{
	clear();
}

void particle_system::clear()
{
	if (pos) delete[] pos; pos = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (acc) delete[] acc; acc = NULL;
	if (omega) delete[] omega; omega = NULL;
	if (alpha) delete[] alpha; alpha = NULL;
	if (ms) delete[] ms; ms = NULL;
	if (iner) delete[] iner; iner = NULL;
	if (fr) delete[] fr; fr = NULL;
	//if (rad) delete[] rad; rad = NULL;
	if (mm) delete[] mm; mm = NULL;
	if (ms) delete[] ms; ms = NULL;
	if (pair_riv) delete[] pair_riv; pair_riv = NULL;
	if (riv) delete[] riv; riv = NULL;

	if (c_p2p) delete c_p2p; c_p2p = NULL;

	if (d_pos) checkCudaErrors(cudaFree(d_pos));
	if (d_vel) checkCudaErrors(cudaFree(d_vel));
	if (d_acc) checkCudaErrors(cudaFree(d_acc));
	if (d_omega) checkCudaErrors(cudaFree(d_omega));
	if (d_alpha) checkCudaErrors(cudaFree(d_alpha));
	if (d_fr) checkCudaErrors(cudaFree(d_fr));
	if (d_mm) checkCudaErrors(cudaFree(d_mm));
	if (d_ms) checkCudaErrors(cudaFree(d_ms));
	//if (d_rad) checkCudaErrors(cudaFree(d_rad));
	if (d_iner) checkCudaErrors(cudaFree(d_iner));
	if (d_riv) checkCudaErrors(cudaFree(d_riv));
}

void particle_system::allocMemory(unsigned int _np)
{
	pos = new VEC4F[_np];
	vel = new VEC3F[_np];
	//vel[0].x = 1.0f;
	acc = new VEC3F[_np];
	omega = new VEC3F[_np];
	alpha = new VEC3F[_np];
	fr = new VEC3F[_np];
	mm = new VEC3F[_np];
	pair_riv = new unsigned int[_np * 5]; memset(pair_riv, UINT_MAX, sizeof(unsigned int)*_np * 5);
	riv = new float[_np * 5]; memset(riv, 0, sizeof(float)*_np * 5);
	ms = new float[_np]; memset(ms, 0, sizeof(float)*_np);
	iner = new float[_np]; memset(iner, 0, sizeof(float)*_np); 
	_isMemoryAlloc = true;
	//rad = new float[_np]; memset(rad, 0, sizeof(float)*_np);
}

void particle_system::cuAllocMemory()
{
	checkCudaErrors(cudaMalloc((void**)&d_pos, sizeof(float)*np * 4));
	checkCudaErrors(cudaMemcpy(d_pos, pos, sizeof(float) * np * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_vel, sizeof(float)*np * 3));
	//vel[0].z = 0.1f;
	checkCudaErrors(cudaMemcpy(d_vel, vel, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_acc, sizeof(float)*np * 3));
	checkCudaErrors(cudaMemcpy(d_acc, acc, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_omega, sizeof(float)*np * 3));
	checkCudaErrors(cudaMemcpy(d_omega, omega, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_alpha, sizeof(float)*np * 3));
	checkCudaErrors(cudaMemcpy(d_alpha, alpha, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_fr, sizeof(float)*np * 3));
	checkCudaErrors(cudaMemcpy(d_fr, fr, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_mm, sizeof(float)*np * 3));
	checkCudaErrors(cudaMemcpy(d_mm, mm, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_ms, sizeof(float)*np));
	checkCudaErrors(cudaMemcpy(d_ms, ms, sizeof(float) * np, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMalloc((void**)&d_rad, sizeof(float)*np));
	//checkCudaErrors(cudaMemcpy(d_rad, rad, sizeof(float) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_iner, sizeof(float)*np));
	checkCudaErrors(cudaMemcpy(d_iner, iner, sizeof(float) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_riv, sizeof(float)*np * 6));
	checkCudaErrors(cudaMemset(d_riv, 0, sizeof(float) * np * 6));
}

void particle_system::resizeMemory(unsigned int _np)
{
	VEC4F_PTR tv4 = new VEC4F[np];
	VEC3F_PTR tv3 = new VEC3F[np];
	float* tv = new float[np];
	memcpy(tv4, pos, sizeof(VEC4F) * np); delete[] pos; pos = new VEC4F[np + _np]; memcpy(pos, tv4, sizeof(VEC4F)*np);
	memcpy(tv3, vel, sizeof(VEC3F) * np); delete[] vel; vel = new VEC3F[np + _np]; memcpy(vel, tv3, sizeof(VEC3F)*np);
	memcpy(tv3, acc, sizeof(VEC3F) * np); delete[] acc; acc = new VEC3F[np + _np]; memcpy(acc, tv3, sizeof(VEC3F)*np);
	memcpy(tv3, omega, sizeof(VEC3F) * np); delete[] omega; omega = new VEC3F[np + _np]; memcpy(omega, tv3, sizeof(VEC3F) * np);
	memcpy(tv3, alpha, sizeof(VEC3F) * np); delete[] alpha; alpha = new VEC3F[np + _np]; memcpy(alpha, tv3, sizeof(VEC3F) * np);
	memcpy(tv3, fr, sizeof(VEC3F) * np); delete[] fr; fr = new VEC3F[np + _np]; memcpy(fr, tv3, sizeof(VEC3F) * np);
	memcpy(tv3, mm, sizeof(VEC3F) * np); delete[] mm; mm = new VEC3F[np + _np]; memcpy(mm, tv3, sizeof(VEC3F) * np);
	memcpy(tv, ms, sizeof(float) * np); delete[] ms; ms = new float[np + _np]; memcpy(ms, tv, sizeof(float) * np);
	memcpy(tv, iner, sizeof(float) * np); delete[] iner; iner = new float[np + _np]; memcpy(iner, tv, sizeof(float) * np);
	delete[] tv4;
	delete[] tv3;

	_isMemoryAlloc = true;
}

void particle_system::addParticles(object *obj)
{
	obj->setRoll(ROLL_PARTICLE);
	unsigned int add_np = obj->makeParticles(pos[0].w, isc, true, NULL);
	resizeMemory(add_np);
	obj->makeParticles(pos[0].w, isc, false, pos, np);
	np += add_np;
}

bool particle_system::makeParticles(object *obj, float spacing, float _rad)
{
	isc = spacing;
	obj->setRoll(ROLL_PARTICLE);
	bo = obj->objectName();
	np = obj->makeParticles(_rad, isc, true, NULL);
	allocMemory(np);
	obj->makeParticles(_rad, isc, false, pos);
	rho = obj->density();
	E = obj->youngs();
	pr = obj->poisson();
	sh = obj->shear();
	for (unsigned int i = 0; i < np; i++){
		//rad[i] = _rad;
		//pos[i].w = _rad;
		if (max_r < pos[i].w)
			max_r = pos[i].w;
		ms[i] = rho * 4.0f * (float)M_PI * pow(pos[i].w, 3.0f) / 3.0f;
		iner[i] = 2.0f * ms[i] * pow(pos[i].w, 2.0f) / 5.0f;
		fr[i] = ms[i] * md->gravity();
		acc[i] = md->gravity();
	}
	// 	std::string file_par = md->modelPath() + "/" + md->modelName() + ".par";
	// 	std::fstream io_par;
	// 	io_par.open(file_par, std::ios::out);
	// 	io_par.write((char*)pos, sizeof(VEC3F) * np);
	// 	io_par.write((char*)vel, sizeof(VEC3F) * np);
	// 	io_par.write((char*)rad, sizeof(float) * np);
	// 	io_par.write((char*)ms, sizeof(float) * np);
	// 	io_par.write((char*)iner, sizeof(float) * np);
	// 	io_par.close();
	//	md->makeCollision("collision_p2p", _rest, _sratio, _fric, NULL, NULL);
	return true;
}

void particle_system::setCollision(float _rest, float _fric, float _rfric, float _coh)
{
	c_p2p = new collision_particles_particles(QString("collision_p2p"), md, this, HMCM);
	c_p2p->setContactParameter(_rest, _fric, _rfric);
	rest = _rest;
	fric = _fric;
	rfric = _rfric;
	coh = _coh;
}

bool particle_system::particleCollision(float dt)
{
	c_p2p->collid(dt);
	
	for (unsigned int i = 0; i < np; i++){
		for (unsigned int j = 0; j < cs.size(); j++){
			cs.at(j)->collid_with_particle(i, dt);
		}
	}

	return true;
}

void particle_system::cuParticleCollision(grid_base* gb)
{
	cu_calculate_p2p(d_pos, d_vel, d_acc, d_omega, d_alpha, d_fr, d_mm, d_ms, d_iner, d_riv, E, pr, rest, sh, fric, rfric, coh, gb->cuSortedID(), gb->cuCellStart(), gb->cuCellEnd(), np);
}

void particle_system::saveParticleSystem(QFile& oss)
{
	QString file_par = md->modelPath() + "/" + md->modelName() + ".par";
	QFile io_par(file_par);
	io_par.open(QIODevice::WriteOnly);
	//io_par.open(file_par, std::ios::out, std::ios::binary);
	io_par.write((char*)pos, sizeof(VEC4F) * np);
	io_par.write((char*)vel, sizeof(VEC3F) * np);
	//io_par.write((char*)rad, sizeof(float) * np);
	io_par.write((char*)ms, sizeof(float) * np);
	io_par.write((char*)iner, sizeof(float) * np);
	io_par.close();

	QTextStream ots(&oss);
	ots << "PARTICLES" << endl;
	ots << "np " << np << endl
		<< "path " << file_par << endl
		<< "base " << bo << endl
		<< "density " << rho << endl
		<< "youngs " << E << endl
		<< "poisson " << pr << endl
		//<< "shear " << sh << endl
		<< "restitution " << rest << endl
		<< "shear " << sh << endl
		<< "friction " << fric << endl
		<< "rollingfriction " << rfric << endl
		<< "cohesion " << c_p2p->cohesion() << endl
		<< "init_spacing " << isc << endl;
}

void particle_system::setParticlesFromFile(QString& pfile, QString& _bo, unsigned int _np, float _rho, float _E, float _pr, float _sh)
{
	allocMemory(_np);
//	float *rad = new float[_np];
	np = _np;
	bo = _bo;
	QFile pf(pfile);
//	VEC3F_PTR ttp = new VEC3F[np];
	pf.open(QIODevice::ReadOnly);
//	pf.read((char*)ttp, sizeof(VEC3F) * np);
	pf.read((char*)pos, sizeof(VEC4F) * np);
	pf.read((char*)vel, sizeof(VEC3F) * np);
//	pf.read((char*)rad, sizeof(float) * np);
	pf.read((char*)ms, sizeof(float) * np);
	pf.read((char*)iner, sizeof(float) * np);
	pf.close();

	rho = _rho;
	E = _E;
	pr = _pr;
	sh = _sh;

	for (unsigned int i = 0; i < np; i++){
		//rad[i] = _rad;

		if (max_r < pos[i].w)
			max_r = pos[i].w;
		//ms[i] = rho * 4.0f * (float)M_PI * pow(rad[i], 3.0f) / 3.0f;
		//iner[i] = 2.0f * ms[i] * pow(rad[i], 2.0f) / 5.0f;
		fr[i] = ms[i] * md->gravity();
		acc[i] = md->gravity();
// 		if (i == 359){
// 			vel[i].x = 0.001f;
// 		}
	}
	//pos[0].z = 0.f;
}

void particle_system::setPosition(float* vpos)
{
	memcpy(vpos, pos, sizeof(float) * 4 * np);
}

void particle_system::setVelocity(float* vvel)
{
	memcpy(vel, vvel, sizeof(float) * 3 * np);
}

void particle_system::changeParticlesFromVP(float* _pos)
{
	memcpy(pos, _pos, sizeof(float) * 4 * np);
}