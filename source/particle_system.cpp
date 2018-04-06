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
	, _isMemoryAlloc(false)
	, nStack(0)
	, cStack(0)
	, npPerStack(0)
	, stack_dt(0)
	, last_stack_time(0)
	, pc(NULL)
	, cid(NULL)
	, tGenParticle(DEFAULT_GENERATION_PARTICLE)
	//, c_p2p(NULL)
	, ot(PARTICLES)
	//, coh(0)
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
	//if (pos) delete[] pos; pos = NULL;
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
	if (cid) delete[] cid; cid = NULL;
	//if (c_p2p) delete c_p2p; c_p2p = NULL;

// 	if (d_pos) checkCudaErrors(cudaFree(d_pos));
// 	if (d_vel) checkCudaErrors(cudaFree(d_vel));
// 	if (d_acc) checkCudaErrors(cudaFree(d_acc));
// 	if (d_omega) checkCudaErrors(cudaFree(d_omega));
// 	if (d_alpha) checkCudaErrors(cudaFree(d_alpha));
// 	if (d_fr) checkCudaErrors(cudaFree(d_fr));
// 	if (d_mm) checkCudaErrors(cudaFree(d_mm));
// 	if (d_ms) checkCudaErrors(cudaFree(d_ms));
// 	//if (d_rad) checkCudaErrors(cudaFree(d_rad));
// 	if (d_iner) checkCudaErrors(cudaFree(d_iner));
// 	if (d_riv) checkCudaErrors(cudaFree(d_riv));

	qDeleteAll(pc);
	//if (pc) delete[] pc; pc = NULL;
}

void particle_system::allocMemory(unsigned int _np)
{
	pos = new VEC4D[_np];
	//pos = new VEC4D[_np];
	//r_pos = new VEC4D[_np];
	vel = new VEC3D[_np];
	//vel[0].x = 1.0f;
	acc = new VEC3D[_np];
	omega = new VEC3D[_np];
	alpha = new VEC3D[_np];
	fr = new VEC3D[_np];
	mm = new VEC3D[_np];
	pair_riv = new unsigned int[_np * 5]; memset(pair_riv, UINT_MAX, sizeof(unsigned int)*_np * 5);
	riv = new double[_np * 5]; memset(riv, 0, sizeof(double)*_np * 5);
	ms = new double[_np]; memset(ms, 0, sizeof(double)*_np);
	iner = new double[_np]; memset(iner, 0, sizeof(double)*_np); 
	_isMemoryAlloc = true;
	//rad = new double[_np]; memset(rad, 0, sizeof(double)*_np);
}

void particle_system::cuAllocMemory()
{
// 	checkCudaErrors(cudaMalloc((void**)&d_pos, sizeof(double)*np * 4));
// 	checkCudaErrors(cudaMemcpy(d_pos, pos, sizeof(double) * np * 4, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_vel, sizeof(double)*np * 3));
// 	//vel[0].z = 0.1f;
// 	checkCudaErrors(cudaMemcpy(d_vel, vel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_acc, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_acc, acc, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_omega, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_omega, omega, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_alpha, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_alpha, alpha, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_fr, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_fr, fr, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_mm, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_mm, mm, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_ms, sizeof(double)*np));
// 	checkCudaErrors(cudaMemcpy(d_ms, ms, sizeof(double) * np, cudaMemcpyHostToDevice));
// 	//checkCudaErrors(cudaMalloc((void**)&d_rad, sizeof(double)*np));
// 	//checkCudaErrors(cudaMemcpy(d_rad, rad, sizeof(double) * np, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_iner, sizeof(double)*np));
// 	checkCudaErrors(cudaMemcpy(d_iner, iner, sizeof(double) * np, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_riv, sizeof(double)*np * 6));
// 	checkCudaErrors(cudaMemset(d_riv, 0, sizeof(double) * np * 6));
	//c_p2p->allocDeviceMemory();
}

// void particle_system::resizeCudaMemory(unsigned int _np)
// {
// 
// }

bool particle_system::updateStackParticle(double ct, tSolveDevice tDev)
{
	if (!tGenParticle)
		return false;
	if (cStack == nStack)
		return false;
	float diff_t = ct - last_stack_time;
	if (diff_t >= stack_dt){
		cStack++;
		last_stack_time = ct;
		return true;
	}
	return false;
}

// void particle_system::resizeMemoryForStack(unsigned int _np)
// {
// 	resizeMemory(_np);
// 	total_stack_particle = _np;
// }

void particle_system::resizeMemory(unsigned int _np)
{
	VEC4D_PTR tv4 = new VEC4D[np];
	VEC3D_PTR tv3 = new VEC3D[np];
	double* tv = new double[np];
	memcpy(tv4, pos, sizeof(VEC4D) * np); delete[] pos; pos = new VEC4D[np + _np]; memcpy(pos, tv4, sizeof(VEC4D)*np);
	memcpy(tv3, vel, sizeof(VEC3D) * np); delete[] vel; vel = new VEC3D[np + _np]; memcpy(vel, tv3, sizeof(VEC3D)*np);
	memcpy(tv3, acc, sizeof(VEC3D) * np); delete[] acc; acc = new VEC3D[np + _np]; memcpy(acc, tv3, sizeof(VEC3D)*np);
	memcpy(tv3, omega, sizeof(VEC3D) * np); delete[] omega; omega = new VEC3D[np + _np]; memcpy(omega, tv3, sizeof(VEC3D) * np);
	memcpy(tv3, alpha, sizeof(VEC3D) * np); delete[] alpha; alpha = new VEC3D[np + _np]; memcpy(alpha, tv3, sizeof(VEC3D) * np);
	memcpy(tv3, fr, sizeof(VEC3D) * np); delete[] fr; fr = new VEC3D[np + _np]; memcpy(fr, tv3, sizeof(VEC3D) * np);
	memcpy(tv3, mm, sizeof(VEC3D) * np); delete[] mm; mm = new VEC3D[np + _np]; memcpy(mm, tv3, sizeof(VEC3D) * np);
	memcpy(tv, ms, sizeof(double) * np); delete[] ms; ms = new double[np + _np]; memcpy(ms, tv, sizeof(double) * np);
	memcpy(tv, iner, sizeof(double) * np); delete[] iner; iner = new double[np + _np]; memcpy(iner, tv, sizeof(double) * np);
	delete[] tv4;
	delete[] tv3;

	_isMemoryAlloc = true;
}

void particle_system::addParticles(object *obj, VEC3UI size)
{
	obj->setRoll(ROLL_PARTICLE);
	unsigned int add_np = obj->makeParticles(pos[0].w, size, isc, 0, true, NULL);
	resizeMemory(add_np);
	obj->makeParticles(pos[0].w, size, isc, 0, false, pos, np);
	np += add_np;
}

bool particle_system::makeParticles(object *obj, VEC3UI size, VEC3D spacing, double _rad, unsigned int nstack)
{
	isc = spacing;
	genParticleSize = size;
	obj->setRoll(ROLL_PARTICLE);
	bo = obj->objectName();
	np = obj->particleCount();// size.x * size.y * size.z; //obj->makeParticles(_rad, size, isc, true, NULL);
	np += np * nstack;
	allocMemory(np);
	obj->makeParticles(_rad, size, isc, nstack, false, pos);
	
	rho = obj->density();
	E = obj->youngs();
	pr = obj->poisson();
	sh = obj->shear();
	for (unsigned int i = 0; i < np; i++){
		//rad[i] = _rad;
		//pos[i].w = _rad;
		if (max_r < pos[i].w)
			max_r = pos[i].w;
		ms[i] = rho * 4.0 * M_PI * pow(pos[i].w, 3.0) / 3.0;
		iner[i] = 2.0 * ms[i] * pow(pos[i].w, 2.0) / 5.0;
		fr[i] = ms[i] * md->gravity();
		acc[i] = md->gravity();
	}
	//appendCluster();
// 	pc = new particle_cluster[1];
// 	pc->setIndice(2, 0, 1);
// 	pc->define(pos, ms, iner);
	return true;
}

// void particle_system::setCollision(float _rest, float _fric, float _rfric, float _coh, float _ratio)
// {
// 	tContactModel tcm = HMCM;
// 	if (_ratio)
// 		tcm = DHS;
// 
// 	c_p2p = new collision_particles_particles(QString("collision_p2p"), md, this, tcm);
// 	c_p2p->setContactParameter(
// 		E, E, pr, pr, sh, sh, _rest, _fric, _rfric, _coh, _ratio);
// 	rest = _rest;
// 	fric = _fric;
// 	rfric = _rfric;
// 	coh = _coh;
// 	sratio = _ratio;
// }

// bool particle_system::particleCollision(float dt)
// {
// 	//c_p2p->collid(dt);
// 	
// 	for (unsigned int i = 0; i < np; i++){
// 		for (unsigned int j = 0; j < cs.size(); j++){
// 			cs.at(j)->collid_with_particle(i, dt);
// 		}
// 	}
// 
// 	return true;
// }

// void particle_system::cuParticleCollision()
// {
// 	c_p2p->cuCollid();
// 	/*cu_calculate_p2p(d_pos, d_vel, d_acc, d_omega, d_alpha, d_fr, d_mm, d_ms, d_iner, d_riv, E, pr, rest, sh, fric, rfric, coh, gb->cuSortedID(), gb->cuCellStart(), gb->cuCellEnd(), np);*/
// }

void particle_system::saveParticleSystem(QFile& oss)
{
	QString file_par = md->modelPath() + "/" + md->modelName() + ".par";
	QFile io_par(file_par);
	io_par.open(QIODevice::WriteOnly);
	//io_par.open(file_par, std::ios::out, std::ios::binary);
	io_par.write((char*)pos, sizeof(VEC4D) * np);
	io_par.write((char*)vel, sizeof(VEC3D) * np);
	//io_par.write((char*)rad, sizeof(float) * np);
	io_par.write((char*)ms, sizeof(double) * np);
	io_par.write((char*)iner, sizeof(double) * np);
	io_par.close();
	QTextStream ots(&oss);

	

	ots << "PARTICLES" << endl;
	ots << "np " << np << endl
		<< "path " << file_par << endl
		<< "base " << bo << endl
		<< "density " << rho << endl
		<< "youngs " << E << endl
		<< "poisson " << pr << endl
		<< "shear " << sh << endl;
	if (tGenParticle){
		ots << "STACK " << tGenParticle << endl
			<< "number " << nStack << endl
			<< "interval " << stack_dt << endl
			<< "per " << npPerStack << endl;
	}
	
	if (pc.size())
	{
		ots << "CLUSTER " << endl
			<< "consist " << 2 << endl;
	}
// 		<< "restitution " << rest << endl
// 		<< "shear " << sh << endl
// 		<< "friction " << fric << endl
// 		<< "rollingfriction " << rfric << endl
// 		<< "cohesion " << coh << endl
// 		<< "stiff_ratio " << sratio << endl;
		//<< "init_spacing " << isc << endl;
}

void particle_system::appendCluster()
{
	int _consist = 2;
	unsigned int id = pc.size() * _consist;
	particle_cluster *_pc = new particle_cluster;
	//pos[i+1] = VEC4D(pos[i] + pos)
	//if (_consist == 2)
	_pc->setIndice(id);
// 		else if (_consist == 3)
// 		_pc->setIndice(_consist, id, id + 1, id + 2);
// 	else if (_consist == 4)
// 		_pc->setIndice(_consist, id, id + 1, id + 2, id + 3);

	_pc->define(pos, ms, iner);
	pc.push_back(_pc);
}

void particle_system::setParticleCluster(int _consist)
{
	particle_cluster::setConsistNumber(_consist);
	appendCluster();
// 	for (unsigned int i = 0; i < np; i += _consist)
// 	{
// 	unsigned int id = pc.size() * _consist;
// 	particle_cluster *_pc = new particle_cluster;
// 	//pos[i+1] = VEC4D(pos[i] + pos)
// 	if (_consist == 2)
// 		_pc->setIndice(_consist, id, id + 1);
// 	else if (_consist == 3)
// 		_pc->setIndice(_consist, id, id + 1, id + 2);
// 	else if (_consist == 4)
// 		_pc->setIndice(_consist, id, id + 1, id + 2, id + 3);
// 
// 	_pc->define(pos, ms, iner);
// 	pc.push_back(_pc);
//	}
}

void particle_system::setParticlesFromFile(QString& pfile, QString& _bo, unsigned int _np, double _rho, double _E, double _pr, double _sh)
{
	allocMemory(_np);
//	float *rad = new float[_np];
	np = _np;
	bo = _bo;
	QFile pf(pfile);
//	VEC3F_PTR ttp = new VEC3F[np];
	pf.open(QIODevice::ReadOnly);
//	pf.read((char*)ttp, sizeof(VEC3F) * np);
	pf.read((char*)pos, sizeof(VEC4D) * np);
	pf.read((char*)vel, sizeof(VEC3D) * np);
//	pf.read((char*)rad, sizeof(float) * np);
	pf.read((char*)ms, sizeof(double) * np);
	pf.read((char*)iner, sizeof(double) * np);
	pf.close();
//	pos[1].z = 0.f;
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
		//fr[i] = ms[i] * md->gravity();
		acc[i] = md->gravity();
// 		if (i == 359){
// 			vel[i].x = 0.001f;
// 		}
	}
	//appendCluster();
// 	for (unsigned int i = 0; i < np; i += 2)
// 	{
// 
// 	}
	//float sq = sqrt(0.05f * 0.05f + 0.05f * 0.05f);
// 	pos[0].x = 0;  pos[0].y = 0.01; pos[0].z = 0.0;
// 	pos[1].x = 0.004; pos[1].y = pos[0].y; pos[1].z = 0.0;
// 
// 	pos[2].x = 0;  pos[2].y = 0.025; pos[2].z = 0.0;
// 	pos[3].x = 0.004; pos[3].y = pos[2].y; pos[3].z = 0.0;
// 	pc = new particle_cluster[2];
// 	cid = new unsigned int[np];
// 	pc[0].setIndice(2, 0, 1);
// 	pc[0].define(pos, ms, iner);
// 	cid[0] = cid[1] = 0;
// 	pc[1].setIndice(2, 2, 3);
// 	pc[1].define(pos, ms, iner);
// 	cid[2] = cid[3] = 1;
	//pos[0].z = 0.f;
}

void particle_system::setPosition(double* vpos)
{
	memcpy(pos, vpos, sizeof(double) * 4 * np);
}

void particle_system::setVelocity(double* vvel)
{
	memcpy(vel, vvel, sizeof(double) * 3 * np);
}

void particle_system::changeParticlesFromVP(double* _pos)
{
	memcpy(pos, _pos, sizeof(double) * 4 * np);
}

void particle_system::clusterUpdatePosition(double dt)
{
	foreach(particle_cluster* value, pc)
	{
		value->updatePosition(pos, omega, alpha, dt);
	}
// 	pc[0].updatePosition(pos, omega, alpha, dt);
// 	pc[1].updatePosition(pos, omega, alpha, dt);
	memset(fr, 0, sizeof(VEC3D) * np);
	memset(mm, 0, sizeof(VEC3D) * np);
}

void particle_system::clusterUpdateVelocity(double dt)
{
	foreach(particle_cluster* value, pc)
	{
		value->updateVelocity(vel, omega, fr, mm, dt);
	}
// 	pc[0].updateVelocity(vel, omega, fr, mm, dt);
// 	pc[1].updateVelocity(vel, omega, fr, mm, dt);
}