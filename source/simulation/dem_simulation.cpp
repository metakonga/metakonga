#include "dem_simulation.h"
#include "velocity_verlet.h"
#include "neighborhood_cell.h"

dem_simulation::dem_simulation()
	: simulation()
	, dtor(NULL)
	, itor(NULL)
	, md(NULL)
	, cm(NULL)
	, pos(NULL), dpos(NULL)
	, vel(NULL), dvel(NULL)
	, acc(NULL), dacc(NULL)
	, avel(NULL), davel(NULL)
	, aacc(NULL), daacc(NULL)
	, force(NULL), dforce(NULL)
	, moment(NULL), dmoment(NULL)
	, mass(NULL), dmass(NULL)
	, inertia(NULL), diner(NULL)
{

}

dem_simulation::dem_simulation(dem_model *_md)
	: simulation()
	, dtor(NULL)
	, itor(NULL)
	, md(_md)
	, cm(NULL)
	, pos(NULL), dpos(NULL)
	, vel(NULL), dvel(NULL)
	, acc(NULL), dacc(NULL)
	, avel(NULL), davel(NULL)
	, aacc(NULL), daacc(NULL)
	, force(NULL), dforce(NULL)
	, moment(NULL), dmoment(NULL)
	, mass(NULL), dmass(NULL)
	, inertia(NULL), diner(NULL)
{

}

dem_simulation::~dem_simulation()
{
	clearMemory();
}

void dem_simulation::applyMassForce()
{
	if (simulation::isCpu())
	{
		for (unsigned int i = 0; i < np; i++)
		{
			dforce[i * 3 + 0] = mass[i] * model::gravity.x;
			dforce[i * 3 + 1] = mass[i] * model::gravity.y;
			dforce[i * 3 + 2] = mass[i] * model::gravity.z;
			dmoment[i * 3 + 0] = 0.0;
			dmoment[i * 3 + 1] = 0.0;
			dmoment[i * 3 + 2] = 0.0;
		}
	}
	else
	{
		checkCudaErrors(cudaMemset(dforce, 0, sizeof(double) * np * 3));
		checkCudaErrors(cudaMemset(dmoment, 0, sizeof(double) * np * 3));
	}
}

void dem_simulation::clearMemory()
{

	if (dtor) delete[] dtor; dtor = NULL;
	if (itor) delete[] itor; itor = NULL;
	if (mass) delete[] mass; mass = NULL;
	if (inertia) delete[] inertia; inertia = NULL;
	if (pos) delete[] pos; pos = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (acc) delete[] acc; acc = NULL;
	if (avel) delete[] avel; avel = NULL;
	if (aacc) delete[] aacc; aacc = NULL;
	if (force) delete[] force; force = NULL;
	if (moment) delete[] moment; moment = NULL;
	if (simulation::isGpu())
	{
		if (dmass) checkCudaErrors(cudaFree(dmass)); dmass = NULL;
		if (diner) checkCudaErrors(cudaFree(diner)); diner = NULL;
		if (dpos) checkCudaErrors(cudaFree(dpos)); dpos = NULL;
		if (dvel) checkCudaErrors(cudaFree(dvel)); dvel = NULL;
		if (dacc) checkCudaErrors(cudaFree(dacc)); dacc = NULL;
		if (davel) checkCudaErrors(cudaFree(davel)); davel = NULL;
		if (daacc) checkCudaErrors(cudaFree(daacc)); daacc = NULL;
		if (dforce) checkCudaErrors(cudaFree(dforce)); dforce = NULL;
		if (dmoment) checkCudaErrors(cudaFree(dmoment)); dmoment = NULL;
	}	
}

void dem_simulation::allocationMemory(unsigned int _np)
{
	clearMemory();
	np = _np;
	mass = new double[np];
	inertia = new double[np];
	pos = new double[np * 4];
	vel = new double[np * 3];
	acc = new double[np * 3];
	avel = new double[np * 3];
	aacc = new double[np * 3];
	force = new double[np * 3];
	moment = new double[np * 3];
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMalloc((void**)&dmass, sizeof(double) * np));
		checkCudaErrors(cudaMalloc((void**)&diner, sizeof(double) * np));
		checkCudaErrors(cudaMalloc((void**)&dpos, sizeof(double) * np * 4));
		checkCudaErrors(cudaMalloc((void**)&dvel, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dacc, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&davel, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&daacc, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dforce, sizeof(double) * np * 3));
		checkCudaErrors(cudaMalloc((void**)&dmoment, sizeof(double) * np * 3));
	}
}


bool dem_simulation::initialize(contactManager* _cm)
{
	cm = _cm;
	particleManager* pm = md->ParticleManager();
	allocationMemory(pm->Np());

	memcpy(pos, pm->Position(), sizeof(double) * np * 4);
	memset(vel, 0, sizeof(double) * np * 3);
	memset(acc, 0, sizeof(double) * np * 3);
	memset(avel, 0, sizeof(double) * np * 3);
	memset(aacc, 0, sizeof(double) * np * 3);
	memset(force, 0, sizeof(double) * np * 3);
	memset(moment, 0, sizeof(double) * np * 3);
	double maxRadius = 0;
	//VEC3D minWorld;
	for (unsigned int i = 0; i < np; i++)
	{
		double r = pos[i * 4 + 3];
		mass[i] = pm->Object()->Density() * (4.0 / 3.0) * M_PI * pow(r, 3.0);
		inertia[i] = (2.0 / 5.0) * mass[i] * pow(r, 2.0);
		acc[i * 3 + 0] = model::gravity.x;
		acc[i * 3 + 1] = model::gravity.y;
		acc[i * 3 + 2] = model::gravity.z;
		if (r > maxRadius)
			maxRadius = r;
	}

	switch (md->SortType())
	{
	case grid_base::NEIGHBORHOOD: dtor = new neighborhood_cell; break;
	}
	if (dtor)
	{
		dtor->setWorldOrigin(VEC3D(-1.0, -1.0, -1.0));
		dtor->setGridSize(VEC3UI(128, 128, 128));
		dtor->setCellSize(maxRadius * 2.0);
		dtor->initialize(np);
	}	
	switch (md->IntegrationType())
	{
	case dem_integrator::VELOCITY_VERLET: itor = new velocity_verlet; break;
	}

	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(dpos, pos, sizeof(double) * np * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dvel, vel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dacc, acc, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(davel, avel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(daacc, aacc, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dforce, force, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmoment, moment, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmass, mass, sizeof(double) * np, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(diner, inertia, sizeof(double) * np, cudaMemcpyHostToDevice));
		if (cm)
		{
			foreach(contact* c, cm->Contacts())
			{
				c->cudaMemoryAlloc();
			}
		}		
		device_parameters dp;
		dp.np = np;
		dp.nsphere = 0;
		dp.ncell = dtor->nCell();
		dp.grid_size.x = grid_base::gs.x;
		dp.grid_size.y = grid_base::gs.y;
		dp.grid_size.z = grid_base::gs.z;
		dp.dt = simulation::dt;
		dp.half2dt = 0.5 * dp.dt * dp.dt;
		dp.cell_size = grid_base::cs;
		dp.cohesion = 0.0;
		dp.gravity.x = model::gravity.x;
		dp.gravity.y = model::gravity.y;
		dp.gravity.z = model::gravity.z;
		dp.world_origin.x = grid_base::wo.x;
		dp.world_origin.y = grid_base::wo.y;
		dp.world_origin.z = grid_base::wo.z;
		setSymbolicParameter(&dp);
	}
	else
	{
		dpos = pos;
		dvel = vel;
		dacc = acc;
		davel = avel;
		daacc = aacc;
		dforce = force;
		dmoment = moment;
		dmass = mass;
		diner = inertia;
	}
		
	return true;
}

bool dem_simulation::oneStepAnalysis()
{
	if (itor->integrationType() == dem_integrator::VELOCITY_VERLET)
		itor->updatePosition(dpos, dvel, dacc, np);
	dtor->detection(dpos, np);
	//applyMassForce();
	cm->runCollision(
		dpos, dvel, davel, 
		dmass, dforce, dmoment, 
		dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(), np);
	if (itor->integrationType() != dem_integrator::VELOCITY_VERLET)
		itor->updatePosition(dpos, dvel, dacc, np);
	itor->updateVelocity(dvel, dacc, davel, daacc, dforce, dmoment, dmass, diner, np);
	return true;
}

QString dem_simulation::saveResult(double *vp, double* vv, double ct, unsigned int pt)
{
	char pname[256] = { 0, };
	QString fname = model::path;// +model::name;
	QString part_name;
	part_name.sprintf("part%04d", pt);
	fname.sprintf("%s/part%04d.bin", fname.toUtf8().data(), pt);
	QFile qf(fname);
	qf.open(QIODevice::WriteOnly);
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(vp, dpos, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vv, dvel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
	}
	else
	{
		memcpy(vp, pos, sizeof(double) * 4 * np);
		memcpy(vv, vel, sizeof(double) * 3 * np);
	}
	qf.write((char*)&ct, sizeof(double));
	qf.write((char*)&np, sizeof(unsigned int));
	qf.write((char*)vp, sizeof(VEC3D) * np);
	qf.write((char*)vv, sizeof(VEC3D) * np);
// 	qf.write((char*)pr, sizeof(double) * np);
// 	qf.write((char*)fs, sizeof(bool) * np);
	qf.close();
	return part_name;
}

// #include "dem_simulation.h"
// #include "collision.h"
// #include "polygonObject.h"
// #include "object.h"
// #include <iomanip>
// #include "mphysics_cuda_dec.cuh"
// #include <QFile>
// #include <QDebug>
// #include <QTime>
// #include <QTextStream>
// 
// dem_simulation::dem_simulation()
// 	:simulation()
// 	, itor(NULL), gb(NULL), paras(NULL)
// 	, d_pos(NULL), d_vel(NULL), d_acc(NULL)
// 	, d_omega(NULL), d_alpha(NULL), d_fr(NULL)
// 	, d_mm(NULL), d_ms(NULL), d_iner(NULL)
// 	, d_riv(NULL)
// {
// 
// }
// 
// dem_simulation::dem_simulation(modeler *_md)
// 	:simulation()
// 	, itor(NULL), gb(NULL), paras(NULL)
// 	, d_pos(NULL), d_vel(NULL), d_acc(NULL)
// 	, d_omega(NULL), d_alpha(NULL), d_fr(NULL)
// 	, d_mm(NULL), d_ms(NULL), d_iner(NULL)
// 	, d_riv(NULL)
// {
// 	
// }
// 
// dem_simulation::~dem_simulation()
// {
// 	clear();
// }
// 
// void dem_simulation::clear()
// {
// 	if (itor) delete itor; itor = NULL;
// 	if (gb) delete gb; gb = NULL;
// 	if (paras) delete paras; paras = NULL;
// 
// 	if (d_pos) checkCudaErrors(cudaFree(d_pos)); d_pos = NULL;
// 	if (d_vel) checkCudaErrors(cudaFree(d_vel)); d_vel = NULL;
// 	if (d_acc) checkCudaErrors(cudaFree(d_acc)); d_acc = NULL;
// 	if (d_omega) checkCudaErrors(cudaFree(d_omega)); d_omega = NULL;
// 	if (d_alpha) checkCudaErrors(cudaFree(d_alpha)); d_alpha = NULL;
// 	if (d_fr) checkCudaErrors(cudaFree(d_fr)); d_fr = NULL;
// 	if (d_mm) checkCudaErrors(cudaFree(d_mm)); d_mm = NULL;
// 	if (d_ms) checkCudaErrors(cudaFree(d_ms)); d_ms = NULL;
// 	//if (d_rad) checkCudaErrors(cudaFree(d_rad));
// 	if (d_iner) checkCudaErrors(cudaFree(d_iner)); d_iner = NULL;
// 	if (d_riv) checkCudaErrors(cudaFree(d_riv)); d_riv = NULL;
// }
// 
// bool dem_simulation::initialize(bool isCpu)
// {
// 	clear();
// 	_isCpu = isCpu;
// 	_isWait = false;
// 	_isWaiting = false;
// 	_abort = false;
// 	_interrupt = false;
// 	//pBar = new QProgressBar;
// 	//durationTime = new QLineEdit;
// 	//durationTime->setFrame(false);
// 	nstep = static_cast<unsigned int>((et / dt) + 1);
// 	//QProgressBar *pBar;
// 	gb = new neighborhood_cell("detector", md);
// 	gb->setWorldOrigin(VEC3D(-1.0, -1.0, -1.0));
// 	gb->setGridSize(VEC3UI(128, 128, 128));
// 	gb->setCellSize(md->particleSystem()->maxRadius() * 2.0);
// 
// 	qDebug() << "- Allocation of contact detection module ------------------ DONE";
// 	itor = new velocity_verlet(md);
// 	qDebug() << "- Allocation of integration module ------------------------ DONE";
// 
// 	unsigned int s_np = 0;
// 	if (md->numPoly()){
// 		s_np = md->numPolygonSphere();
// 	}
// 	np = md->numParticle();
// 	m_pos = new VEC4D[np];
// 	m_vel = new VEC3D[np];
// 	m_force = new VEC3D[np];
// 	if (isCpu){
// 		gb->allocMemory(np);
// 		memcpy(m_pos, md->particleSystem()->position(), sizeof(double) * 4 * np);
// 		memcpy(m_vel, md->particleSystem()->velocity(), sizeof(double) * 3 * np);
// 	}
// 	else{
// 		gb->cuAllocMemory(np);
// 		//md->particleSystem()->cuAllocMemory();
// 		cudaAllocMemory(np);
// 		foreach(object* value, md->objects())
// 		{
// 			if (value->rolltype() != ROLL_PARTICLE)
// 			{
// 				value->cuAllocData(md->numParticle());
// 			}
// 		}
// 		foreach(collision* value, md->collisions())
// 		{
// 			value->allocDeviceMemory();
// 		}
// 		//device_parameters paras;
// 		paras = new device_parameters;
// 		if (md->particleSystem()->generationMethod()){
// 			paras->np = md->particleSystem()->numParticlePerStack();
// 		}
// 		else{
// 			paras->np = md->numParticle();
// 		}
// 		//paras->np = ->particleSystem()->numParticlePerStack();// md->numParticle();
// 		paras->nsphere = s_np;
// 		paras->dt = dt;
// 		paras->cohesion = 0.0;// 1.0E+6;
// 		paras->half2dt = 0.5 * dt * dt;
// 		paras->gravity = make_double3(md->gravity().x, md->gravity().y, md->gravity().z);
// 		paras->cell_size = grid_base::cs;
// 		paras->ncell = gb->nCell();
// 		paras->grid_size = make_uint3(grid_base::gs.x, grid_base::gs.y, grid_base::gs.z);
// 		paras->world_origin = make_double3(grid_base::wo.x, grid_base::wo.y, grid_base::wo.z);
// 		setSymbolicParameter(paras);
// 	}
// 	//gb->reorderElements(isCpu);
// 	foreach(collision* value, md->collisions())
// 	{
// 		if (value->getCollisionPairType() == PARTICLES_POLYGONOBJECT){
// 			value->setGridBase(gb);
// 		}
// 	}
// 
// 	//md->particleSystem()->velocity()[0].x = 0.1f;
// 	
// 	return true;
// }
// 
// bool dem_simulation::saveResult(double ct, unsigned int p)
// {
// 	char partName[256] = { 0, };
// 	//double radius = 0.0;
// 	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", md->modelPath().toStdString().c_str(), p);
// 	//std::fstream of;
// 	QFile of(partName);
// 	//unsigned int np = md->numParticle();
// 	//of.open(partName, std::ios::out, std::ios::binary);
// 	of.open(QIODevice::WriteOnly);
// 	of.write((char*)&np, sizeof(unsigned int));
// 	of.write((char*)&ct, sizeof(double));
// 	of.write((char*)m_pos, sizeof(VEC4D) * np);
// 	of.write((char*)m_vel, sizeof(VEC3D) * np);
// 	of.write((char*)m_force, sizeof(VEC3D) * np);
// 	of.close();
// 	return true;
// }
// 
// bool dem_simulation::savePartResult(double ct, unsigned int p)
// {
// 	char partName[256] = { 0, };
// 	//double radius = 0.0;
// 	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", md->modelPath().toStdString().c_str(), p);
// 	//std::fstream of;
// 	QFile of(partName);
// //	unsigned int np = md->numParticle();
// 	//unsigned int snp = md->particleSystem()->numStackParticle();
// 	if (!_isCpu)
// 	{
// 		checkCudaErrors(cudaMemcpy(m_pos, d_pos, sizeof(double)*np * 4, cudaMemcpyDeviceToHost));
// 		checkCudaErrors(cudaMemcpy(m_vel, d_vel, sizeof(double)*np * 3, cudaMemcpyDeviceToHost));
// 		checkCudaErrors(cudaMemcpy(m_force, d_acc, sizeof(double)*np * 3, cudaMemcpyDeviceToHost));
// 	}	
// 	of.open(QIODevice::WriteOnly);
// 	of.write((char*)&np, sizeof(unsigned int));
// 	//of.write((char*)&snp, sizeof(unsigned int));
// 	of.write((char*)&ct, sizeof(double));
// 	of.write((char*)m_pos, sizeof(VEC4D) * np);
// 	of.write((char*)m_vel, sizeof(VEC3D) * np);
// 	of.write((char*)m_force, sizeof(VEC3D) * np);
// 	of.close();
// 	return true;
// }
// 
// void dem_simulation::cudaUpdatePosition()
// {
// 	itor->cuUpdatePosition(d_pos, d_vel, d_acc, np);
// }
// 
// void dem_simulation::cudaDetection()
// {
// 	gb->cuDetection(d_pos);
// }
// 
// void dem_simulation::cudaUpdateVelocity()
// {
// 	itor->cuUpdateVelocity(d_vel, d_acc, d_omega, d_alpha, d_fr, d_mm, d_ms, d_iner, np);
// }
// 
// void dem_simulation::collision_dem(double dt)
// {
// //	md->particleSystem()->particleCollision(dt);
// 
// // 	std::map<std::string, collision*>::iterator c;
// // 	for (c = md->collision_map().begin(); c != md->collision_map().end(); c++){
// // 		c->second->collid(dt);
// // 	}
// 	foreach(collision* value, md->collisions())
// 	{
// 		value->collid(dt);
// 	} 
// }
// 
// void dem_simulation::cuCollision_dem()
// {
// 	//md->particleSystem()->cuParticleCollision();
// 	foreach(collision* value, md->collisions())
// 	{
// 		value->cuCollid(d_pos, d_vel, d_omega, d_ms, d_fr, d_mm, np);
// 	}
// 
// }
// 
// void dem_simulation::cudaAllocMemory(unsigned int np)
// {
// 	checkCudaErrors(cudaMalloc((void**)&d_pos, sizeof(double)*np * 4));
// 	checkCudaErrors(cudaMemcpy(d_pos, md->particleSystem()->position(), sizeof(double) * np * 4, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_vel, sizeof(double)*np * 3));
// 	//vel[0].z = 0.1f;
// 	checkCudaErrors(cudaMemcpy(d_vel, md->particleSystem()->velocity(), sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_acc, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_acc, md->particleSystem()->acceleration(), sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_omega, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemset(d_omega, 0, sizeof(double) * np * 3));
// //	checkCudaErrors(cudaMemcpy(d_omega, omega, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_alpha, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemset(d_alpha, 0, sizeof(double) * np * 3));
// 	//checkCudaErrors(cudaMemcpy(d_alpha, alpha, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_fr, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemcpy(d_fr, md->particleSystem()->force() , sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_mm, sizeof(double)*np * 3));
// 	checkCudaErrors(cudaMemset(d_mm, 0, sizeof(double) * np * 3));
// 	//checkCudaErrors(cudaMemcpy(d_mm, mm, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_ms, sizeof(double)*np));
// 	checkCudaErrors(cudaMemcpy(d_ms, md->particleSystem()->mass(), sizeof(double) * np, cudaMemcpyHostToDevice));
// 	//checkCudaErrors(cudaMalloc((void**)&d_rad, sizeof(double)*np));
// 	//checkCudaErrors(cudaMemcpy(d_rad, rad, sizeof(double) * np, cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&d_iner, sizeof(double)*np));
// 	checkCudaErrors(cudaMemcpy(d_iner, md->particleSystem()->inertia(), sizeof(double) * np, cudaMemcpyHostToDevice));
// }
// 
// bool dem_simulation::cpuRun()
// {
// 	unsigned int part = 0;
// 	unsigned int cStep = 0;
// 	unsigned int eachStep = 0;
// 	
// 	double ct = dt * cStep;
// 	qDebug() << "-------------------------------------------------------------" << endl
// 			 << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |" << endl
// 			 << "-------------------------------------------------------------";
// 	QTextStream::AlignRight;
// 	//QTextStream::setRealNumberPrecision(6);
// 	QTextStream os(stdout);
// 	os.setRealNumberPrecision(6);
// 	if (saveResult(ct, part)){
// 		os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << "0"  << qSetFieldWidth(0) << " |" << endl;
// 		//std::cout << "| " << std::setw(9) << part << std::setw(12) << ct << std::setw(10) << eachStep << std::setw(11) << cStep << std::setw(15) << 0 << std::endl;
// 	}
// 	QTime tme;
// 	tme.start();
// 	cStep++;	
// // 	std::fstream fs;
// // 	fs.open("C:/C++/presult.txt", std::ios::out);
// 	//md->particleSystem()->velocity()[0].x = 1.0f; //initial particles velocity setting 
// 	while (cStep < nstep)
// 	{
// 		std::cout << cStep << std::endl;
// 		if (cStep == 31617)
// 			bool p = true;
// 		if (!(cStep % 2000))
// 			md->particleSystem()->appendCluster();
// 
// 		if (_abort){
// 			_interrupt = true;
// 			return false;
// 		}
// 		if (_isWait){
// 			_isWaiting = true;
// 			continue;
// 		}
// 		//mutex.lock();
// 		ct = dt * cStep;
// 		
// 		md->particleSystem()->clusterUpdatePosition(dt);
// 		//itor->updatePosition(dt);
// // 		for (unsigned int i = 0; i < 8; i++)
// // 			ppf << ct << " " << md->particleSystem()->position()[i].x << " " << md->particleSystem()->position()[i].y << " " << md->particleSystem()->position()[i].z;
// 		gb->detection();
// 		collision_dem(dt);
// // 		for (unsigned int i = 0; i < 8; i++)
// // 			ppf << md->particleSystem()->force()[i].x << " " << md->particleSystem()->force()[i].y << " " << md->particleSystem()->force()[i].z;
// // 		ppf << endl;
// 		md->particleSystem()->clusterUpdateVelocity(dt);
// 		//itor->updateVelocity(dt);
// 		//md->updateObject(dt);
// 		if (!((cStep) % step)){
// 			//mutex.lock();
// 			part++;
// 			//pBar->setValue(part);
// 			emit sendProgress(part);
// 			if (saveResult(ct, part)){
// 				os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << tme.elapsed() * 0.001 << qSetFieldWidth(0) << " |" << endl;
// 			}
// 			eachStep = 0;
// 			
// 		}
// 		cStep++;
// 		eachStep++;
// 	}
// 	emit finished();
// 	return true;
// }
// 
// bool dem_simulation::gpuRun()
// {
// 	unsigned int part = 0;
// 	unsigned int cStep = 0;
// 	unsigned int eachStep = 0;
// 	ct = dt * cStep;
// 	qDebug() << "-------------------------------------------------------------" << endl
// 			 << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |" << endl
// 			 << "-------------------------------------------------------------";
// 	QTextStream::AlignRight;
// 	QTextStream os(stdout);
// 	os.setRealNumberPrecision(6);
// 	if (savePartResult(ct, part)){
// 		os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << "0" << qSetFieldWidth(0) << " |" << endl;
// 	}
// 	QTime tme;
// 	tme.start();
// 	cStep++;
// 	while (cStep < nstep)
// 	{
// 		if (_isWait)
// 			continue;
// 		if (_abort){
// 			emit finished();
// 			return false;
// 		}
// 		ct = dt * cStep;
// // 		if (md->particleSystem()->updateStackParticle(ct)){
// // 			paras->np += md->particleSystem()->numParticlePerStack();// ->numParticle();
// // 			//gb->cuResizeMemory(paras.np);
// // 			setSymbolicParameter(paras);
// // 		}
// 		itor->updatePosition(d_pos, d_vel, d_acc, np);
//  		gb->cuDetection(d_pos);
//  		cuCollision_dem();
// 		itor->updateVelocity(d_vel, d_acc, d_omega, d_alpha, d_fr, d_mm, d_ms, d_iner, np);
// 		if (!((cStep) % step)){
// 			part++;
// 			emit sendProgress(part);
// 			if (savePartResult(ct, part)){
// 				os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << tme.elapsed() * 0.001 << qSetFieldWidth(0) << " |" << endl;
// 			}
// 			eachStep = 0;
// 		}
// 		cStep++;
// 		eachStep++;
// 	}
// 	emit finished();
// 	return true;
// }