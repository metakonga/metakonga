#include "dem_simulation.h"
#include <QDebug>
#include "velocity_verlet.h"
#include "neighborhood_cell.h"
#include "contact_particles_polygonObjects.h"
//#include "polygonObject.h"

dem_simulation::dem_simulation()
	: simulation()
	, np(0)
	, nPolySphere(0)
	, dtor(NULL)
	, itor(NULL)
	, md(NULL)
	, cm(NULL)
	, itor_type(DEM_INTEGRATOR)
	, pos(NULL), dpos(NULL)
	, vel(NULL), dvel(NULL)
	, acc(NULL), dacc(NULL)
	, avel(NULL), davel(NULL)
	, aacc(NULL), daacc(NULL)
	, force(NULL), dforce(NULL)
	, moment(NULL), dmoment(NULL)
	, mass(NULL), dmass(NULL)
	, inertia(NULL), diner(NULL)
	, pos_f(NULL), dpos_f(NULL)
	, vel_f(NULL), dvel_f(NULL)
	, acc_f(NULL), dacc_f(NULL)
	, avel_f(NULL), davel_f(NULL)
	, aacc_f(NULL), daacc_f(NULL)
	, force_f(NULL), dforce_f(NULL)
	, moment_f(NULL), dmoment_f(NULL)
	, mass_f(NULL), dmass_f(NULL)
	, inertia_f(NULL), diner_f(NULL)
{

}

dem_simulation::dem_simulation(dem_model *_md)
	: simulation()
	, np(0)
	, nPolySphere(0)
	, dtor(NULL)
	, itor(NULL)
	, md(_md)
	, cm(NULL)
	, itor_type(DEM_INTEGRATOR)
	, pos(NULL), dpos(NULL)
	, vel(NULL), dvel(NULL)
	, acc(NULL), dacc(NULL)
	, avel(NULL), davel(NULL)
	, aacc(NULL), daacc(NULL)
	, force(NULL), dforce(NULL)
	, moment(NULL), dmoment(NULL)
	, mass(NULL), dmass(NULL)
	, inertia(NULL), diner(NULL)
	, pos_f(NULL), dpos_f(NULL)
	, vel_f(NULL), dvel_f(NULL)
	, acc_f(NULL), dacc_f(NULL)
	, avel_f(NULL), davel_f(NULL)
	, aacc_f(NULL), daacc_f(NULL)
	, force_f(NULL), dforce_f(NULL)
	, moment_f(NULL), dmoment_f(NULL)
	, mass_f(NULL), dmass_f(NULL)
	, inertia_f(NULL), diner_f(NULL)
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
	if (mass_f) delete[] mass_f; mass_f = NULL;
	if (inertia_f) delete[] inertia_f; inertia_f = NULL;
	if (pos_f) delete[] pos_f; pos_f = NULL;
	if (vel_f) delete[] vel_f; vel_f = NULL;
	if (acc_f) delete[] acc_f; acc_f = NULL;
	if (avel_f) delete[] avel_f; avel_f = NULL;
	if (aacc_f) delete[] aacc_f; aacc_f = NULL;
	if (force_f) delete[] force_f; force_f = NULL;
	if (moment_f) delete[] moment_f; moment_f = NULL;
	if (simulation::isGpu())
	{
		if (dmass_f) checkCudaErrors(cudaFree(dmass_f)); dmass_f = NULL;
		if (diner_f) checkCudaErrors(cudaFree(diner_f)); diner_f = NULL;
		if (dpos_f) checkCudaErrors(cudaFree(dpos_f)); dpos_f = NULL;
		if (dvel_f) checkCudaErrors(cudaFree(dvel_f)); dvel_f = NULL;
		if (dacc_f) checkCudaErrors(cudaFree(dacc_f)); dacc_f = NULL;
		if (davel_f) checkCudaErrors(cudaFree(davel_f)); davel_f = NULL;
		if (daacc_f) checkCudaErrors(cudaFree(daacc_f)); daacc_f = NULL;
		if (dforce_f) checkCudaErrors(cudaFree(dforce_f)); dforce_f = NULL;
		if (dmoment_f) checkCudaErrors(cudaFree(dmoment_f)); dmoment_f = NULL;
	}
}

void dem_simulation::allocationMemory()
{
	clearMemory();
	//np = _np;
	if (model::isSinglePrecision)
	{
		mass_f = new float[np];
		inertia_f = new float[np];
		pos_f = new float[np * 4];
		vel_f = new float[np * 3];
		acc_f = new float[np * 3];
		avel_f = new float[np * 3];
		aacc_f = new float[np * 3];
		force_f = new float[np * 3];
		moment_f = new float[np * 3];
	}
	else
	{
		mass = new double[np];
		inertia = new double[np];
		pos = new double[np * 4];
		vel = new double[np * 3];
		acc = new double[np * 3];
		avel = new double[np * 3];
		aacc = new double[np * 3];
		force = new double[np * 3];
		moment = new double[np * 3];
	}
	
	if (simulation::isGpu())
	{
		if (model::isSinglePrecision)
		{
			checkCudaErrors(cudaMalloc((void**)&dmass_f, sizeof(float) * np));
			checkCudaErrors(cudaMalloc((void**)&diner_f, sizeof(float) * np));
			checkCudaErrors(cudaMalloc((void**)&dpos_f, sizeof(float) * np * 4));
			checkCudaErrors(cudaMalloc((void**)&dvel_f, sizeof(float) * np * 3));
			checkCudaErrors(cudaMalloc((void**)&dacc_f, sizeof(float) * np * 3));
			checkCudaErrors(cudaMalloc((void**)&davel_f, sizeof(float) * np * 3));
			checkCudaErrors(cudaMalloc((void**)&daacc_f, sizeof(float) * np * 3));
			checkCudaErrors(cudaMalloc((void**)&dforce_f, sizeof(float) * np * 3));
			checkCudaErrors(cudaMalloc((void**)&dmoment_f, sizeof(float) * np * 3));
		}
		else
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
}


bool dem_simulation::initialize(contactManager* _cm)
{
	double maxRadius = 0;
	particleManager* pm = md->ParticleManager();
	np = pm->Np();
	cm = _cm;
	if (pm->RealTimeCreating() && !pm->ChangedParticleModel())
	{
		if (pm->OneByOneCreating())
			per_np = static_cast<unsigned int>((1.0 / pm->NumCreatingPerSecond()) / simulation::dt);
		else
		{
			per_np = static_cast<unsigned int>((pm->TimeCreatingPerGroup() + 1e-9) / simulation::dt);
			//pm->setCreatingPerGroupIterator();
		}
			
	}
	else
		per_np = 0;
	if (cm)
	{
		nPolySphere = cm->setupParticlesPolygonObjectsContact();
		if (nPolySphere)
			maxRadius = cm->ContactParticlesPolygonObjects()->MaxRadiusOfPolySphere();
	}
	
	allocationMemory();

	memcpy(pos, pm->Position(), sizeof(double) * np * 4);
	memset(vel, 0, sizeof(double) * np * 3);
	memset(acc, 0, sizeof(double) * np * 3);
	memset(avel, 0, sizeof(double) * np * 3);
	memset(aacc, 0, sizeof(double) * np * 3);
	memset(force, 0, sizeof(double) * np * 3);
	memset(moment, 0, sizeof(double) * np * 3);

	for (unsigned int i = 0; i < np; i++)
	{
		double r = pos[i * 4 + 3];
		//vel[i * 3 + 0] = 0.1;
		mass[i] = pm->Object()->Density() * (4.0 / 3.0) * M_PI * pow(r, 3.0);
		inertia[i] = (2.0 / 5.0) * mass[i] * pow(r, 2.0);
		force[i * 3 + 0] = mass[i] * model::gravity.x;
		force[i * 3 + 1] = mass[i] * model::gravity.y;
		force[i * 3 + 2] = mass[i] * model::gravity.z;
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
		dtor->initialize(np + nPolySphere);
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
			if (cm->ContactParticles())
				cm->ContactParticles()->cudaMemoryAlloc();
			if (cm->ContactParticlesPolygonObjects())
				cm->ContactParticlesPolygonObjects()->cudaMemoryAlloc();
			foreach(contact* c, cm->Contacts())
				c->cudaMemoryAlloc();		
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
	if (per_np)
		np = 0;
	//dynamic_cast<neighborhood_cell*>(dtor)->reorderElements(pos, (double*)cm->HostSphereData(), np, nPolySphere);
	return true;
}

bool dem_simulation::initialize_f(contactManager* _cm)
{
	float maxRadius = 0;
	particleManager* pm = md->ParticleManager();
	np = pm->Np();
	cm = _cm;
	if (cm)
	{
		nPolySphere = cm->setupParticlesPolygonObjectsContact();
		if (nPolySphere)
			maxRadius = cm->ContactParticlesPolygonObjects()->MaxRadiusOfPolySphere_f();
	}

	allocationMemory();

	memcpy(pos_f, pm->Position_f(), sizeof(float) * np * 4);
	memset(vel_f, 0, sizeof(float) * np * 3);
	memset(acc_f, 0, sizeof(float) * np * 3);
	memset(avel_f, 0, sizeof(float) * np * 3);
	memset(aacc_f, 0, sizeof(float) * np * 3);
	memset(force_f, 0, sizeof(float) * np * 3);
	memset(moment_f, 0, sizeof(float) * np * 3);

	//	nPolySphere = 0;
	// 	foreach(object* obj, modelManager::MM()->GeometryObject()->Objects())
	// 	{
	// 		if (obj->ObjectType() == POLYGON_SHAPE)
	// 		{
	// 			polygonObject* pobj = dynamic_cast<polygonObject*>(obj);
	// 			unsigned int ntri = pobj->NumTriangle();
	// 			memcpy(pos + (np + nPolySphere) * 4, pobj->hostSphereSet(), sizeof(double) * ntri * 4);
	// 			nPolySphere += ntri;
	// 		}
	// 	}

	//VEC3D minWorld;
	for (unsigned int i = 0; i < np; i++)
	{
		float r = pos_f[i * 4 + 3];
		//vel[i * 3 + 0] = 0.1;
		mass_f[i] = static_cast<float>(pm->Object()->Density() * (4.0 / 3.0) * M_PI * pow(r, 3.0));
		inertia_f[i] = (2.0f / 5.0f) * mass_f[i] * pow(r, 2.0f);
		force_f[i * 3 + 0] = mass_f[i] * static_cast<float>(model::gravity.x);
		force_f[i * 3 + 1] = mass_f[i] * static_cast<float>(model::gravity.y);
		force_f[i * 3 + 2] = mass_f[i] * static_cast<float>(model::gravity.z);
		if (r > maxRadius)
			maxRadius = r;
	}
	//	vel[0] = 1.0;
	switch (md->SortType())
	{
	case grid_base::NEIGHBORHOOD: dtor = new neighborhood_cell; break;
	}
	if (dtor)
	{
		dtor->setWorldOrigin(VEC3D(-1.0, -1.0, -1.0));
		dtor->setGridSize(VEC3UI(128, 128, 128));
		dtor->setCellSize(maxRadius * 2.0f);
		dtor->initialize(np + nPolySphere);
	}
	switch (md->IntegrationType())
	{
	case dem_integrator::VELOCITY_VERLET: itor = new velocity_verlet; break;
	}

	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(dpos_f, pos_f, sizeof(float) * np * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dvel_f, vel_f, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dacc_f, acc_f, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(davel_f, avel_f, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(daacc_f, aacc_f, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dforce_f, force_f, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmoment_f, moment_f, sizeof(float) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dmass_f, mass_f, sizeof(float) * np, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(diner_f, inertia_f, sizeof(float) * np, cudaMemcpyHostToDevice));
		if (cm)
		{
			if (cm->ContactParticles())
				cm->ContactParticles()->cudaMemoryAlloc_f();
			if (cm->ContactParticlesPolygonObjects())
				cm->ContactParticlesPolygonObjects()->cudaMemoryAlloc_f();
			foreach(contact* c, cm->Contacts())
				c->cudaMemoryAlloc_f();
		}
		device_parameters_f dp;
		dp.np = np;
		dp.nsphere = 0;
		dp.ncell = dtor->nCell();
		dp.grid_size.x = grid_base::gs.x;
		dp.grid_size.y = grid_base::gs.y;
		dp.grid_size.z = grid_base::gs.z;
		dp.dt = static_cast<float>(simulation::dt);
		dp.half2dt = static_cast<float>(0.5 * dp.dt * dp.dt);
		dp.cell_size = static_cast<float>(grid_base::cs);
		dp.cohesion = 0.0;
		dp.gravity.x = static_cast<float>(model::gravity.x);
		dp.gravity.y = static_cast<float>(model::gravity.y);
		dp.gravity.z = static_cast<float>(model::gravity.z);
		dp.world_origin.x = static_cast<float>(grid_base::wo.x);
		dp.world_origin.y = static_cast<float>(grid_base::wo.y);
		dp.world_origin.z = static_cast<float>(grid_base::wo.z);
		setSymbolicParameter_f(&dp);
	}
	else
	{
		dpos_f = pos_f;
		dvel_f = vel_f;
		dacc_f = acc_f;
		davel_f = avel_f;
		daacc_f = aacc_f;
		dforce_f = force_f;
		dmoment_f = moment_f;
		dmass_f = mass_f;
		diner_f = inertia_f;
	}

	return true;
}

bool dem_simulation::oneStepAnalysis(double ct, unsigned int cstep)
{
	if (per_np && !((cstep-1) % per_np) && np < md->ParticleManager()->Np())
		md->ParticleManager()->OneByOneCreating() ? np++ : np += md->ParticleManager()->NextCreatingPerGroup();

	//qDebug() << np;
	if (itor->integrationType() == dem_integrator::VELOCITY_VERLET)
		itor->updatePosition(dpos, dvel, dacc, np);
	dtor->detection(dpos, (cm ? cm->SphereData() : NULL),  np, nPolySphere);
	if (cm)
	{
		cm->runCollision(
			dpos, dvel, davel,
			dmass, dforce, dmoment,
			dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(), np);
	}
	
	if (itor->integrationType() != dem_integrator::VELOCITY_VERLET)
		itor->updatePosition(dpos, dvel, dacc, np);
	
	itor->updateVelocity(dvel, dacc, davel, daacc, dforce, dmoment, dmass, diner, np);
	return true;
}

bool dem_simulation::oneStepAnalysis_f(double ct, unsigned int cstep)
{
	if (itor->integrationType() == dem_integrator::VELOCITY_VERLET)
		itor->updatePosition(dpos_f, dvel_f, dacc_f, np);
	dtor->detection_f(dpos_f, (cm ? cm->SphereData_f() : NULL), np, nPolySphere);
	//applyMassForce();
	if (cm)
	{
		cm->runCollision(
			dpos_f, dvel_f, davel_f,
			dmass_f, dforce_f, dmoment_f,
			dtor->sortedID(), dtor->cellStart(), dtor->cellEnd(), np);
	}

	if (itor->integrationType() != dem_integrator::VELOCITY_VERLET)
		itor->updatePosition(dpos_f, dvel_f, dacc_f, np);
	itor->updateVelocity(dvel_f, dacc_f, davel_f, daacc_f, dforce_f, dmoment_f, dmass_f, diner_f, np);
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
	//	checkCudaErrors(cudaMemcpy(vv, dvel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
	}
	else
	{
		memcpy(vp, pos, sizeof(double) * 4 * np);
	//	memcpy(vv, vel, sizeof(double) * 3 * np);
	}
	//qDebug() << vp[0] << " " << vp[1] << " " << vp[2] << endl;
	qf.write((char*)&ct, sizeof(double));
	qf.write((char*)&np, sizeof(unsigned int));
	qf.write((char*)vp, sizeof(double) * np * 4);
//	qf.write((char*)vv, sizeof(VEC3D) * np);
// 	qf.write((char*)pr, sizeof(double) * np);
// 	qf.write((char*)fs, sizeof(bool) * np);
	qf.close();
	return fname;
}

QString dem_simulation::saveResult_f(float *vp, float* vv, double ct, unsigned int pt)
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
		checkCudaErrors(cudaMemcpy(vp, dpos_f, sizeof(float) * np * 4, cudaMemcpyDeviceToHost));
		//	checkCudaErrors(cudaMemcpy(vv, dvel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
	}
	else
	{
		memcpy(vp, pos_f, sizeof(float) * 4 * np);
		//	memcpy(vv, vel, sizeof(double) * 3 * np);
	}
	//qDebug() << vp[0] << " " << vp[1] << " " << vp[2] << endl;
	qf.write((char*)&ct, sizeof(double));
	qf.write((char*)&np, sizeof(unsigned int));
	qf.write((char*)vp, sizeof(float) * np * 4);
	//	qf.write((char*)vv, sizeof(VEC3D) * np);
	// 	qf.write((char*)pr, sizeof(double) * np);
	// 	qf.write((char*)fs, sizeof(bool) * np);
	qf.close();
	return fname;
}

void dem_simulation::saveFinalResult(QFile& qf)
{
	int flag = 1;
	int precision = 2;
	qf.write((char*)&precision, sizeof(int));
	qf.write((char*)&flag, sizeof(int));
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(pos, dpos, sizeof(double) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vel, dvel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(avel, davel, sizeof(double) * np * 3, cudaMemcpyDeviceToHost));
	}
	qf.write((char*)pos, sizeof(double) * np * 4);
	qf.write((char*)vel, sizeof(double) * np * 3);
	qf.write((char*)avel, sizeof(double) * np * 3);
}

void dem_simulation::saveFinalResult_f(QFile& qf)
{
	int flag = 1;
	int precision = 1;
	qf.write((char*)&precision, sizeof(int));
	qf.write((char*)&flag, sizeof(int));
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(pos_f, dpos_f, sizeof(float) * np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(vel_f, dvel_f, sizeof(float) * np * 3, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(avel_f, davel_f, sizeof(float) * np * 3, cudaMemcpyDeviceToHost));
	}
	qf.write((char*)pos_f, sizeof(float) * np * 4);
	qf.write((char*)vel_f, sizeof(float) * np * 3);
	qf.write((char*)avel_f, sizeof(float) * np * 3);
}

void dem_simulation::setStartingData(startingModel* stm)
{
	stm->copyDEMData(np, pos, vel, avel);
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(dpos, pos, sizeof(double) * np * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dvel, vel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(davel, avel, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	}
}