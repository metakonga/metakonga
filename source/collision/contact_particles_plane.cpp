#include "contact_particles_plane.h"
#include "algebraMath.h"

contact_particles_plane::contact_particles_plane(
	QString _name, contactForce_type t, object* o1, object* o2)
	: contact(_name, t)
	, pe(NULL)
	, dpi(NULL)
{
	contact::iobj = o1;
	contact::jobj = o2;
	pe = dynamic_cast<plane*>((o1->ObjectType() == PLANE ? o1 : o2));
	p = o1->ObjectType() != PLANE ? o1 : o2;
}

contact_particles_plane::contact_particles_plane(const contact* c)
	: contact(c)
	, dpi(NULL)
{

}

contact_particles_plane::~contact_particles_plane()
{
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
}

void contact_particles_plane::setPlane(plane* _pe)
{
	pe = _pe;
}

void contact_particles_plane::cuda_collision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	geometry_motion_condition gmc = pe->MotionCondition();
	if (gmc.enable && simulation::ctime >= gmc.st)
	{
		hpi.xw = pe->XW();
		hpi.w2 = pe->W2();
		hpi.w3 = pe->W3();
		hpi.w4 = pe->W4();
		hpi.pa = pe->PA();
		hpi.pb = pe->PB();
		hpi.u1 = pe->U1();
		hpi.u2 = pe->U2();
		hpi.uw = pe->UW();
		
		checkCudaErrors(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
	}
	cu_plane_contact_force(1, dpi, pos, vel, omega, force, moment, mass, np, dcp);
}

void contact_particles_plane::cuda_collision(float *pos, float *vel, float *omega, float *mass, float *force, float *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	cu_plane_contact_force(1, dpi_f, pos, vel, omega, force, moment, mass, np, dcp_f);
}

void contact_particles_plane::collision(
	double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& F, VEC3D& M)
{
// 	simulation::isGpu()
// 		? cu_plane_contact_force
// 		(
// 		1, dpi, pos, vel, omega, force, moment, mass, np, dcp
// 		)
// 		: hostCollision
// 		(
// 		pos, vel, omega, mass, force, moment, np
// 		);
// 
// 	return true;
	singleCollision(pe, m, r, pos, vel, omega, F, M);
// 	;// force[i] += F;
// 	;// moment[i] += M;
}

void contact_particles_plane::cudaMemoryAlloc_planeObject()
{
	//device_plane_info *_dpi = new device_plane_info;
	hpi.l1 = pe->L1();
	hpi.l2 = pe->L2();
	hpi.xw = pe->XW();
	hpi.uw = pe->UW();
	hpi.u1 = pe->U1();
	hpi.u2 = pe->U2();
	hpi.pa = pe->PA();
	hpi.pb = pe->PB();
	hpi.w2 = pe->W2();
	hpi.w3 = pe->W3();
	hpi.w4 = pe->W4();
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));
	checkCudaErrors(cudaMemcpy(dpi, &hpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
	//delete _dpi;
}

void contact_particles_plane::cudaMemoryAlloc()
{
	contact::cudaMemoryAlloc();
	cudaMemoryAlloc_planeObject();
}

void contact_particles_plane::cudaMemoryAlloc_f()
{
	contact::cudaMemoryAlloc_f();
	device_plane_info_f *_dpi = new device_plane_info_f;
	_dpi->l1 = pe->L1();
	_dpi->l2 = pe->L2();
	_dpi->xw = make_float3(pe->XW().x, pe->XW().y, pe->XW().z);
	_dpi->uw = make_float3(pe->UW().x, pe->UW().y, pe->UW().z);
	_dpi->u1 = make_float3(pe->U1().x, pe->U1().y, pe->U1().z);
	_dpi->u2 = make_float3(pe->U2().x, pe->U2().y, pe->U2().z);
	_dpi->pa = make_float3(pe->PA().x, pe->PA().y, pe->PA().z);
	_dpi->pb = make_float3(pe->PB().x, pe->PB().y, pe->PB().z);
	_dpi->w2 = make_float3(pe->W2().x, pe->W2().y, pe->W2().z);
	_dpi->w3 = make_float3(pe->W3().x, pe->W3().y, pe->W3().z);
	_dpi->w4 = make_float3(pe->W4().x, pe->W4().y, pe->W4().z);
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info_f)));
	checkCudaErrors(cudaMemcpy(dpi, _dpi, sizeof(device_plane_info_f), cudaMemcpyHostToDevice));
	delete _dpi;
}

double contact_particles_plane::particle_plane_contact_detection(
	plane* _pe, VEC3D& u, VEC3D& xp, VEC3D& wp, double r
	)
{
	double a_l1 = pow(wp.x - _pe->L1(), 2.0);
	double b_l2 = pow(wp.y - _pe->L2(), 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r*r;

	// The sphere contacts with the wall face
	if (abs(wp.z) < r && (wp.x > 0 && wp.x < _pe->L1()) && (wp.y > 0 && wp.y < _pe->L2())){
		VEC3D dp = xp - _pe->XW();
		vector3<double> uu = _pe->UW() / _pe->UW().length();
		int pp = -sign(dp.dot(_pe->UW()));
		u = pp * uu;
		double collid_dist = r - abs(dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		VEC3D Xsw = xp - _pe->XW();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > _pe->L1() && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		VEC3D Xsw = xp - _pe->W2();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > _pe->L1() && wp.y > _pe->L2() && (a_l1 + b_l2 + sqc) < sqr){
		VEC3D Xsw = xp - _pe->W3();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > _pe->L2() && (sqa + b_l2 + sqc) < sqr){
		VEC3D Xsw = xp - _pe->W4();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < _pe->L1()) && wp.y < 0 && (sqb + sqc) < sqr){
		VEC3D Xsw = xp - _pe->XW();
		VEC3D wj_wi = _pe->W2() - _pe->XW();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < _pe->L1()) && wp.y > _pe->L2() && (b_l2 + sqc) < sqr){
		VEC3D Xsw = xp - _pe->W4();
		VEC3D wj_wi = _pe->W3() - _pe->W4();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < _pe->L2()) && wp.x < 0 && (sqr + sqc) < sqr){
		VEC3D Xsw = xp - _pe->XW();
		VEC3D wj_wi = _pe->W4() - _pe->XW();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < _pe->L2()) && wp.x > _pe->L1() && (a_l1 + sqc) < sqr){
		VEC3D Xsw = xp - _pe->W2();
		VEC3D wj_wi = _pe->W3() - _pe->W2();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;
	}


	return -1.0f;
}

void contact_particles_plane::singleCollision(
	plane* _pe, double mass, double rad, VEC3D& pos, VEC3D& vel, 
	VEC3D& omega, VEC3D& force, VEC3D& moment)
{
	VEC3D dp = pos - _pe->XW();
	VEC3D wp = VEC3D(dp.dot(_pe->U1()), dp.dot(_pe->U2()), dp.dot(_pe->UW()));
	VEC3D u;

	double cdist = particle_plane_contact_detection(_pe, u, pos, wp, rad);
	if (cdist > 0){
		double rcon = rad - 0.5 * cdist;
		VEC3D cp = rcon * u;
		//unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
		//VEC3D c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
		VEC3D dv = -(vel);// +omega.cross(rad * u));

		contactParameters c = getContactParameters(
			rad, 0.0,
			mass, 0.0,
			mpp->Ei, mpp->Ej,
			mpp->pri, mpp->prj,
			mpp->Gi, mpp->Gj);
		switch (f_type)
		{
		case DHS: DHSModel(c, cdist, cp, dv, u, force, moment); break;
		}
	}
}

// bool contact_particles_plane::hostCollision(
// 	double *m_pos, double *m_vel, double *m_omega,
// 	double *m_mass, double *m_force, double *m_moment,
// 	unsigned int np)
// {
// 	//contactParameters c;
// 	VEC4D* pos = (VEC4D*)m_pos;
// 	VEC3D* vel = (VEC3D*)m_vel;
// 	VEC3D* omega = (VEC3D*)m_omega;
// 	VEC3D* force = (VEC3D*)m_force;
// 	VEC3D* moment = (VEC3D*)m_moment;
// 	double* mass = m_mass;	
// //	double dt = simulation::ctime;
// 	for (unsigned int i = 0; i < np; i++)
// 	{
// 		double ms = mass[i];
// 		double rad = pos[i].w;
// 		VEC3D p = VEC3D(pos[i].x, pos[i].y, pos[i].z);
// 		VEC3D v = vel[i];
// 		VEC3D w = omega[i];
// 		VEC3D F, M;
// 		singleCollision(pe, ms, rad, p, v, w, F, M);
// 		force[i] += F;
// 		moment[i] += M;
// 	}
// 	return true;
// }


// #include "collision_particles_plane.h"
// #include "particle_system.h"
// #include "plane.h"
// 
// collision_particles_plane::collision_particles_plane()
// {
// 
// }
// 
// collision_particles_plane::collision_particles_plane(
// 	QString& _name, 
// 	modeler* _md, 
// 	particle_system *_ps, 
// 	plane *_p, 
// 	tContactModel _tcm)
// 	: collision(_name, _md, _ps->name(), _p->objectName(), PARTICLES_PLANE, _tcm)
// 	, ps(_ps)
// 	, pe(_p)
// {
// 
// }
// 
// collision_particles_plane::~collision_particles_plane()
// {
// 
// }
// 
// bool collision_particles_plane::collid(double dt)
// {
// 	if (ps->particleCluster().size())
// 	{
// 		foreach(particle_cluster* value, ps->particleCluster())
// 		{
// 			for (int i = 0; i < value->perCluster(); i++)
// 			{
// 				unsigned int id = value->indice(i);
// 				switch (tcm)
// 				{
// 				case HMCM: this->HMCModel(id, dt); break;
// 				case DHS: this->DHSModel(id, dt); break;
// 				}
// 			}
// 		}
// 	}
// 	else
// 	{
// 		for (unsigned int i = 0; i < ps->numParticle(); i++){
// 			switch (tcm)
// 			{
// 			case HMCM: this->HMCModel(i, dt); break;
// 			case DHS: this->DHSModel(i, dt); break;
// 			}
// 		}
// 	}
// 	
// // 	switch (tcm)
// // 	{
// // 	case HMCM: this->HMCModel(i, dt); break;
// // 	case DHS: this->DHSModel(i, dt); break;
// // 	}
// 	return true;
// }
// 
// bool collision_particles_plane::cuCollid(
// 	double *dpos, double *dvel,
// 	double *domega, double *dmass,
// 	double *dforce,	double *dmoment, unsigned int np)
// {
// 	switch (tcm)
// 	{
// 	case HMCM: cu_plane_hertzian_contact_force(
// 			0, pe->devicePlaneInfo(), 
// 			dpos, dvel, domega, 
// 			dforce, dmoment, dmass, 
// 			ps->numParticle(), dcp); 
// 		break;
// 	case DHS:
// 		cu_plane_hertzian_contact_force(
// 			1, pe->devicePlaneInfo(),
// 			dpos, dvel, domega,
// 			dforce, dmoment, dmass,
// 			ps->numParticle(), dcp);
// 		break;
// 	}
// 	
// 	return true;
// }
// 
// double collision_particles_plane::particle_plane_contact_detection(VEC3D& u, VEC3D& xp, VEC3D& wp, double r)
// {
// 	double a_l1 = pow(wp.x - pe->L1(), 2.0);
// 	double b_l2 = pow(wp.y - pe->L2(), 2.0);
// 	double sqa = wp.x * wp.x;
// 	double sqb = wp.y * wp.y;
// 	double sqc = wp.z * wp.z;
// 	double sqr = r*r;
// 
// 	// The sphere contacts with the wall face
// 	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe->L1()) && (wp.y > 0 && wp.y < pe->L2())){
// 		VEC3D dp = xp - pe->XW();
// 		vector3<double> uu = pe->UW() / pe->UW().length();
// 		int pp = -sign(dp.dot(pe->UW()));
// 		u = pp * uu;
// 		double collid_dist = r - abs(dp.dot(u));
// 		return collid_dist;
// 	}
// 
// 	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->XW();
// 		double h = Xsw.length();
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x > pe->L1() && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->W2();
// 		double h = Xsw.length();
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x > pe->L1() && wp.y > pe->L2() && (a_l1 + b_l2 + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->W3();
// 		double h = Xsw.length();
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	else if (wp.x < 0 && wp.y > pe->L2() && (sqa + b_l2 + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->W4();
// 		double h = Xsw.length();
// 		u = Xsw / h;
// 		return r - h;
// 	}
// 	if ((wp.x > 0 && wp.x < pe->L1()) && wp.y < 0 && (sqb + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->XW();
// 		VEC3D wj_wi = pe->W2() - pe->XW();
// 		VEC3D us = wj_wi / wj_wi.length();
// 		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
// 		double h = h_star.length();
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 	else if ((wp.x > 0 && wp.x < pe->L1()) && wp.y > pe->L2() && (b_l2 + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->W4();
// 		VEC3D wj_wi = pe->W3() - pe->W4();
// 		VEC3D us = wj_wi / wj_wi.length();
// 		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
// 		double h = h_star.length();
// 		u = -h_star / h;
// 		return r - h;
// 
// 	}
// 	else if ((wp.x > 0 && wp.y < pe->L2()) && wp.x < 0 && (sqr + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->XW();
// 		VEC3D wj_wi = pe->W4() - pe->XW();
// 		VEC3D us = wj_wi / wj_wi.length();
// 		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
// 		double h = h_star.length();
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 	else if ((wp.x > 0 && wp.y < pe->L2()) && wp.x > pe->L1() && (a_l1 + sqc) < sqr){
// 		VEC3D Xsw = xp - pe->W2();
// 		VEC3D wj_wi = pe->W3() - pe->W2();
// 		VEC3D us = wj_wi / wj_wi.length();
// 		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
// 		double h = h_star.length();
// 		u = -h_star / h;
// 		return r - h;
// 	}
// 
// 	
// 	return -1.0f;
// }
// 
// bool collision_particles_plane::collid_with_particle(unsigned int i, double dt)
// {
// 	switch (tcm)
// 	{
// 	case HMCM: this->HMCModel(i, dt); break;
// 	case DHS: this->DHSModel(i, dt); break;
// 	}
// 	return true;
// }
// 
// bool collision_particles_plane::DHSModel(unsigned int i, double dt)
// {
// 	double ms = ps->mass()[i];
// 	VEC4D p = ps->position()[i];
// 	VEC3D v = ps->velocity()[i];
// 	VEC3D w = ps->angVelocity()[i];
// 	VEC3D Fn, Ft, M;
// 	VEC3D dp = p.toVector3() - pe->XW();
// 	VEC3D wp = VEC3D(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
// 	VEC3D u;
// 
// 	double cdist = particle_plane_contact_detection(u, p.toVector3(), wp, p.w);
// 	if (cdist > 0){
// 		double rcon = p.w - 0.5 * cdist;
// 		VEC3D cp = p.toVector3() + rcon * u;
// 		unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
// 		VEC3D c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
// 		VEC3D dv = -(v + w.cross(p.w * u));
// 
// 		constant c = getConstant(p.w, 0.0, ms, 0.0, ps->youngs(), pe->youngs(), ps->poisson(), pe->poisson(), ps->shear(), pe->shear());
// 		//double collid_dist2 = particle_plane_contact_detection(unit, p, wp, rad);
// 
// 		double fsn = (-c.kn * pow(cdist, 1.5));
// 		double fca = cohesionForce(p.w, 0.0, ps->youngs(), 0.0, ps->poisson(), 0.0, fsn);
// 		double fsd = c.vn * dv.dot(u);
// 		Fn = (fsn + fca + c.vn * dv.dot(u)) * u;
// 		//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
//  		VEC3D e = dv - dv.dot(u) * u;
// 		double mag_e = e.length();
// 		//vector3<double> shf;
// 		if (mag_e){
// 			VEC3D s_hat = e / mag_e;
// 			double ds = mag_e * dt;
// 			double ft1 = c.ks * ds + c.vs * dv.dot(s_hat);
// 			double ft2 = c.mu * Fn.length();
// 			Ft = min(ft1, ft2) * s_hat;
// 			//M = (p.w * u).cross(Ft);
// 			M = c2p.cross(Ft);
// 		}
// 		//mf = sf + shf;
// 		ps->force()[i] += Fn + Ft;
// 		//ps->moment()[i] += M;
// 		ps->moment()[i] += c2p.cross(Ft + Fn);
// 	}
// 	return true;
// }
// 
// bool collision_particles_plane::HMCModel(unsigned int i, double dt)
// {
// 	double ms = ps->mass()[i];
// 	VEC4D p = ps->position()[i];
// 	VEC3D v = ps->velocity()[i];
// 	VEC3D w = ps->angVelocity()[i];
// 	VEC3D Fn, Ft, M;
// 	VEC3D dp = p.toVector3() - pe->XW();
// 	VEC3D wp = VEC3D(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
// 	VEC3D u;
// 	
// 	double cdist = particle_plane_contact_detection(u, p.toVector3(), wp, p.w);
// 	if (cdist > 0){
// 		double rcon = p.w - 0.5 * cdist;
// 		VEC3D rv = -(v + w.cross(p.w * u));
// 		constant c = getConstant(p.w, 0.0, ms, 0.0, ps->youngs(), pe->youngs(), ps->poisson(), pe->poisson(), ps->shear(), pe->shear());
// 		double fsn = -c.kn * pow(cdist, 1.5);
// 		double fdn = c.vn * rv.dot(u);
// 		Fn = (fsn + fdn) * u;
// 		VEC3D e = rv - rv.dot(u) * u;
// 		double mag_e = e.length();
// 		VEC3D Ft;
// 		if (mag_e){
// 			VEC3D sh = -(e / mag_e);
// 			double ds = mag_e * dt;
// 			double fst = -c.ks * ds;
// 			double fdt = c.vs * rv.dot(sh);
// 			Ft = (fst + fdt) * sh;
// 			if (Ft.length() >= c.mu * Fn.length())
// 				Ft = c.mu * fsn * sh;
// 			M = (rcon * u).cross(Ft);
// 			if (w.length()){
// 				VEC3D on = w / w.length();
// 				M += c.rf * fsn * rcon * on;
// 			}
// 		}
// 		ps->force()[i] += Fn + Ft;
// 		ps->moment()[i] += M;
// 
// //  		double fsn = (-c.kn * pow(collid_dist, 1.5f));
// //  		double fca = cohesionForce(p.w, 0.f, ps->youngs(), 0.f, ps->poisson(), 0.f, fsn);
// //  		//double fsd = c.vn * dv.dot(unit);
// //  		Fn = (fsn + fca + c.vn * dv.dot(unit)) * unit;
// // 		//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
// // 		vector3<double> e = dv - dv.dot(unit) * unit;
// // 		double mag_e = e.length();
// // 		vector3<double> shf;
// // 		if (mag_e){
// // 			vector3<double> s_hat = e / mag_e;
// // 			double ds = mag_e * dt;
// // 			Ft = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * Fn.length()) * s_hat;
// // 			M = (p.w * unit).cross(Ft);
// // 		}
// // 		//mf = sf + shf;
// // 		ps->force()[i] += Fn + Ft;
// // 		ps->moment()[i] += M;
// 	}
// 
// 	//ps->force()[i] += mf;
// 	//ps->moment()[i] += mm;
// 	return true;
// }