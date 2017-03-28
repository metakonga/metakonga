#include "collision_particles_plane.h"
#include "particle_system.h"
#include "plane.h"

collision_particles_plane::collision_particles_plane()
{

}

collision_particles_plane::collision_particles_plane(
	QString& _name, 
	modeler* _md, 
	particle_system *_ps, 
	plane *_p, 
	tContactModel _tcm)
	: collision(_name, _md, _ps->name(), _p->objectName(), PARTICLES_PLANE, _tcm)
	, ps(_ps)
	, pe(_p)
{

}

collision_particles_plane::~collision_particles_plane()
{

}

bool collision_particles_plane::collid(double dt)
{
	for (unsigned int i = 0; i < ps->numParticle(); i++){
		switch (tcm)
		{
		case HMCM: this->HMCModel(i, dt); break;
		case DHS: this->DHSModel(i, dt); break;
		}
	}
// 	switch (tcm)
// 	{
// 	case HMCM: this->HMCModel(i, dt); break;
// 	case DHS: this->DHSModel(i, dt); break;
// 	}
	return true;
}

bool collision_particles_plane::cuCollid()
{
	switch (tcm)
	{
	case HMCM: cu_plane_hertzian_contact_force(
			0, pe->devicePlaneInfo(), 
			ps->cuPosition(), ps->cuVelocity(), ps->cuOmega(), 
			ps->cuForce(), ps->cuMoment(), ps->cuMass(), 
			ps->numParticle(), dcp); 
		break;
	case DHS:
		cu_plane_hertzian_contact_force(
			1, pe->devicePlaneInfo(),
			ps->cuPosition(), ps->cuVelocity(), ps->cuOmega(),
			ps->cuForce(), ps->cuMoment(), ps->cuMass(),
			ps->numParticle(), dcp);
		break;
	}
	
	return true;
}

double collision_particles_plane::particle_plane_contact_detection(VEC3D& u, VEC3D& xp, VEC3D& wp, double r)
{
	double a_l1 = pow(wp.x - pe->L1(), 2.0);
	double b_l2 = pow(wp.y - pe->L2(), 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r*r;

	// The sphere contacts with the wall face
	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe->L1()) && (wp.y > 0 && wp.y < pe->L2())){
		VEC3D dp = xp - pe->XW();
		vector3<double> uu = pe->UW() / pe->UW().length();
		int pp = -sign(dp.dot(pe->UW()));
		u = pp * uu;
		double collid_dist = r - abs(dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		VEC3D Xsw = xp - pe->XW();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->L1() && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		VEC3D Xsw = xp - pe->W2();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->L1() && wp.y > pe->L2() && (a_l1 + b_l2 + sqc) < sqr){
		VEC3D Xsw = xp - pe->W3();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe->L2() && (sqa + b_l2 + sqc) < sqr){
		VEC3D Xsw = xp - pe->W4();
		double h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe->L1()) && wp.y < 0 && (sqb + sqc) < sqr){
		VEC3D Xsw = xp - pe->XW();
		VEC3D wj_wi = pe->W2() - pe->XW();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe->L1()) && wp.y > pe->L2() && (b_l2 + sqc) < sqr){
		VEC3D Xsw = xp - pe->W4();
		VEC3D wj_wi = pe->W3() - pe->W4();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe->L2()) && wp.x < 0 && (sqr + sqc) < sqr){
		VEC3D Xsw = xp - pe->XW();
		VEC3D wj_wi = pe->W4() - pe->XW();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe->L2()) && wp.x > pe->L1() && (a_l1 + sqc) < sqr){
		VEC3D Xsw = xp - pe->W2();
		VEC3D wj_wi = pe->W3() - pe->W2();
		VEC3D us = wj_wi / wj_wi.length();
		VEC3D h_star = Xsw - (Xsw.dot(us)) * us;
		double h = h_star.length();
		u = -h_star / h;
		return r - h;
	}

	
	return -1.0f;
}

bool collision_particles_plane::collid_with_particle(unsigned int i, double dt)
{
	switch (tcm)
	{
	case HMCM: this->HMCModel(i, dt); break;
	case DHS: this->DHSModel(i, dt); break;
	}
	return true;
}

bool collision_particles_plane::DHSModel(unsigned int i, double dt)
{
	double ms = ps->mass()[i];
	VEC4D p = ps->position()[i];
	VEC3D v = ps->velocity()[i];
	VEC3D w = ps->angVelocity()[i];
	VEC3D Fn, Ft, M;
	VEC3D dp = p.toVector3() - pe->XW();
	VEC3D wp = VEC3D(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
	VEC3D u;

	double cdist = particle_plane_contact_detection(u, p.toVector3(), wp, p.w);
	if (cdist > 0){
		double rcon = p.w - 0.5 * cdist;
		VEC3D cp = p.toVector3() + rcon * u;
		VEC3D c2p = cp - ps->getParticleClusterFromParticleID(i)->center();
		VEC3D dv = -(v + w.cross(p.w * u));

		constant c = getConstant(p.w, 0.0, ms, 0.0, ps->youngs(), pe->youngs(), ps->poisson(), pe->poisson(), ps->shear(), pe->shear());
		//double collid_dist2 = particle_plane_contact_detection(unit, p, wp, rad);

		double fsn = (-c.kn * pow(cdist, 1.5));
		double fca = cohesionForce(p.w, 0.0, ps->youngs(), 0.0, ps->poisson(), 0.0, fsn);
		double fsd = c.vn * dv.dot(u);
		Fn = (fsn + fca + c.vn * dv.dot(u)) * u;
		//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
 		VEC3D e = dv - dv.dot(u) * u;
		double mag_e = e.length();
		//vector3<double> shf;
		if (mag_e){
			VEC3D s_hat = e / mag_e;
			double ds = mag_e * dt;
			double ft1 = c.ks * ds + c.vs * dv.dot(s_hat);
			double ft2 = c.mu * Fn.length();
			Ft = min(ft1, ft2) * s_hat;
			//M = (p.w * u).cross(Ft);
			M = c2p.cross(Ft);
		}
		//mf = sf + shf;
		ps->force()[i] += Fn + Ft;
		//ps->moment()[i] += M;
		ps->moment()[i] += c2p.cross(Ft + Fn);
	}
	return true;
}

bool collision_particles_plane::HMCModel(unsigned int i, double dt)
{
	double ms = ps->mass()[i];
	VEC4D p = ps->position()[i];
	VEC3D v = ps->velocity()[i];
	VEC3D w = ps->angVelocity()[i];
	VEC3D Fn, Ft, M;
	VEC3D dp = p.toVector3() - pe->XW();
	VEC3D wp = VEC3D(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
	VEC3D u;
	
	double cdist = particle_plane_contact_detection(u, p.toVector3(), wp, p.w);
	if (cdist > 0){
		double rcon = p.w - 0.5 * cdist;
		VEC3D rv = -(v + w.cross(p.w * u));
		constant c = getConstant(p.w, 0.0, ms, 0.0, ps->youngs(), pe->youngs(), ps->poisson(), pe->poisson(), ps->shear(), pe->shear());
		double fsn = -c.kn * pow(cdist, 1.5);
		double fdn = c.vn * rv.dot(u);
		Fn = (fsn + fdn) * u;
		VEC3D e = rv - rv.dot(u) * u;
		double mag_e = e.length();
		VEC3D Ft;
		if (mag_e){
			VEC3D sh = -(e / mag_e);
			double ds = mag_e * dt;
			double fst = -c.ks * ds;
			double fdt = c.vs * rv.dot(sh);
			Ft = (fst + fdt) * sh;
			if (Ft.length() >= c.mu * Fn.length())
				Ft = c.mu * fsn * sh;
			M = (rcon * u).cross(Ft);
			if (w.length()){
				VEC3D on = w / w.length();
				M += c.rf * fsn * rcon * on;
			}
		}
		ps->force()[i] += Fn + Ft;
		ps->moment()[i] += M;

//  		double fsn = (-c.kn * pow(collid_dist, 1.5f));
//  		double fca = cohesionForce(p.w, 0.f, ps->youngs(), 0.f, ps->poisson(), 0.f, fsn);
//  		//double fsd = c.vn * dv.dot(unit);
//  		Fn = (fsn + fca + c.vn * dv.dot(unit)) * unit;
// 		//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
// 		vector3<double> e = dv - dv.dot(unit) * unit;
// 		double mag_e = e.length();
// 		vector3<double> shf;
// 		if (mag_e){
// 			vector3<double> s_hat = e / mag_e;
// 			double ds = mag_e * dt;
// 			Ft = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * Fn.length()) * s_hat;
// 			M = (p.w * unit).cross(Ft);
// 		}
// 		//mf = sf + shf;
// 		ps->force()[i] += Fn + Ft;
// 		ps->moment()[i] += M;
	}

	//ps->force()[i] += mf;
	//ps->moment()[i] += mm;
	return true;
}