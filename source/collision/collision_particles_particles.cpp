#include "collision_particles_particles.h"
#include "grid_base.h"
#include "particle_system.h"


collision_particles_particles::collision_particles_particles()
{

}

collision_particles_particles::collision_particles_particles(
	QString& _name, 
	modeler* _md,
	particle_system* _ps,
	tContactModel _tcm)
	: collision(_name, _md, _ps->name(), _ps->name(), NO_COLLISION_PAIR, _tcm)
	, ps(_ps)
{
	
}

collision_particles_particles::~collision_particles_particles()
{

}

bool collision_particles_particles::cuCollid(
	double *dpos, double *dvel,
	double *domega, double *dmass,
	double *dforce, double *dmoment, unsigned int np)
{
	switch (tcm)
	{
	case HMCM: 
		cu_calculate_p2p(
			0, dpos, dvel, domega, dforce, dmoment, dmass,
			grid_base::cuSortedID(), grid_base::cuCellStart(), grid_base::cuCellEnd(), dcp, ps->numParticle());
		break;
	case DHS:
		cu_calculate_p2p(
			1, dpos, dvel, domega, dforce, dmoment, dmass,
			grid_base::cuSortedID(), grid_base::cuCellStart(), grid_base::cuCellEnd(), dcp, ps->numParticle());
		break;
	}
	return true;
}

bool collision_particles_particles::collid(double dt)
{

	switch (tcm)
	{
	case HMCM: this->HMCModel(dt); break;
	case DHS: this->DHSModel(dt); break;
	}
	return true;
}

bool collision_particles_particles::DHSModel(double dt)
{
	//cohesion = 1.0E+6;
	unsigned int _np = 0;
	VEC3I neigh, gp;
	double dist, cdist, mag_e, ds;
	unsigned int hash, sid, eid;
	constant c;
	VEC3D ipos, jpos, rp, u, rv, Fn, e, sh, M;
	VEC4D *pos = ps->position();
	VEC3D *vel = ps->velocity();
	VEC3D *omega = ps->angVelocity();
	VEC3D *fr = ps->force();
	VEC3D *mm = ps->moment();
	double* ms = ps->mass();
	if (ps->particleCluster().size())
		_np = ps->particleCluster().size() * 2;
	else
		_np = ps->numParticle();
	for (unsigned int i = 0; i < _np; i++){
		ipos = VEC3D(pos[i].x, pos[i].y, pos[i].z);
		gp = grid_base::getCellNumber(pos[i].x, pos[i].y, pos[i].z);

// 		fr[i] = 0.0f;// ms[i] * md->gravity();
// 		mm[i] = 0.0f;
		for (int z = -1; z <= 1; z++){
			for (int y = -1; y <= 1; y++){
				for (int x = -1; x <= 1; x++){
					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
					hash = grid_base::getHash(neigh);
					sid = grid_base::cellStart(hash);
					if (sid != 0xffffffff){
						eid = grid_base::cellEnd(hash);
						for (unsigned int j = sid; j < eid; j++){
							unsigned int k = grid_base::sortedID(j);
							if (i == k || k >= md->numParticle())
								continue;
// 							if (abs((int)(i - k)) < 2)
// 								continue;
// 							if (((i % 2) == 0 || (k % 2) == 0) && abs((int)(i - k)) == 1)
// 								continue;
							jpos = pos[k].toVector3();
							rp = jpos - ipos;
							dist = rp.length();
							cdist = (pos[i].w + pos[k].w) - dist;
							//double rcon = pos[i].w - cdist;
							unsigned int rid = 0;
							if (cdist > 0.0001)
								continue;
							if (cdist > 0){
								u = rp / dist;
								VEC3D cp = ipos + pos[i].w * u;
								unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
								VEC3D c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
								double rcon = pos[i].w - 0.5 * cdist;
								rv = vel[k] + omega[k].cross(-pos[k].w * u) - (vel[i] + omega[i].cross(pos[i].w * u));
								c = getConstant(pos[i].w, pos[k].w, ms[i], ms[k], ps->youngs(), ps->youngs(), ps->poisson(), ps->poisson(), ps->shear(), ps->shear());
								double cor = u.length();
								double fsn = -c.kn * pow(cdist, 1.5f);
								double fca = cohesionForce(pos[i].w, pos[k].w, ps->youngs(), ps->youngs(), ps->poisson(), ps->poisson(), fsn);
								double fsd = rv.dot(u) * c.vn;

								Fn = (fsn + fca + fsd) * u;
								e = rv - rv.dot(u) * u;
								mag_e = e.length();
								VEC3D Ft;
								if (mag_e){
									sh = e / mag_e;
									ds = mag_e * dt;
									double ft1 = c.ks * ds + c.vs * (rv.dot(sh));
									double ft2 = c.mu * Fn.length();
									Ft = min(ft1, ft2) * sh;
									M = (pos[i].w * u).cross(Ft);
								}
								fr[i] += Fn + Ft;
								mm[i] += c2p.cross(Fn + Ft);
							}
						}
					}
				}
			}
		}
	}
	return true;
}

bool collision_particles_particles::HMCModel(double dt)
{
	//cohesion = 1.0E+6;
	VEC3I neigh, gp;
	double dist, cdist, mag_e, ds;
	unsigned int hash, sid, eid;
	constant c;
	VEC3D ipos, jpos, rp, u, rv, Fn, e, sh, M;
	VEC4D *pos = ps->position();
	VEC3D *vel = ps->velocity();
	VEC3D *omega = ps->angVelocity();
	VEC3D *fr = ps->force();
	VEC3D *mm = ps->moment();
	double* ms = ps->mass();
	for (unsigned int i = 0; i < ps->numParticle(); i++){
		ipos = VEC3D(pos[i].x, pos[i].y, pos[i].z);
		gp = grid_base::getCellNumber(pos[i].x, pos[i].y, pos[i].z);
		fr[i] = ms[i] * md->gravity();
		mm[i] = 0.0f;
		for (int z = -1; z <= 1; z++){
			for (int y = -1; y <= 1; y++){
				for (int x = -1; x <= 1; x++){
					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
					hash = grid_base::getHash(neigh);
					sid = grid_base::cellStart(hash);
					if (sid != 0xffffffff){
						eid = grid_base::cellEnd(hash);
						for (unsigned int j = sid; j < eid; j++){
							unsigned int k = grid_base::sortedID(j);
							if (i == k || k >= md->numParticle())
								continue;
							jpos = pos[k].toVector3();
							rp = jpos - ipos;
							dist = rp.length();
							cdist = (pos[i].w + pos[k].w) - dist;
							//double rcon = pos[i].w - cdist;
							unsigned int rid = 0;
							if (cdist > 0){	
								double rcon = pos[i].w - 0.5 * cdist;
								u = rp / dist;
								rv = vel[k] + omega[k].cross(-pos[k].w * u) - (vel[i] + omega[i].cross(pos[i].w * u));
								c = getConstant(pos[i].w, pos[k].w, ms[i], ms[k], ps->youngs(), ps->youngs(), ps->poisson(), ps->poisson(), ps->shear(), ps->shear());
								
								//double cor = u.length();
 								double fsn = -c.kn * pow(cdist, 1.5);
								double fca = cohesionForce(pos[i].w, pos[k].w, ps->youngs(), ps->youngs(), ps->poisson(), ps->poisson(), fsn);
 								double fdn = c.vn * rv.dot(u);

								if ((fsn + fca + fdn < 0.0)){
									fdn = fdn;
								}
								Fn = (fsn + fca + fdn) * u;
								e = rv - rv.dot(u) * u;
								mag_e = e.length();
								VEC3D Ft;
								if (mag_e){
									sh = -(e / mag_e);
									ds = mag_e * dt;
									double fst = -c.ks * ds;
									double fdt = c.vs * rv.dot(sh);				
									Ft = (fst + fdt) * sh;
									if (Ft.length() >= c.mu * Fn.length())
										Ft = c.mu * fsn * sh;
									//Ft = min(c.ks * ds + c.vs * (rv.dot(sh)), c.mu * Fn.length()) * sh;
									M = (rcon * u).cross(Ft);
									if (omega->length()){
										VEC3D on = *omega / omega->length();
										M += c.rf * fsn * rcon * on;
									}
								}
								fr[i] += Fn + Ft;
								mm[i] += M;
							}
						}
					}
				}
			}
		}
	}
	return true;
}