#include "collision.h"
#include "object.h"
#include <cmath>

collision::collision()
	: coh(0)
	, tcm(HMCM)
	, gb(NULL)
	, dcp(NULL)
{

}

collision::collision(
	QString& _name, 
	modeler *_md,
	QString& o1,
	QString& o2, 
	tCollisionPair _tp,
	tContactModel _tcm)
	: name(_name)
	, md(_md)
	, oname1(o1)
	, oname2(o2)
	, coh(0)
	, tcm(_tcm)
	, gb(NULL)
	, tcp(_tp)
	, rfric(0)
	, fric(0)
	, dcp(NULL)
{

}

collision::collision(const collision& cs)
{

}

collision::~collision()
{
	if (dcp) checkCudaErrors(cudaFree(dcp));
}

void collision::setContactParameter(
	double Ei, double Ej, double Pi, double Pj, double Gi, double Gj,
	double _rest, double _fric, double _rfric, double _coh, double _ratio)
{
	hcp = contact_parameter{ Ei, Ej, Pi, Pj, Gi, Gj, _rest, _fric, _rfric, _coh, _ratio};
}

void collision::allocDeviceMemory()
{
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(contact_parameter)));
	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(contact_parameter), cudaMemcpyHostToDevice));
}

constant collision::getConstant(double ir, double jr, double im, double jm, double iE, double jE, double ip, double jp, double iG, double jG)
{
// 	particle_system* ps = md->particleSystem();
	constant c = { 0, 0, 0, 0, 0, 0 };
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	
	//double lne = log(rest);
	//double tk = (16.f / 15.f)*sqrt(er) * eym * pow((15.f * em * 1.0f) / (16.f * sqrt(er) * eym), 0.2f);
 	//double Geq = 1 / (((2 - ip) / si) + ((2 - jp) / sj));
	switch (tcm)
	{
	case HMCM:{
		double Geq = (iG * jG) / (iG*(2.0 - jp) + jG*(2.0 - ip));
		double ln_e = log(hcp.rest);
		double xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
		c.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		c.vn = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(c.kn * Meq);
		c.ks = 8.0 * Geq * sqrt(Req);
		c.vs = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(c.ks * Meq);
		c.mu = hcp.fric;
		c.rf = hcp.rfric;
		break;
	}
	case DHS:{
		double beta = (M_PI / log(hcp.rest));
		c.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		c.vn = sqrt((4.0*Meq * c.kn) / (1 + beta * beta));
		c.ks = c.kn * hcp.sratio;
		c.vs = c.vn * hcp.sratio; 
		c.mu = hcp.fric;
		c.rf = hcp.rfric;
		break;
	}
	}
// 	c.kn = (4.0f / 3.0f)*sqrt(er)*eym;
// 	c.vn = sqrt((4.0f*em * c.kn) / (1 + beta * beta));
// 	c.ks = c.kn * sratio;
// 	c.vs = c.vn * sratio;
// 	c.mu = fric;
 	return c;
}

double collision::cohesionForce(double ri, double rj, double Ei, double Ej, double pri, double prj, double Fn)
{
	double cf = 0.0;
	//cohesion = 5.f;
	if (hcp.coh){
		double req = (ri * rj / (ri + rj));
		double Eeq_inv = ((1 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej);
		double rcp = (3.0 * req * (-Fn)) / (4.0 * (1 / Eeq_inv));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = hcp.coh * Ac;
	}
	
	return cf;
}

void collision::save_collision_data(QTextStream& ts)
{
	ts << "COLLISION " << name << " " << hcp.rest << " " << hcp.fric << " " << hcp.rfric << " " << hcp.coh  << " " << hcp.sratio << " " << tcm << endl;
	ts << "i_object " << oname1 << endl
		<< "j_object " << oname2 << endl;
}

// bool collision::collid_p2p(double dt)
// {
// 	particle_system *ps = md->particleSystem();
// 	iE = jE = ps->youngs(); ip = jp = ps->poisson();
// 	VEC3I neigh, gp;
// 	double im, jm, ir, jr, dist, cdist, mag_e, ds;
// 	unsigned int hash, sid, eid;
// 	constant c;
// 	VEC3F ipos, jpos, ivel, jvel, iomega, jomega, f, m, rp, u, rv, sf, sm, e, sh, shf;
// 	for (unsigned int i = 0; i < md->numParticle(); i++){
// 		ipos = ps->position()[i];
// 		ivel = ps->velocity()[i];
// 		iomega = ps->angVelocity()[i];
// 		gp = grid_base::getCellNumber(ipos.x, ipos.y, ipos.z);
// 		im = ps->mass()[i];
// 		ir = ps->radius()[i];
// 		f = im * md->gravity();
// 		m = 0.0f;
// 		for (int z = -1; z <= 1; z++){
// 			for (int y = -1; y <= 1; y++){
// 				for (int x = -1; x <= 1; x++){
// 					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
// 					hash = grid_base::getHash(neigh);
// 					sid = grid_base::cellStart(hash);
// 					if (sid != 0xffffffff){
// 						eid = grid_base::cellEnd(hash);
// 						for (unsigned int j = sid; j < eid; j++){
// 							unsigned int k = grid_base::sortedID(j);
// 							if (i == k || k >= md->numParticle())
// 								continue;
// 							jm = ps->mass()[k];
// 							jr = ps->radius()[k];
// 							c = getConstant(ir, jr, im, jm);
// 							jpos = ps->position()[k];
// 							jvel = ps->velocity()[k];
// 							jomega = ps->angVelocity()[k];
// 							rp = jpos - ipos;
// 							dist = rp.length();
// 							cdist = (ir + jr) - dist;
// 							if (cdist > 0){
// 								u = rp / dist;
// 								rv = jvel + jomega.cross(jr * u) - (ivel + iomega.cross(ir * u));
// 								sf = (-c.kn * pow(cdist, 1.5f) + rv.dot(u) * c.vn) * u;
// 								e = rv - rv.dot(u) * u;
// 								mag_e = e.length();
// 								if (mag_e){
// 									sh = e / mag_e;
// 									ds = mag_e * dt;
// 									shf = min(c.ks * ds + c.vs * (rv.dot(sh)), c.mu * sf.length());
// 									sm = (ir * u).cross(shf);
// 								}
// 								ps->force()[i] += sf;
// 								ps->moment()[i] += sm;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return true;
// }

// bool collision::collid(double dt)
//{
//	if(!iobj && !jobj)
//		collid_p2p(dt);
//	if (iobj->objectType() == PLANE || jobj->objectType() == PLANE)
//		if (jobj == NULL){
//			collision_particle_plane(md->particleSystem(), dy
//		}
//
//	return true;
//}