#include "contact.h"

unsigned int contact::count = 0;

void contact::DHSModel
(
	contactParameters& c, double cdist, VEC3D& cp,
	VEC3D& dv, VEC3D& unit, VEC3D& F, VEC3D& M
)
{
	VEC3D Fn, Ft;
	double fsn = (-c.kn * pow(cdist, 1.5));
	//	double fca = cohesionForce(p.w, 0.0, ps->youngs(), 0.0, ps->poisson(), 0.0, fsn);
	double fsd = c.vn * dv.dot(unit);
	Fn = (fsn/* + fca*/ + c.vn * dv.dot(unit)) * unit;
	//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
	VEC3D e = dv - dv.dot(unit) * unit;
	double mag_e = e.length();
	//vector3<double> shf;
	if (mag_e){
		VEC3D s_hat = e / mag_e;
		double ds = mag_e * simulation::dt;
		double ft1 = c.ks * ds + c.vs * dv.dot(s_hat);
		double ft2 = friction * Fn.length();
		Ft = min(ft1, ft2) * s_hat;
		//M = (p.w * u).cross(Ft);
		M = cp.cross(Ft);
	}
	F = Fn + Ft;
}

contact::contact(QString nm, contactForce_type t)
	: name(nm)
	, restitution(0)
	, stiffnessRatio(0)
	, friction(0)
	, f_type(t)
	, type(NO_CONTACT_PAIR)
{
	count++;
	mpp = { 0, };
}

contact::contact(const contact* c)
	: name(c->Name())
	, restitution(c->Restitution())
	, friction(c->Friction())
	, stiffnessRatio(c->StiffnessRatio())
	, f_type(c->ForceMethod())
	, mpp(c->MaterialPropertyPair())
	, type(c->PairType())
{
	if (c->DeviceContactProperty())
		checkCudaErrors(cudaMemcpy(dcp, c->DeviceContactProperty(), sizeof(device_contact_property), cudaMemcpyDeviceToDevice));
}

contact::~contact()
{
	if (dcp) cudaFree(dcp); dcp = NULL;
}

void contact::setContactParameters(double r, double rt, double f)
{
	restitution = r;
	stiffnessRatio = rt;
	friction = f;
}

void contact::cudaMemoryAlloc()
{
	device_contact_property hcp = device_contact_property
	{
		mpp.Ei, mpp.Ej, mpp.pri, mpp.prj, mpp.Gi, mpp.Gj,
		restitution, friction, 0.0, 0.0, stiffnessRatio
	};
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(device_contact_property), cudaMemcpyHostToDevice));
}

contact::contactParameters contact::getContactParameters
(
	double ir, double jr,
	double im, double jm,
	double iE, double jE,
	double ip, double jp,
	double is, double js)
{

	// 	particle_system* ps = md->particleSystem();
	contactParameters cp;
 	double Meq = jm ? (im * jm) / (im + jm) : im;
 	double Req = jr ? (ir * jr) / (ir + jr) : ir;
 	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
 	double lne = log(restitution);
	double beta = 0.0;
// 	//double tk = (16.f / 15.f)*sqrt(er) * eym * pow((15.f * em * 1.0f) / (16.f * sqrt(er) * eym), 0.2f);
// 	//double Geq = 1 / (((2 - ip) / si) + ((2 - jp) / sj));
	switch (f_type)
	{
	case DHS:
		beta = (M_PI / log(restitution));
		cp.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		cp.vn = sqrt((4.0*Meq * cp.kn) / (1 + beta * beta));
		cp.ks = cp.kn * stiffnessRatio;
		cp.vs = cp.vn * stiffnessRatio;
		//c.mu = hcp.fric;
		//c.rf = hcp.rfric;
		break;
	}
	// 	c.kn = (4.0f / 3.0f)*sqrt(er)*eym;
	// 	c.vn = sqrt((4.0f*em * c.kn) / (1 + beta * beta));
	// 	c.ks = c.kn * sratio;
	// 	c.vs = c.vn * sratio;
	// 	c.mu = fric;
	return cp;
}

contact::pairType contact::getContactPair(geometry_type t1, geometry_type t2)
{
	return static_cast<pairType>(t1 + t2);
}

// #include "collision.h"
// #include "object.h"
// #include <cmath>
// 
// collision::collision()
// 	: coh(0)
// 	, tcm(HMCM)
// 	, gb(NULL)
// 	, dcp(NULL)
// {
// 
// }
// 
// collision::collision(
// 	QString& _name, 
// 	modeler *_md,
// 	QString& o1,
// 	QString& o2, 
// 	tCollisionPair _tp,
// 	tContactModel _tcm)
// 	: name(_name)
// 	, md(_md)
// 	, oname1(o1)
// 	, oname2(o2)
// 	, coh(0)
// 	, tcm(_tcm)
// 	, gb(NULL)
// 	, tcp(_tp)
// 	, rfric(0)
// 	, fric(0)
// 	, dcp(NULL)
// {
// 
// }
// 
// collision::collision(const collision& cs)
// {
// 
// }
// 
// collision::~collision()
// {
// 	if (dcp) checkCudaErrors(cudaFree(dcp));
// }
// 
// 
// void collision::allocDeviceMemory()
// {
// 	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(contact_parameter)));
// 	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(contact_parameter), cudaMemcpyHostToDevice));
// }
// 
// constant collision::getConstant(double ir, double jr, double im, double jm, double iE, double jE, double ip, double jp, double iG, double jG)
// {
// // 	particle_system* ps = md->particleSystem();
// 	constant c = { 0, 0, 0, 0, 0, 0 };
// 	double Meq = jm ? (im * jm) / (im + jm) : im;
// 	double Req = jr ? (ir * jr) / (ir + jr) : ir;
// 	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
// 	
// 	//double lne = log(rest);
// 	//double tk = (16.f / 15.f)*sqrt(er) * eym * pow((15.f * em * 1.0f) / (16.f * sqrt(er) * eym), 0.2f);
//  	//double Geq = 1 / (((2 - ip) / si) + ((2 - jp) / sj));
// 	switch (tcm)
// 	{
// 	case HMCM:{
// 		double Geq = (iG * jG) / (iG*(2.0 - jp) + jG*(2.0 - ip));
// 		double ln_e = log(hcp.rest);
// 		double xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
// 		c.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
// 		c.vn = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(c.kn * Meq);
// 		c.ks = 8.0 * Geq * sqrt(Req);
// 		c.vs = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(c.ks * Meq);
// 		c.mu = hcp.fric;
// 		c.rf = hcp.rfric;
// 		break;
// 	}
// 	case DHS:{
// 		double beta = (M_PI / log(hcp.rest));
// 		c.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
// 		c.vn = sqrt((4.0*Meq * c.kn) / (1 + beta * beta));
// 		c.ks = c.kn * hcp.sratio;
// 		c.vs = c.vn * hcp.sratio; 
// 		c.mu = hcp.fric;
// 		c.rf = hcp.rfric;
// 		break;
// 	}
// 	}
// // 	c.kn = (4.0f / 3.0f)*sqrt(er)*eym;
// // 	c.vn = sqrt((4.0f*em * c.kn) / (1 + beta * beta));
// // 	c.ks = c.kn * sratio;
// // 	c.vs = c.vn * sratio;
// // 	c.mu = fric;
//  	return c;
// }
// 
// double collision::cohesionForce(double ri, double rj, double Ei, double Ej, double pri, double prj, double Fn)
// {
// 	double cf = 0.0;
// 	//cohesion = 5.f;
// 	if (hcp.coh){
// 		double req = (ri * rj / (ri + rj));
// 		double Eeq_inv = ((1 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej);
// 		double rcp = (3.0 * req * (-Fn)) / (4.0 * (1 / Eeq_inv));
// 		double rc = pow(rcp, 1.0 / 3.0);
// 		double Ac = M_PI * rc * rc;
// 		cf = hcp.coh * Ac;
// 	}
// 	
// 	return cf;
// }
// 
// void collision::save_collision_data(QTextStream& ts)
// {
// 	ts << "COLLISION " << name << " " << hcp.rest << " " << hcp.fric << " " << hcp.rfric << " " << hcp.coh  << " " << hcp.sratio << " " << tcm << endl;
// 	ts << "i_object " << oname1 << endl
// 		<< "j_object " << oname2 << endl;
// }
// 
// // bool collision::collid_p2p(double dt)
// // {
// // 	particle_system *ps = md->particleSystem();
// // 	iE = jE = ps->youngs(); ip = jp = ps->poisson();
// // 	VEC3I neigh, gp;
// // 	double im, jm, ir, jr, dist, cdist, mag_e, ds;
// // 	unsigned int hash, sid, eid;
// // 	constant c;
// // 	VEC3F ipos, jpos, ivel, jvel, iomega, jomega, f, m, rp, u, rv, sf, sm, e, sh, shf;
// // 	for (unsigned int i = 0; i < md->numParticle(); i++){
// // 		ipos = ps->position()[i];
// // 		ivel = ps->velocity()[i];
// // 		iomega = ps->angVelocity()[i];
// // 		gp = grid_base::getCellNumber(ipos.x, ipos.y, ipos.z);
// // 		im = ps->mass()[i];
// // 		ir = ps->radius()[i];
// // 		f = im * md->gravity();
// // 		m = 0.0f;
// // 		for (int z = -1; z <= 1; z++){
// // 			for (int y = -1; y <= 1; y++){
// // 				for (int x = -1; x <= 1; x++){
// // 					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
// // 					hash = grid_base::getHash(neigh);
// // 					sid = grid_base::cellStart(hash);
// // 					if (sid != 0xffffffff){
// // 						eid = grid_base::cellEnd(hash);
// // 						for (unsigned int j = sid; j < eid; j++){
// // 							unsigned int k = grid_base::sortedID(j);
// // 							if (i == k || k >= md->numParticle())
// // 								continue;
// // 							jm = ps->mass()[k];
// // 							jr = ps->radius()[k];
// // 							c = getConstant(ir, jr, im, jm);
// // 							jpos = ps->position()[k];
// // 							jvel = ps->velocity()[k];
// // 							jomega = ps->angVelocity()[k];
// // 							rp = jpos - ipos;
// // 							dist = rp.length();
// // 							cdist = (ir + jr) - dist;
// // 							if (cdist > 0){
// // 								u = rp / dist;
// // 								rv = jvel + jomega.cross(jr * u) - (ivel + iomega.cross(ir * u));
// // 								sf = (-c.kn * pow(cdist, 1.5f) + rv.dot(u) * c.vn) * u;
// // 								e = rv - rv.dot(u) * u;
// // 								mag_e = e.length();
// // 								if (mag_e){
// // 									sh = e / mag_e;
// // 									ds = mag_e * dt;
// // 									shf = min(c.ks * ds + c.vs * (rv.dot(sh)), c.mu * sf.length());
// // 									sm = (ir * u).cross(shf);
// // 								}
// // 								ps->force()[i] += sf;
// // 								ps->moment()[i] += sm;
// // 							}
// // 						}
// // 					}
// // 				}
// // 			}
// // 		}
// // 	}
// // 	return true;
// // }
// 
// // bool collision::collid(double dt)
// //{
// //	if(!iobj && !jobj)
// //		collid_p2p(dt);
// //	if (iobj->objectType() == PLANE || jobj->objectType() == PLANE)
// //		if (jobj == NULL){
// //			collision_particle_plane(md->particleSystem(), dy
// //		}
// //
// //	return true;
// //}