#include "contact_particles_polygonObject.h"

contact_particles_polygonObject::contact_particles_polygonObject(
	QString _name, contactForce_type t, object* o1, object* o2)
	: contact(_name, t)
// 	, dpi(NULL)
// 	, hpi(NULL)
	, maxRadii(0)
	, p(NULL)
	, po(NULL)
{
	contact::iobj = o1;
	contact::jobj = o2;
	po = dynamic_cast<polygonObject*>((o1->ObjectType() == POLYGON_SHAPE ? o1 : o2));
	p = o1->ObjectType() != POLYGON_SHAPE ? o1 : o2;
// 	hpi = new host_polygon_info[po->numIndex()];
// 	for (unsigned int i = 0; i < po->numIndex(); i++)
// 	{
// 		host_polygon_info d;
// 		d.P = po->Vertex0(i);//vertice[indice[i].x];
// 		d.Q = po->Vertex1(i);
// 		d.R = po->Vertex2(i);
// 		d.V = d.Q - d.P;
// 		d.W = d.R - d.P;
// 		d.N = d.V.cross(d.W);
// 		d.N = d.N / d.N.length();
// 		hpi[i] = d;
// 	}
}

contact_particles_polygonObject::~contact_particles_polygonObject()
{
// 	if (hpi) delete[] hpi; hpi = NULL;
// 	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
	//if (hsphere) delete[] hsphere; hsphere = NULL;
}

// bool contact_particles_polygonObject::collision(
// 	double *dpos, double *dvel,
// 	double *domega, double *dmass, 
// 	double *dforce, double *dmoment, 
// 	unsigned int *sorted_id, unsigned int *cell_start, 
// 	unsigned int *cell_end, unsigned int np)
// {
// // 	simulation::isGpu()
// // 		? cu_particle_polygonObject_collision
// // 		(1, dpi, po->deviceSphereSet(), dpos, dvel, domega, dforce, dmoment, dmass
// // 		, sorted_id, cell_start, cell_end, dcp, np)
// // 		: hostCollision(
// // 		dpos, dvel, domega, dmass, dforce, dmoment, sorted_id, cell_start, cell_end, np);
// 	return true;
// }

void contact_particles_polygonObject::cudaMemoryAlloc()
{
// 	contact::cudaMemoryAlloc();
// 	if (!dpi)
// 	{
// 		checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_polygon_info) * nPolySphere));
// 		checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_polygon_info) * nPolySphere, cudaMemcpyHostToDevice));
// 	}
}

void contact_particles_polygonObject::insertContactParameters(unsigned int id, double r, double rt, double fr)
{
// 	contactParameters cp = { r, rt, fr };
// 	cps[id] = cp;
}

// void contact_particles_polygonObject::allocPolygonInformation(unsigned int _nPolySphere)
// {
// //	nPolySphere = _nPolySphere;
// 	//hsphere = new VEC4D[nPolySphere];
// //	hpi = new host_polygon_info[nPolySphere];
// }
// 
// void contact_particles_polygonObject::definePolygonInformation(
// 	unsigned int id, unsigned int bPolySphere, 
// 	unsigned int ePolySphere, double *vList, unsigned int *iList)
// {
// // 	unsigned int a, b, c;
// // 	maxRadii = 0;
// // 	for (unsigned int i = bPolySphere; i < bPolySphere + ePolySphere; i++)
// // 	{
// // 		a = iList[i * 3 + 0];
// // 		b = iList[i * 3 + 1];
// // 		c = iList[i * 3 + 2];
// // 		host_polygon_info po;
// // 		po.id = id;
// // 		po.P = VEC3D(vList[a * 3 + 0], vList[a * 3 + 1], vList[a * 3 + 2]);
// // 		po.Q = VEC3D(vList[b * 3 + 0], vList[b * 3 + 1], vList[b * 3 + 2]);
// // 		po.R = VEC3D(vList[c * 3 + 0], vList[c * 3 + 1], vList[c * 3 + 2]);
// // 		po.V = po.Q - po.P;
// // 		po.W = po.R - po.P;
// // 		po.N = po.V.cross(po.W);
// // 		po.N = po.N / po.N.length();
// // 		hpi[i] = po;
// // 		VEC3D M1 = (po.Q + po.P) / 2;
// // 		VEC3D M2 = (po.R + po.P) / 2;
// // 		VEC3D D1 = po.N.cross(po.V);
// // 		VEC3D D2 = po.N.cross(po.W);
// // 		double t = 0;
// // 		if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
// // 		{
// // 			t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
// // 		}
// // 		else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
// // 		{
// // 			t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
// // 		}
// // 		else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
// // 		{
// // 			t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
// // 		}
// // 		VEC3D Ctri = M1 + t * D1;
// // 		VEC4D sph;
// // 		sph.w = (Ctri - po.P).length();
// // 		sph.x = Ctri.x; sph.y = Ctri.y; sph.z = Ctri.z;
// // 		//com += Ctri;
// // 		// 		while (abs(fc - ft) > 0.00001)
// // 		// 		{
// // 		// 			d = ft * sph.w;
// // 		// 			double p = d / po.N.length();
// // 		// 			VEC3D _c = Ctri - p * po.N;
// // 		// 			sph.x = _c.x; sph.y = _c.y; sph.z = _c.z;
// // 		// 			sph.w = (_c - po.P).length();
// // 		// 			fc = d / sph.w;
// // 		// 		}
// // 		if (sph.w > maxRadii)
// // 			maxRadii = sph.w;
// // 		hsphere[i] = sph;
// // 	}
// //	com = com / ntriangle;
// }

// bool contact_particles_polygonObject::hostCollision(
// 	double *m_pos, double *m_vel, 
// 	double *m_omega, double *m_mass, 
// 	double *m_force, double *m_moment, 
// 	unsigned int *sorted_id, unsigned int *cell_start, 
// 	unsigned int *cell_end, unsigned int np)
// {
// // 	unsigned int _np = 0;
// // 	VEC3I neigh, gp;
// // 	double dist, cdist, mag_e, ds;
// // 	unsigned int hash, sid, eid;
// // 	contactParameters c;
// // 	VEC3D ipos, jpos, rp, u, rv, Fn, Ft, e, sh, M;
// // 	VEC4D *pos = (VEC4D*)m_pos;
// // 	VEC3D *vel = (VEC3D*)m_vel;
// // 	VEC3D *omega = (VEC3D*)m_omega;
// // 	VEC3D *fr = (VEC3D*)m_force;
// // 	VEC3D *mm = (VEC3D*)m_moment;
// // 	double* ms = m_mass;
// // 	double dt = simulation::ctime;
// // 	for (unsigned int i = 0; i < np; i++){
// // 		ipos = VEC3D(pos[i].x, pos[i].y, pos[i].z);
// // 		gp = grid_base::getCellNumber(pos[i].x, pos[i].y, pos[i].z);
// // 		for (int z = -1; z <= 1; z++){
// // 			for (int y = -1; y <= 1; y++){
// // 				for (int x = -1; x <= 1; x++){
// // 					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
// // 					hash = grid_base::getHash(neigh);
// // 					sid = cell_start[hash];
// // 					if (sid != 0xffffffff){
// // 						eid = cell_end[hash];
// // 						for (unsigned int j = sid; j < eid; j++){
// // 							unsigned int k = sorted_id[j];
// // 							if (i == k || k >= np)
// // 								continue;
// // 							jpos = VEC3D(pos[k].x, pos[k].y, pos[k].z);// toVector3();
// // 							rp = jpos - ipos;
// // 							dist = rp.length();
// // 							cdist = (pos[i].w + pos[k].w) - dist;
// // 							//double rcon = pos[i].w - cdist;
// // 							unsigned int rid = 0;
// // 							if (cdist > 0){
// // 								u = rp / dist;
// // 								VEC3D cp = ipos + pos[i].w * u;
// // 								//unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
// // 								//VEC3D c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
// // 								//double rcon = pos[i].w - 0.5 * cdist;
// // 								rv = vel[k] + omega[k].cross(-pos[k].w * u) - (vel[i] + omega[i].cross(pos[i].w * u));
// // 								c = getContactParameters(
// // 									pos[i].w, pos[k].w,
// // 									ms[i], ms[k],
// // 									mpp.Ei, mpp.Ej,
// // 									mpp.pri, mpp.prj,
// // 									mpp.Gi, mpp.Gj);
// // 								switch (f_type)
// // 								{
// // 								case DHS: DHSModel(c, cdist, cp, rv, u, Fn, Ft); break;
// // 								}
// // 
// // 								fr[i] += Fn/* + Ft*/;
// // 								mm[i] += M;
// // 							}
// // 						}
// // 					}
// // 				}
// // 			}
// // 		}
// // 	}
// 	return true;
// }
// 
// collision_particles_polygonObject::collision_particles_polygonObject()
// {
// 
// }
// 
// collision_particles_polygonObject::collision_particles_polygonObject(
// 	QString& _name, 
// 	modeler* _md, 
// 	particle_system *_ps, 
// 	polygonObject * _poly, 
// 	tContactModel _tcm)
// 	: collision(_name, _md, _ps->name(), _poly->objectName(), PARTICLES_POLYGONOBJECT, _tcm)
// 	, ps(_ps)
// 	, po(_poly)
// {
// 
// }
// 
// collision_particles_polygonObject::~collision_particles_polygonObject()
// {
// 
// }
// 
// bool collision_particles_polygonObject::collid(double dt)
// {
// 	return true;
// }
// 
// bool collision_particles_polygonObject::cuCollid(
// 	double *dpos, double *dvel,
// 	double *domega, double *dmass,
// 	double *dforce, double *dmoment, unsigned int np)
// {
// 	double3 *mforce;
// 	double3 *mmoment;
// 	double3 *mpos;
// 	VEC3D _mp;
// 	double3 _mf = make_double3(0.0, 0.0, 0.0);
// 	double3 _mm = make_double3(0.0, 0.0, 0.0);
// 	if (po->pointMass())
// 		_mp = po->pointMass()->Position();
// 	checkCudaErrors(cudaMalloc((void**)&mpos, sizeof(double3)));
// 	checkCudaErrors(cudaMemcpy(mpos, &_mp, sizeof(double3), cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMalloc((void**)&mforce, sizeof(double3)*ps->numParticle()));
// 	checkCudaErrors(cudaMalloc((void**)&mmoment, sizeof(double3)*ps->numParticle()));
// 	checkCudaErrors(cudaMemset(mforce, 0, sizeof(double3)*ps->numParticle()));
// 	checkCudaErrors(cudaMemset(mmoment, 0, sizeof(double3)*ps->numParticle()));
// 
// 	switch (tcm)
// 	{
// 	case HMCM: 
// 		cu_particle_polygonObject_collision(
// 			0, po->devicePolygonInfo(), po->deviceSphereSet(), po->deviceMassInfo(), 
// 			dpos, dvel, domega,
// 			dforce, dmoment, dmass,
// 			gb->cuSortedID(), gb->cuCellStart(), gb->cuCellEnd(), dcp, 
// 			ps->numParticle(), mpos, mforce, mmoment, _mf, _mm); 
// 		break;
// 	}
// 	
// 	_mf = reductionD3(mforce, ps->numParticle());
// 	if (po->pointMass()){
// 		po->pointMass()->addCollisionForce(VEC3D(_mf.x, _mf.y, _mf.z));
// 	}
// 	_mm = reductionD3(mmoment, ps->numParticle());
// 	if (po->pointMass()){
// 		po->pointMass()->addCollisionMoment(VEC3D(_mm.x, _mm.y, _mm.z));
// 	}
// 	checkCudaErrors(cudaFree(mforce)); mforce = NULL;
// 	checkCudaErrors(cudaFree(mmoment)); mmoment = NULL;
// 	checkCudaErrors(cudaFree(mpos)); mpos = NULL;
// 	return true;
// }
// 
// VEC3D collision_particles_polygonObject::particle_polygon_contact_detection(host_polygon_info& hpi, VEC3D& p, double pr)
// {
// 	VEC3D a = hpi.P.To<double>();
// 	VEC3D b = hpi.Q.To<double>();
// 	VEC3D c = hpi.R.To<double>();
// 	VEC3D ab = b - a;
// 	VEC3D ac = c - a;
// 	VEC3D ap = p - a;
// 
// 	double d1 = ab.dot(ap);// dot(ab, ap);
// 	double d2 = ac.dot(ap);// dot(ac, ap);
// 	if (d1 <= 0.0 && d2 <= 0.0){
// 		//	*wc = 0;
// 		return a;
// 	}
// 
// 	VEC3D bp = p - b;
// 	double d3 = ab.dot(bp);
// 	double d4 = ac.dot(bp);
// 	if (d3 >= 0.0 && d4 <= d3){
// 		//	*wc = 0;
// 		return b;
// 	}
// 	double vc = d1 * d4 - d3 * d2;
// 	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
// 		//	*wc = 1;
// 		double v = d1 / (d1 - d3);
// 		return a + v * ab;
// 	}
// 
// 	VEC3D cp = p - c;
// 	double d5 = ab.dot(cp);
// 	double d6 = ac.dot(cp);
// 	if (d6 >= 0.0 && d5 <= d6){
// 		//	*wc = 0;
// 		return c;
// 	}
// 
// 	double vb = d5 * d2 - d1 * d6;
// 	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
// 		//	*wc = 1;
// 		double w = d2 / (d2 - d6);
// 		return a + w * ac; // barycentric coordinates (1-w, 0, w)
// 	}
// 
// 	// Check if P in edge region of BC, if so return projection of P onto BC
// 	double va = d3 * d6 - d5 * d4;
// 	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
// 		//	*wc = 1;
// 		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
// 		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
// 	}
// 	//*wc = 2;
// 	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
// 	double denom = 1.0 / (va + vb + vc);
// 	double v = vb * denom;
// 	double w = vc * denom;
// 
// 	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
// 	//return 0.f;
// }
// 
// bool collision_particles_polygonObject::collid_with_particle(unsigned int i, double dt)
// {
// 	double overlap = 0.f;
// 	VEC4D ipos = ps->position()[i];
// 	VEC3D ivel = ps->velocity()[i];
// 	VEC3D iomega = ps->angVelocity()[i];
// 	double ir = ipos.w;
// 	VEC3D m_moment = 0.f;
// 	VEC3I neighbour_pos = 0;
// 	unsigned int grid_hash = 0;
// 	VEC3D single_force = 0.f;
// 	VEC3D shear_force = 0.f;
// 	VEC3I gridPos = gb->getCellNumber(ipos.x, ipos.y, ipos.z);
// 	unsigned int sindex = 0;
// 	unsigned int eindex = 0;
// 	VEC3D ip = VEC3D(ipos.x, ipos.y, ipos.z);
// 	double ms = ps->mass()[i];
// 	unsigned int np = md->numParticle();
// 	for (int z = -1; z <= 1; z++){
// 		for (int y = -1; y <= 1; y++){
// 			for (int x = -1; x <= 1; x++){
// 				neighbour_pos = VEC3I(gridPos.x + x, gridPos.y + y, gridPos.z + z);
// 				grid_hash = gb->getHash(neighbour_pos);
// 				sindex = gb->cellStart(grid_hash);
// 				if (sindex != 0xffffffff){
// 					eindex = gb->cellEnd(grid_hash);
// 					for (unsigned int j = sindex; j < eindex; j++){
// 						unsigned int k = gb->sortedID(j);
// 						if (k >= np)
// 						{
// 							k -= np;
// 							VEC3D cp = particle_polygon_contact_detection(po->hostPolygonInfo()[k], ip, ir);
// 							VEC3D distVec = ip - cp;
// 							double dist = distVec.length();
// 							overlap = ir - dist;
// 							if (overlap > 0)
// 							{
// 								VEC3D unit = -po->hostPolygonInfo()[k].N;
// 								VEC3D dv = -(ivel + iomega.cross(ir * unit));
// 								constant c = getConstant(ir, 0, ms, 0, ps->youngs(), po->youngs(), ps->poisson(), po->poisson(), ps->shear(), po->shear());
// 								double fsn = -c.kn * pow(overlap, 1.5);
// 								single_force = (fsn + c.vn * dv.dot(unit)) * unit;
// 								//std::cout << k << ", " << single_force.x << ", " << single_force.y << ", " << single_force.z << std::endl;
// 								VEC3D e = dv - dv.dot(unit) * unit;
// 								double mag_e = e.length();
// 								if (mag_e){
// 									VEC3D s_hat = e / mag_e;
// 									double ds = mag_e * dt;
// 									shear_force = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * single_force.length()) * s_hat;
// 									m_moment = (ir*unit).cross(shear_force);
// 								}
// 								ps->force()[i] += single_force;
// 								ps->moment()[i] += m_moment;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return true;
// }