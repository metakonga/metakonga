#include "contact_particles_polygonObjects.h"
#include "contact_particles_polygonObject.h"
#include "polygonObject.h"

contact_particles_polygonObjects::contact_particles_polygonObjects()
	: contact("particles_polygonObjects", DHS)
	, hsphere(NULL)
	, dsphere(NULL)
	, hpi(NULL)
	, dpi(NULL)
	, hcp(NULL)
	, nPobjs(NULL)
	, pct(NULL)
	, maxRadius(0)
	, npolySphere(0)
	, ncontact(0)
{

}

contact_particles_polygonObjects::~contact_particles_polygonObjects()
{
	if (hsphere) delete[] hsphere; hsphere = NULL;
	if (hpi) delete[] hpi; hpi = NULL;
	if (hcp) delete[] hcp; hcp = NULL;
	if (pct) delete[] pct; pct = NULL;
	checkCudaErrors(cudaFree(dsphere)); dsphere = NULL;
	checkCudaErrors(cudaFree(dpi)); dpi = NULL;
}

unsigned int contact_particles_polygonObjects::define(
	QMap<QString, contact_particles_polygonObject*>& cppos)
{
	foreach(contact_particles_polygonObject* cppo, cppos)
	{
		polygonObject* pobj = cppo->PolygonObject();
		npolySphere += pobj->NumTriangle();
	}
	nPobjs = cppos.size();
	if (simulation::isCpu())
	{
		hsphere = new VEC4D[npolySphere];
		hpi = new host_polygon_info[npolySphere];
		hcp = new contact_parameter[nPobjs];
		mpp = new material_property_pair[nPobjs];
	}
	if (!pct)
		pct = new polygonContactType[nPobjs];
// 	else
// 	{
// 		checkCudaErrors(cudaMalloc((void**)&dsphere, sizeof(double) * n * 4));
// 		checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_polygon_info) * n));
// 		checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(contact_parameter) * nPobjs));
// 	}
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	double maxRadii = 0.0;
	unsigned int idx = 0;
	foreach(contact_particles_polygonObject* cppo, cppos)
	{
		contact_parameter cp = { 0, };
		material_property_pair mp = { 0, };
		polygonObject* pobj = cppo->PolygonObject();
		double *vList = pobj->VertexList();
		unsigned int *iList = pobj->IndexList();
		//unsigned int id = pobj->ID();
		unsigned int a, b, c;
		maxRadii = 0;
		cp.rest = cppo->Restitution();
		cp.sratio = cppo->StiffnessRatio();
		cp.fric = cppo->Friction();
		mp = *(cppo->MaterialPropertyPair());
		hcp[idx] = cp;
		mpp[idx] = mp;
		pair_ip[idx] = pobj;
		ePolySphere += pobj->NumTriangle();
		for (unsigned int i = bPolySphere; i < ePolySphere; i++)
		{
			host_polygon_info po;
			po.id = idx;
			if (iList)
			{
				a = iList[i * 3 + 0];
				b = iList[i * 3 + 1];
				c = iList[i * 3 + 2];
				po.P = VEC3D(vList[a * 3 + 0], vList[a * 3 + 1], vList[a * 3 + 2]);
				po.Q = VEC3D(vList[b * 3 + 0], vList[b * 3 + 1], vList[b * 3 + 2]);
				po.R = VEC3D(vList[c * 3 + 0], vList[c * 3 + 1], vList[c * 3 + 2]);
			}
			else
			{
				po.P = VEC3D(vList[i * 9 + 0], vList[i * 9 + 1], vList[i * 9 + 2]);
				po.Q = VEC3D(vList[i * 9 + 3], vList[i * 9 + 4], vList[i * 9 + 5]);
				po.R = VEC3D(vList[i * 9 + 6], vList[i * 9 + 7], vList[i * 9 + 8]);
			}
			po.V = po.Q - po.P;
			po.W = po.R - po.P;
			po.N = po.V.cross(po.W);
			po.N = po.N / po.N.length();
			hpi[i] = po;
			VEC3D M1 = (po.Q + po.P) / 2;
			VEC3D M2 = (po.R + po.P) / 2;
			VEC3D D1 = po.N.cross(po.V);
			VEC3D D2 = po.N.cross(po.W);
			double t = 0;
			if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
			{
				t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
			}
			else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
			{
				t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
			}
			else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
			{
				t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
			}
			VEC3D Ctri = M1 + t * D1;
			VEC4D sph;
			sph.w = (Ctri - po.P).length();
			sph.x = Ctri.x; sph.y = Ctri.y; sph.z = Ctri.z;
			//com += Ctri;
			// 		while (abs(fc - ft) > 0.00001)
			// 		{
			// 			d = ft * sph.w;
			// 			double p = d / po.N.length();
			// 			VEC3D _c = Ctri - p * po.N;
			// 			sph.x = _c.x; sph.y = _c.y; sph.z = _c.z;
			// 			sph.w = (_c - po.P).length();
			// 			fc = d / sph.w;
			// 		}
			if (sph.w > maxRadii)
				maxRadii = sph.w;
			hsphere[i] = sph;
		}
//		com = com / ntriangle;
		bPolySphere += pobj->NumTriangle();
	}
	maxRadius = maxRadii;
	return npolySphere;
}

bool contact_particles_polygonObjects::cppolyCollision(
	unsigned int idx, double r, double m,
	VEC3D& p, VEC3D& v, VEC3D& o, VEC3D& F, VEC3D& M)
{
	unsigned int ct = 0;
	unsigned int nc = 0;
	unsigned int i = hpi[idx].id;
// 	if (pct[i])
// 		return false;
	pointMass* pm = pair_ip[i];
	material_property_pair mat = mpp[i];
	contact_parameter cpa = hcp[i];
	polygonContactType _pct;
	VEC3D u;
	restitution = cpa.rest;
	friction = cpa.fric;
	stiffnessRatio = cpa.sratio;
	VEC3D mp = pm->getPosition();
	VEC3D mv = pm->getVelocity();
	VEC3D mo = pm->getAngularVelocity();
	VEC3D cpt = particle_polygon_contact_detection(hpi[idx], p, r, _pct);
	VEC3D po2cp = cpt - mp;
	VEC3D distVec = p - cpt;
	double dist = distVec.length();
	double cdist = r - dist;
	VEC3D m_f, m_m;
	if (cdist > 0)
	{
		ncontact++;
		if (_pct == FACE)
			u = -(hpi[idx].Q - hpi[idx].P).cross(hpi[idx].R - hpi[idx].P);
		else
			u = -distVec;
		u = u / u.length();
		double rcon = r - 0.5 * cdist;
		VEC3D cp = rcon * u;
		VEC3D dv = mv + mo.cross(po2cp) - (v + o.cross(r * u));
		contactParameters c = getContactParameters(
			r, 0.0,
			m, 0.0,
			mat.Ei, mat.Ej,
			mat.pri, mat.prj,
			mat.Gi, mat.Gj);
		switch (f_type)
		{
		case DHS: DHSModel(c, cdist, cp, dv, u, m_f, m_m); break;
		}

		F += m_f;
		M += m_m;
		pm->addCollisionForce(-m_f);
		VEC3D m_mm = -po2cp.cross(m_f);
		pm->addCollisionMoment(m_mm);
		pct[i] = _pct;
		return true;
	}
	return false;
}

// bool contact_particles_polygonObjects::collision(
// 	double *dpos, double *dvel, double *domega, double *dmass, double *dforce, double *dmoment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
// {
// 	simulation::isGpu()
// 		? cu_particle_polygonObject_collision
// 		(1, dpi, dsphere, dpos, dvel, domega, dforce, dmoment, dmass
// 		, sorted_id, cell_start, cell_end, dcp, np)
// 		: hostCollision(
// 		dpos, dvel, domega, dmass, dforce, dmoment, sorted_id, cell_start, cell_end, np);
// 	return true;
// }

void contact_particles_polygonObjects::cudaMemoryAlloc()
{
	device_contact_property *hcp = new device_contact_property[nPobjs];
	for (unsigned int i = 0; i < nPobjs; i++)
	{
		hcp[i] = { hcp[i].Ei, hcp[i].Ej, hcp[i].pri, hcp[i].prj, hcp[i].Gi, hcp[i].Gj,
			restitution, friction, 0.0, 0.0, stiffnessRatio };
	}
	checkCudaErrors(cudaMalloc((void**)&dsphere, sizeof(double) * npolySphere * 4));
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_polygon_info) * npolySphere));
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property) * nPobjs));
	checkCudaErrors(cudaMemcpy(dsphere, hsphere, sizeof(double) * npolySphere * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_polygon_info) * npolySphere, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp, hcp, sizeof(device_contact_property) * nPobjs, cudaMemcpyHostToDevice));
	delete[] hcp;
}

VEC3D contact_particles_polygonObjects::particle_polygon_contact_detection(
	host_polygon_info& hpi, VEC3D& p, double r, polygonContactType &_pct)
{
	VEC3D a = hpi.P;
	VEC3D b = hpi.Q;
	VEC3D c = hpi.R;
	VEC3D ab = b - a;
	VEC3D ac = c - a;
	VEC3D ap = p - a;

	double d1 = ab.dot(ap);
	double d2 = ac.dot(ap);
	if (d1 <= 0.0 && d2 <= 0.0){
		//	*wc = 0;
		_pct = VERTEX;
		return a;
	}

	VEC3D bp = p - b;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if (d3 >= 0.0 && d4 <= d3){
		//	*wc = 0;
		_pct = VERTEX;
		return b;
	}
	double vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
		//	*wc = 1;
		double v = d1 / (d1 - d3);
		_pct = EDGE;
		return a + v * ab;
	}

	VEC3D cp = p - c;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if (d6 >= 0.0 && d5 <= d6){
		//	*wc = 0;
		_pct = VERTEX;
		return c;
	}

	double vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
		//	*wc = 1;
		double w = d2 / (d2 - d6);
		_pct = EDGE;
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
		//	*wc = 1;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		_pct = EDGE;
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	//*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;
	_pct = FACE;
	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

// bool contact_particles_polygonObjects::hostCollision(
// 	double *m_pos, double *m_vel, 
// 	double *m_omega, double *m_mass, 
// 	double *m_force, double *m_moment, 
// 	unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, 
// 	unsigned int np)
// {
// 	unsigned int _np = 0;
// 	VEC3I neigh, gp;
// 	double dist, cdist, mag_e, ds;
// 	unsigned int hash, sid, eid;
// 	contactParameters c;
// 	VEC3D ipos, jpos, rp, u, rv, Fn, Ft, e, sh, M;
// 	VEC4D *pos = (VEC4D*)m_pos;
// 	VEC3D *vel = (VEC3D*)m_vel;
// 	VEC3D *omega = (VEC3D*)m_omega;
// 	VEC3D *fr = (VEC3D*)m_force;
// 	VEC3D *mm = (VEC3D*)m_moment;
// 	double* ms = m_mass;
// 	double dt = simulation::ctime;
// 	for (unsigned int i = 0; i < np; i++){
// 		ipos = VEC3D(pos[i].x, pos[i].y, pos[i].z);
// 		gp = grid_base::getCellNumber(pos[i].x, pos[i].y, pos[i].z);
// 		for (int z = -1; z <= 1; z++){
// 			for (int y = -1; y <= 1; y++){
// 				for (int x = -1; x <= 1; x++){
// 					neigh = VEC3I(gp.x + x, gp.y + y, gp.z + z);
// 					hash = grid_base::getHash(neigh);
// 					sid = cell_start[hash];
// 					if (sid != 0xffffffff){
// 						eid = cell_end[hash];
// 						for (unsigned int j = sid; j < eid; j++){
// 							unsigned int k = sorted_id[j];
// 							if (k <= np)
// 								continue;
// 							jpos = VEC3D(pos[k].x, pos[k].y, pos[k].z);// toVector3();
// 							rp = jpos - ipos;
// 							dist = rp.length();
// 							cdist = (pos[i].w + pos[k].w) - dist;
// 							//double rcon = pos[i].w - cdist;
// 							unsigned int rid = 0;
// 							if (cdist > 0){
// 								u = rp / dist;
// 								VEC3D cp = ipos + pos[i].w * u;
// 								//unsigned int ci = (unsigned int)(i / particle_cluster::perCluster());
// 								//VEC3D c2p = cp - ps->getParticleClusterFromParticleID(ci)->center();
// 								//double rcon = pos[i].w - 0.5 * cdist;
// 								rv = vel[k] + omega[k].cross(-pos[k].w * u) - (vel[i] + omega[i].cross(pos[i].w * u));
// 								c = getContactParameters(
// 									pos[i].w, pos[k].w,
// 									ms[i], ms[k],
// 									mpp->Ei, mpp->Ej,
// 									mpp->pri, mpp->prj,
// 									mpp->Gi, mpp->Gj);
// 								switch (f_type)
// 								{
// 								case DHS: DHSModel(c, cdist, cp, rv, u, Fn, Ft); break;
// 								}
// 
// 								fr[i] += Fn/* + Ft*/;
// 								mm[i] += M;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return true;
// }

