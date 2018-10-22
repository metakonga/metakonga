#include "contact_particles_polygonObjects.h"
#include "contact_particles_polygonObject.h"
#include "polygonObject.h"
#include "numeric_utility.h"

contact_particles_polygonObjects::contact_particles_polygonObjects()
	: contact("particles_polygonObjects", DHS)
	, hsphere(NULL)
	, dsphere(NULL)
	, hsphere_f(NULL)
	, dsphere_f(NULL)
	, hpi(NULL)
	, hpi_f(NULL)
	, dpi(NULL)
	, dpi_f(NULL)
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
	if (hpi_f) delete[] hpi_f; hpi = NULL;
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
	hsphere = new VEC4D[npolySphere];
	hpi = new host_polygon_info[npolySphere];
	hcp = new contact_parameter[nPobjs];
	mpp = new material_property_pair[nPobjs];
	if (simulation::isCpu())
		dsphere = (double*)hsphere;
	if (!pct)
		pct = new polygonContactType[nPobjs];

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
		//maxRadii = 0;
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
				po.P = pobj->Position() + pobj->toGlobal(VEC3D(vList[i * 9 + 0], vList[i * 9 + 1], vList[i * 9 + 2]));
				po.Q = pobj->Position() + pobj->toGlobal(VEC3D(vList[i * 9 + 3], vList[i * 9 + 4], vList[i * 9 + 5]));
				po.R = pobj->Position() + pobj->toGlobal(VEC3D(vList[i * 9 + 6], vList[i * 9 + 7], vList[i * 9 + 8]));
			}
			VEC3D ctri = numeric::utility::calculate_center_of_triangle(po.P, po.Q, po.R);
			double rad = (ctri - po.P).length();
			if (rad > maxRadii)
				maxRadii = rad;
			hsphere[i] = VEC4D(ctri.x, ctri.y, ctri.z, rad);
			hpi[i] = po;
		}
		bPolySphere += pobj->NumTriangle();
		idx++;
	}
	maxRadius = maxRadii;
	return npolySphere;
}

unsigned int contact_particles_polygonObjects::define_f(
	QMap<QString, contact_particles_polygonObject*>& cppos)
{
	foreach(contact_particles_polygonObject* cppo, cppos)
	{
		polygonObject* pobj = cppo->PolygonObject();
		npolySphere += pobj->NumTriangle();
	}
	nPobjs = cppos.size();
	hsphere_f = new VEC4F[npolySphere];
	hpi_f = new host_polygon_info_f[npolySphere];
	hcp = new contact_parameter[nPobjs];
	mpp = new material_property_pair[nPobjs];
	if (simulation::isCpu())
		dsphere_f = (float*)hsphere_f;
	if (!pct)
		pct = new polygonContactType[nPobjs];

	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	float maxRadii = 0.0f;
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
		//maxRadii = 0.f;
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
			host_polygon_info_f po;
			po.id = idx;
			if (iList)
			{
				a = iList[i * 3 + 0];
				b = iList[i * 3 + 1];
				c = iList[i * 3 + 2];
				po.P = VEC3F(vList[a * 3 + 0], vList[a * 3 + 1], vList[a * 3 + 2]);
				po.Q = VEC3F(vList[b * 3 + 0], vList[b * 3 + 1], vList[b * 3 + 2]);
				po.R = VEC3F(vList[c * 3 + 0], vList[c * 3 + 1], vList[c * 3 + 2]);
			}
			else
			{
				int s = i * 9;
				VEC3D P = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 0], vList[s + 1], vList[s + 2]));
				VEC3D Q = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 3], vList[s + 4], vList[s + 5]));
				VEC3D R = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 6], vList[s + 7], vList[s + 8]));
				po.P = P.To<float>();
				po.Q = Q.To<float>();
				po.R = R.To<float>();
			}
			VEC3F ctri = numeric::utility::calculate_center_of_triangle_f(po.P, po.Q, po.R);
			float rad = (ctri - po.P).length();
			if (rad > maxRadii)
				maxRadii = rad;
			hsphere_f[i] = VEC4F(ctri.x, ctri.y, ctri.z, rad);
			hpi_f[i] = po;
		}
		bPolySphere += pobj->NumTriangle();
		idx++;
	}
	maxRadius_f = maxRadii;
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
	VEC3D mp = pm->Position();
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
		u = -(hpi[idx].Q - hpi[idx].P).cross(hpi[idx].R - hpi[idx].P);
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
		//VEC3D m_mm = -po2cp.cross(m_f);
		//pm->addCollisionMoment(m_mm);
		pct[i] = _pct;
		return true;
	}
	return false;
}

void contact_particles_polygonObjects::updatePolygonObjectData()
{
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	foreach(polygonObject* pobj, pair_ip)
	{
		double *vList = pobj->VertexList();
		unsigned int *iList = pobj->IndexList();
		ePolySphere += pobj->NumTriangle();
		for (unsigned int i = bPolySphere; i < ePolySphere; i++)
		{
			host_polygon_info po;
			unsigned int a, b, c;
			po.id = hpi[i].id;
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
				int s = i * 9;
				po.P = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 0], vList[s + 1], vList[s + 2]));
				po.Q = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 3], vList[s + 4], vList[s + 5]));
				po.R = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 6], vList[s + 7], vList[s + 8]));
			}
			VEC3D ctri = numeric::utility::calculate_center_of_triangle(po.P, po.Q, po.R);
			hsphere[i].x = ctri.x;
			hsphere[i].y = ctri.y;
			hsphere[i].z = ctri.z;
			hpi[i] = po;
		}
		bPolySphere += pobj->NumTriangle();
	}
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(dsphere, hsphere, sizeof(double) * npolySphere * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_polygon_info) * npolySphere, cudaMemcpyHostToDevice));
	}
}

void contact_particles_polygonObjects::updatePolygonObjectData_f()
{
	unsigned int bPolySphere = 0;
	unsigned int ePolySphere = 0;
	foreach(polygonObject* pobj, pair_ip)
	{
		double *vList = pobj->VertexList();
		unsigned int *iList = pobj->IndexList();
		ePolySphere += pobj->NumTriangle();
		for (unsigned int i = bPolySphere; i < ePolySphere; i++)
		{
			host_polygon_info_f po;
			unsigned int a, b, c;
			po.id = hpi_f[i].id;
			if (iList)
			{
				a = iList[i * 3 + 0];
				b = iList[i * 3 + 1];
				c = iList[i * 3 + 2];
				po.P = VEC3F(vList[a * 3 + 0], vList[a * 3 + 1], vList[a * 3 + 2]);
				po.Q = VEC3F(vList[b * 3 + 0], vList[b * 3 + 1], vList[b * 3 + 2]);
				po.R = VEC3F(vList[c * 3 + 0], vList[c * 3 + 1], vList[c * 3 + 2]);
			}
			else
			{
				int s = i * 9;
				VEC3D P = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 0], vList[s + 1], vList[s + 2]));
				VEC3D Q = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 3], vList[s + 4], vList[s + 5]));
				VEC3D R = pobj->Position() + pobj->toGlobal(VEC3D(vList[s + 6], vList[s + 7], vList[s + 8]));
				po.P = P.To<float>();
				po.Q = Q.To<float>();
				po.R = R.To<float>();
			}
			VEC3F ctri = numeric::utility::calculate_center_of_triangle_f(po.P, po.Q, po.R);
			hsphere_f[i].x = ctri.x;
			hsphere_f[i].y = ctri.y;
			hsphere_f[i].z = ctri.z;
			hpi_f[i] = po;
		}
		bPolySphere += pobj->NumTriangle();
	}
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMemcpy(dsphere_f, hsphere_f, sizeof(float) * npolySphere * 4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dpi_f, hpi_f, sizeof(device_polygon_info_f) * npolySphere, cudaMemcpyHostToDevice));
	}
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
	device_contact_property *_hcp = new device_contact_property[nPobjs];
	for (unsigned int i = 0; i < nPobjs; i++)
	{
		_hcp[i] = { mpp[i].Ei, mpp[i].Ej, mpp[i].pri, mpp[i].prj, mpp[i].Gi, mpp[i].Gj,
			hcp[i].rest, hcp[i].fric, 0.0, 0.0, hcp[i].sratio };
	}
	checkCudaErrors(cudaMalloc((void**)&dsphere, sizeof(double) * npolySphere * 4));
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_polygon_info) * npolySphere));
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property) * nPobjs));
	checkCudaErrors(cudaMemcpy(dsphere, hsphere, sizeof(double) * npolySphere * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dpi, hpi, sizeof(device_polygon_info) * npolySphere, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp, _hcp, sizeof(device_contact_property) * nPobjs, cudaMemcpyHostToDevice));
	delete[] _hcp;
}

void contact_particles_polygonObjects::cudaMemoryAlloc_f()
{
	device_contact_property_f *_hcp = new device_contact_property_f[nPobjs];
	for (unsigned int i = 0; i < nPobjs; i++)
	{
		_hcp[i] = 
		{ (float)mpp[i].Ei, (float)mpp[i].Ej, (float)mpp[i].pri, (float)mpp[i].prj, (float)mpp[i].Gi, (float)mpp[i].Gj,
		(float)hcp[i].rest, (float)hcp[i].fric, 0.0f, 0.0f, (float)hcp[i].sratio };
	}
	checkCudaErrors(cudaMalloc((void**)&dsphere_f, sizeof(float) * npolySphere * 4));
	checkCudaErrors(cudaMalloc((void**)&dpi_f, sizeof(device_polygon_info_f) * npolySphere));
	checkCudaErrors(cudaMalloc((void**)&dcp_f, sizeof(device_contact_property_f) * nPobjs));
	checkCudaErrors(cudaMemcpy(dsphere_f, hsphere_f, sizeof(float) * npolySphere * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dpi_f, hpi_f, sizeof(device_polygon_info_f) * npolySphere, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dcp_f, _hcp, sizeof(device_contact_property_f) * nPobjs, cudaMemcpyHostToDevice));
	delete[] _hcp;
}

void contact_particles_polygonObjects::cuda_collision(
	double *pos, double *vel, double *omega, 
	double *mass, double *force, double *moment, 
	unsigned int *sorted_id, unsigned int *cell_start, 
	unsigned int *cell_end, unsigned int np)
{
	device_polygon_mass_info* hpmi = new device_polygon_mass_info[nPobjs];
	device_polygon_mass_info* dpmi = NULL;
	checkCudaErrors(cudaMalloc((void**)&dpmi, sizeof(device_polygon_mass_info) * nPobjs));
	QMapIterator<unsigned int, polygonObject*> po(pair_ip);
	while (po.hasNext())
	{
		po.next();
		unsigned int id = po.key();
		polygonObject* p = po.value();
		VEC3D pos = p->Position();
		VEC3D vel = p->getVelocity();
		VEC3D omega = p->getAngularVelocity();
		EPD ep = p->getEP();
		hpmi[id] =
		{
			make_double3(pos.x, pos.y, pos.z),
			make_double4(ep.e0, ep.e1, ep.e2, ep.e3),
			make_double3(vel.x, vel.y, vel.z),
			make_double3(omega.x, omega.y, omega.z),
			make_double3(0.0, 0.0, 0.0),
			make_double3(0.0, 0.0, 0.0)
		};
	}
	checkCudaErrors(cudaMemcpy(dpmi, hpmi, sizeof(device_polygon_mass_info) * nPobjs, cudaMemcpyHostToDevice));
	cu_particle_polygonObject_collision(1, dpi, dsphere, dpmi, pos, vel, omega, force, moment, mass, sorted_id, cell_start, cell_end, dcp, np);
	checkCudaErrors(cudaMemcpy(hpmi, dpmi, sizeof(device_polygon_mass_info) * nPobjs, cudaMemcpyDeviceToHost));
	po.toFront();
	while (po.hasNext())
	{
		po.next();
		unsigned int id = po.key();
		polygonObject* p = po.value();
		if (hpmi[id].force.z > 0.0)
			bool gg = true;
		p->setCollisionForce(VEC3D(hpmi[id].force.x, hpmi[id].force.y, hpmi[id].force.z));
		//p->setCollisionMoment(VEC3D(hpmi[id].moment.x, hpmi[id].moment.y, hpmi[id].moment.z));
	}
	checkCudaErrors(cudaFree(dpmi));
	delete[] hpmi;
}

void contact_particles_polygonObjects::cuda_collision(float *pos, float *vel, float *omega, float *mass, float *force, float *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{
	device_polygon_mass_info_f* hpmi = new device_polygon_mass_info_f[nPobjs];
	device_polygon_mass_info_f* dpmi = NULL;
	checkCudaErrors(cudaMalloc((void**)&dpmi, sizeof(device_polygon_mass_info_f) * nPobjs));
	QMapIterator<unsigned int, polygonObject*> po(pair_ip);
	while (po.hasNext())
	{
		po.next();
		unsigned int id = po.key();
		polygonObject* p = po.value();
		VEC3F pos = p->Position().To<float>();
		VEC3F vel = p->getVelocity().To<float>();
		VEC3F omega = p->getAngularVelocity().To<float>();
		EPF ep = p->getEP().To<float>();
		hpmi[id] =
		{
			make_float3(pos.x, pos.y, pos.z),
			make_float4(ep.e0, ep.e1, ep.e2, ep.e3),
			make_float3(vel.x, vel.y, vel.z),
			make_float3(omega.x, omega.y, omega.z),
			make_float3(0.0, 0.0, 0.0),
			make_float3(0.0, 0.0, 0.0)
		};
	}
	checkCudaErrors(cudaMemcpy(dpmi, hpmi, sizeof(device_polygon_mass_info_f) * nPobjs, cudaMemcpyHostToDevice));
	cu_particle_polygonObject_collision(1, dpi_f, dsphere_f, dpmi, pos, vel, omega, force, moment, mass, sorted_id, cell_start, cell_end, dcp_f, np);
	checkCudaErrors(cudaMemcpy(hpmi, dpmi, sizeof(device_polygon_mass_info_f) * nPobjs, cudaMemcpyDeviceToHost));
	po.toFront();
	while (po.hasNext())
	{
		po.next();
		unsigned int id = po.key();
		polygonObject* p = po.value();
		if (hpmi[id].force.z > 0.0)
			bool gg = true;
		p->setCollisionForce(
			VEC3D(
			(float)hpmi[id].force.x, 
			(float)hpmi[id].force.y, 
			(float)hpmi[id].force.z));
		//p->setCollisionMoment(VEC3D(hpmi[id].moment.x, hpmi[id].moment.y, hpmi[id].moment.z));
	}
	checkCudaErrors(cudaFree(dpmi));
	delete[] hpmi;
}

void contact_particles_polygonObjects::setZeroCollisionForce()
{
	foreach(polygonObject* pobj, pair_ip)
	{
		pobj->setCollisionForce(VEC3D(0.0));
		pobj->setCollisionMoment(VEC3D(0.0));
	}
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

