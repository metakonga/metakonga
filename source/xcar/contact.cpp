#include "contact.h"
#include <QDebug>

unsigned int contact::count = 0;

double contact::cohesionForce(double coh_r, double coh_e, double Fn)
{
	double cf = 0.0;
	if (cohesion){
// 		double req = (ri * rj / (ri + rj));
// 		double Eeq_inv = ((1.0 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej);
		double rcp = (3.0 * coh_r * (-Fn)) / (4.0 * (1.0 / coh_e));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = cohesion * Ac;
	}
	return cf;
}

void contact::DHSModel
(
	contactParameters& c, double cdist, VEC3D& cp,
	VEC3D& dv, VEC3D& unit, VEC3D& F, VEC3D& M
)
{
	VEC3D Fn, Ft;
	double fsn = (-c.kn * pow(cdist, 1.5));
	double fca = cohesionForce(c.coh_r, c.coh_e, fsn);
	double fsd = c.vn * dv.dot(unit);
//	qDebug() << "Cohesion force : " << fca;
	Fn = (fsn + fca + fsd) * unit;
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
	, cohesion(0)
	, ignore_time(0)
	, f_type(t)
	, dcp(NULL)
	, dcp_f(NULL)
	, mpp(NULL)
	, type(NO_CONTACT_PAIR)
	, iobj(NULL)
	, jobj(NULL)
	, is_enabled(true)
{
	count++;
	mpp = { 0, };
}

contact::contact(const contact* c)
	: name(c->Name())
	, dcp(NULL)
	, dcp_f(NULL)
	, mpp(NULL)
	, iobj(NULL)
	, jobj(NULL)
	, is_enabled(true)
	, ignore_time(0)
	, cohesion(c->Cohesion())
	, restitution(c->Restitution())
	, friction(c->Friction())
	, stiffnessRatio(c->StiffnessRatio())
	, f_type(c->ForceMethod())
	, type(c->PairType())
{
	mpp = new material_property_pair;
	memcpy(mpp, c->MaterialPropertyPair(), sizeof(material_property_pair));
	if (c->DeviceContactProperty())
	{
		checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
		checkCudaErrors(cudaMemcpy(dcp, c->DeviceContactProperty(), sizeof(device_contact_property), cudaMemcpyDeviceToDevice));
	}
}

contact::~contact()
{
	if (dcp) cudaFree(dcp); dcp = NULL;
	if (mpp) delete[] mpp; mpp = NULL;
}

void contact::setContactParameters(double r, double rt, double f, double c)
{
	restitution = r;
	stiffnessRatio = rt;
	friction = f;
	cohesion = c;
}

void contact::cudaMemoryAlloc()
{
	device_contact_property hcp = device_contact_property
	{
		mpp->Ei, mpp->Ej, mpp->pri, mpp->prj, mpp->Gi, mpp->Gj,
		restitution, friction, 0.0, cohesion, stiffnessRatio
	};
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(device_contact_property), cudaMemcpyHostToDevice));
}

void contact::cudaMemoryAlloc_f()
{
	device_contact_property_f hcp_f = device_contact_property_f
	{
		(float)mpp->Ei, (float)mpp->Ej, (float)mpp->pri, (float)mpp->prj, (float)mpp->Gi, (float)mpp->Gj,
		(float)restitution, (float)friction, 0.0f, (float)cohesion, (float)stiffnessRatio
	};
	checkCudaErrors(cudaMalloc((void**)&dcp_f, sizeof(device_contact_property_f)));
	checkCudaErrors(cudaMemcpy(dcp_f, &hcp_f, sizeof(device_contact_property_f), cudaMemcpyHostToDevice));
}

contact::contactParameters contact::getContactParameters
(
	double ir, double jr,
	double im, double jm,
	double iE, double jE,
	double ip, double jp,
	double is, double js)
{
	contactParameters cp;
 	double Meq = jm ? (im * jm) / (im + jm) : im;
 	double Req = jr ? (ir * jr) / (ir + jr) : ir;
 	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	cp.coh_e = ((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE);
 	double lne = log(restitution);
	double beta = 0.0;
	switch (f_type)
	{
	case DHS:
		beta = (M_PI / log(restitution));
		cp.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		cp.vn = sqrt((4.0 * Meq * cp.kn) / (1.0 + beta * beta));
		cp.ks = cp.kn * stiffnessRatio;
		cp.vs = cp.vn * stiffnessRatio;
		break;
	}
	cp.coh_r = Req;
	return cp;
}

void contact::setMaterialPair(material_property_pair _mpp)
{
	if (!mpp)
		mpp = new material_property_pair;
	memcpy(mpp, &_mpp, sizeof(material_property_pair));
}

contact::pairType contact::getContactPair(geometry_type t1, geometry_type t2)
{
	return static_cast<pairType>(t1 + t2);
}

void contact::collision(
	double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& fn, VEC3D& ft)
{

}

void contact::cuda_collision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{

}

void contact::cuda_collision(float *pos, float *vel, float *omega, float *mass, float *force, float *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{

}