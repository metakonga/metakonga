#include "contact_plane_polygonObject.h"
#include "algebraMath.h"

contact_plane_polygonObject::contact_plane_polygonObject(
	QString _name, contactForce_type t, object* o1, object* o2)
	: contact(_name, t)
	, pe(NULL)
	, po(NULL)
	, dpi(NULL)
{
	contact::iobj = o1;
	contact::jobj = o2;
	pe = dynamic_cast<plane*>((o1->ObjectType() == PLANE ? o1 : o2));
	po = dynamic_cast<polygonObject*>((o1->ObjectType() == PLANE ? o2 : o1));
}

contact_plane_polygonObject::~contact_plane_polygonObject()
{
	if (dpi) checkCudaErrors(cudaFree(dpi)); dpi = NULL;
}

void contact_plane_polygonObject::collision(
	double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& F, VEC3D& M)
{
	singleCollision(pe, m, r, pos, vel, omega, F, M);
}

void contact_plane_polygonObject::cudaMemoryAlloc()
{
	contact::cudaMemoryAlloc();
	device_plane_info *_dpi = new device_plane_info;
	_dpi->l1 = pe->L1();
	_dpi->l2 = pe->L2();
	_dpi->xw = make_double3(pe->XW().x, pe->XW().y, pe->XW().z);
	_dpi->uw = make_double3(pe->UW().x, pe->UW().y, pe->UW().z);
	_dpi->u1 = make_double3(pe->U1().x, pe->U1().y, pe->U1().z);
	_dpi->u2 = make_double3(pe->U2().x, pe->U2().y, pe->U2().z);
	_dpi->pa = make_double3(pe->PA().x, pe->PA().y, pe->PA().z);
	_dpi->pb = make_double3(pe->PB().x, pe->PB().y, pe->PB().z);
	_dpi->w2 = make_double3(pe->W2().x, pe->W2().y, pe->W2().z);
	_dpi->w3 = make_double3(pe->W3().x, pe->W3().y, pe->W3().z);
	_dpi->w4 = make_double3(pe->W4().x, pe->W4().y, pe->W4().z);
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));
	checkCudaErrors(cudaMemcpy(dpi, _dpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
	delete _dpi;
}

double contact_plane_polygonObject::plane_polygon_contact_detection(
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

void contact_plane_polygonObject::singleCollision(
	plane* _pe, double mass, double rad, VEC3D& pos, VEC3D& vel,
	VEC3D& omega, VEC3D& force, VEC3D& moment)
{
	VEC3D dp = pos - _pe->XW();
	VEC3D wp = VEC3D(dp.dot(_pe->U1()), dp.dot(_pe->U2()), dp.dot(_pe->UW()));
	VEC3D u;

	double cdist = plane_polygon_contact_detection(_pe, u, pos, wp, rad);
	if (cdist > 0){
		double rcon = rad - 0.5 * cdist;
		VEC3D cp = rcon * u;
		VEC3D dv = -(vel + omega.cross(rad * u));

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