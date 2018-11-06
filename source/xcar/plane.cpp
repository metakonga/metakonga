#include "plane.h"
#include "mphysics_cuda_dec.cuh"
#include "pointMass.h"

unsigned int plane::nPlane = 0;

plane::plane()
	: pointMass()
	, l1(0)
	, l2(0)
{

}

plane::plane(QString& _name,  geometry_use _roll)
	: pointMass(_name, PLANE, _roll)
	, l1(0)
	, l2(0)
{

}

plane::plane(const plane& _plane)
	: pointMass(_plane)
	, l1(_plane.L1())
	, l2(_plane.L2())
	, xw(_plane.XW())
	, uw(_plane.UW())
	, u1(_plane.U1())
	, u2(_plane.U2())
	, pa(_plane.PA())
	, pb(_plane.PB())
	, w2(_plane.W2())
	, w3(_plane.W3())
	, w4(_plane.W4())
{

}

plane::~plane()
{
	if (!name.isEmpty())
		nPlane--;
//	if (dpi) checkCudaErrors(cudaFree(dpi));
}

bool plane::define(VEC3D& _xw, VEC3D& _pa, VEC3D& _pc, VEC3D& _pb)
{
	w2 = _pa; 
	minp.x = w2.x < minp.x ? w2.x : minp.x; minp.y = w2.y < minp.y ? w2.y : minp.y; minp.z = w2.z < minp.z ? w2.z : minp.z;
	maxp.x = w2.x < maxp.x ? w2.x : maxp.x; maxp.y = w2.y < maxp.y ? w2.y : maxp.y; maxp.z = w2.z < maxp.z ? w2.z : maxp.z;
	w3 = _pc;
	minp.x = w3.x < minp.x ? w3.x : minp.x; minp.y = w3.y < minp.y ? w3.y : minp.y; minp.z = w3.z < minp.z ? w3.z : minp.z;
	maxp.x = w3.x < maxp.x ? w3.x : maxp.x; maxp.y = w3.y < maxp.y ? w3.y : maxp.y; maxp.z = w3.z < maxp.z ? w3.z : maxp.z;
	w4 = _pb;
	minp.x = w4.x < minp.x ? w4.x : minp.x; minp.y = w4.y < minp.y ? w4.y : minp.y; minp.z = w4.z < minp.z ? w4.z : minp.z;
	maxp.x = w4.x < maxp.x ? w4.x : maxp.x; maxp.y = w4.y < maxp.y ? w4.y : maxp.y; maxp.z = w4.z < maxp.z ? w4.z : maxp.z;

	xw = _xw;
	minp.x = xw.x < minp.x ? xw.x : minp.x; minp.y = xw.y < minp.y ? xw.y : minp.y; minp.z = xw.z < minp.z ? xw.z : minp.z;
	maxp.x = xw.x < maxp.x ? xw.x : maxp.x; maxp.y = xw.y < maxp.y ? xw.y : maxp.y; maxp.z = xw.z < maxp.z ? xw.z : maxp.z;
	pa = _pa;
	pb = _pb;

	pa -= xw;
	pb -= xw;
	l1 = pa.length();
	l2 = pb.length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = u1.cross(u2);
	pointMass::pos = xw + (l1 * u1) + (l2 * u2);
	nPlane++;

	return true;
}

bool plane::define(VEC3D& _xw, VEC3D& _pa, VEC3D& _pb)
{
	xw = _xw;
	pa = _pa;
	pb = _pb;

	pa -= xw;
	pb -= xw;
	l1 = pa.length();
	l2 = pb.length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = u1.cross(u2);

	return true;
}