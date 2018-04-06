#include "particle_cluster.h"
//#include <mkl.h>

int particle_cluster::nc = 0;

particle_cluster::particle_cluster()
	: dist(0)
	, mass(0)
	, c_indice(NULL)
	, local(NULL)
{

}

particle_cluster::particle_cluster(unsigned int _nc)
	: dist(0)
	, mass(0)
	, c_indice(NULL)
	, local(NULL)
{
	c_indice = new unsigned int[nc];
	local = new VEC3D[nc];
}

particle_cluster::~particle_cluster()
{
	if (c_indice) delete [] c_indice; c_indice = NULL;
	if (local) delete [] local; local = NULL;
}

void particle_cluster::setIndice(unsigned int sid)
{
	//va_list ap;
	//nc = _nc;
	if (!c_indice)
		c_indice = new unsigned int[nc];
	if (!local)
		local = new VEC3D[nc];

	//va_start(ap, nc);
	//va_arg(ap, unsigned int);
	for (unsigned int i = 0; i < nc; i++){
		//unsigned int arg = va_arg(ap, unsigned int);
		c_indice[i] = sid + i;
	}
	//va_end(ap);
}

void particle_cluster::define(VEC4D* pos, double* _mass, double* _iner)
{
	
	VEC3D rc;
	for (unsigned int i = 0; i < nc; i++){
		unsigned int id = c_indice[i];
		rc += VEC3D(pos[id].x, pos[id].y, pos[id].z);
		mass += _mass[id];
	}
	com = rc / nc;

	VEC3D _in;
	for (unsigned int i = 0; i < nc; i++){
		unsigned int id = c_indice[i];
		VEC3D qk = VEC3D(pos[id].x, pos[id].y, pos[id].z) - com;
		_in.x += _iner[id] + _mass[id] * qk.z * qk.z + _mass[id] * qk.y * qk.y;
		_in.y += _iner[id] + _mass[id] * qk.x * qk.x + _mass[id] * qk.z * qk.z;
		_in.z += _iner[id] + _mass[id] * qk.y * qk.y + _mass[id] * qk.x * qk.x;
		local[i] = qk;
	}
	iner = 0.5 * _in;

	th.x = 0.0;// 0.25 * M_PI;
	th.y = 0.5 * M_PI;
	setTM();
	acc = VEC3D(0.0, -9.80665, 0.f);
	for (unsigned int i = 0; i < nc; i++){
		unsigned int id = c_indice[i];
		VEC3D el = A * local[i];
		VEC3D gp = com + el;
		pos[id] = VEC4D(gp.x, gp.y, gp.z, pos[id].w);
	}
}

void particle_cluster::updatePosition(VEC4D *pp, VEC3D* avp, VEC3D* aap, double dt)
{
	double sqt_dt = 0.5 * dt * dt;
	VEC3D _p;

	com += dt * vel + sqt_dt * acc;
	th = th + dt * dth + sqt_dt * ddth;
	setTM();

	for (unsigned int i = 0; i < nc; i++){
		unsigned int id = c_indice[i];
		VEC3D el = A * local[i];
		VEC3D gp = com + el;
		pp[id] = VEC4D(gp.x, gp.y, gp.z, pp[id].w);
	}
}

void particle_cluster::updateVelocity(VEC3D* v, VEC3D* w, VEC3D* force, VEC3D* moment, double dt)
{
	VEC3D fr;
	VEC3D mm;
	for (unsigned int i = 0; i < nc; i++)
	{
		unsigned int id = c_indice[i];
		fr += force[id];
		mm += moment[id];
	}
	fr += mass * VEC3D(0.f, -9.80665f, 0.f);
	double inv_m = 0;
	double inv_i = 0;
	inv_m = 1.0 / mass;
	double inv_ixx = 1.0 / iner.x;
	double inv_iyy = 1.0 / iner.y;
	double inv_izz = 1.0 / iner.z;
	vel += 0.5 * dt * acc;
	dth = dth + 0.5 * dt * ddth;

	setEOM(fr, mm);
	vel += 0.5f * dt * acc;
	dth = dth + 0.5f * dt * ddth;

	for (unsigned int i = 0; i < nc; i++)
	{
		unsigned int id = c_indice[i];
		v[id] = vel;
		w[id] = angularVelocity_in_globalCoordinate();
	}
}

void particle_cluster::setEOM(VEC3D& f, VEC3D& n)
{
	double sx = sin(th.x); double sy = sin(th.y); double sz = sin(th.z);
	double cx = cos(th.x); double cy = cos(th.y); double cz = cos(th.z);
	MAT33D iner_mat = MAT33D(iner.x, 0, 0, 0, iner.y, 0, 0, 0, iner.z);
	
	double inv_m = 1.0 / mass;

	MAT33D G(
		0.0, cx, sy * sx,
		0.0, sx, -sy * cx,
		1.0, 0.0, cy);

	MAT33D _G(
		sy * sz, cz, 0.0,
		sy * cz, -sz, 0.0,
		cy, 0.0, 1.0);

	MAT33D ii;
	MAT33D mtt = _G.t() * iner_mat * _G;
	double det = 1.0 / (mtt.a00*mtt.a11*mtt.a22 - mtt.a00*mtt.a12*mtt.a21 - mtt.a01*mtt.a10*mtt.a22 + mtt.a01*mtt.a12*mtt.a20 + mtt.a02*mtt.a10*mtt.a21 - mtt.a02*mtt.a11*mtt.a20);
	ii.a00 =  (mtt.a11 * mtt.a22 - mtt.a12 * mtt.a21) * det;
	ii.a01 = -(mtt.a01 * mtt.a22 - mtt.a02 * mtt.a21) * det;
	ii.a02 =  (mtt.a01 * mtt.a12 - mtt.a02 * mtt.a11) * det;
	ii.a10 = -(mtt.a10 * mtt.a22 - mtt.a12 * mtt.a20) * det;
	ii.a11 =  (mtt.a00 * mtt.a22 - mtt.a02 * mtt.a20) * det;
	ii.a12 = -(mtt.a00 * mtt.a12 - mtt.a02 * mtt.a10) * det;
	ii.a20 =  (mtt.a10 * mtt.a21 - mtt.a11 * mtt.a20) * det;
	ii.a21 = -(mtt.a00 * mtt.a21 - mtt.a01 * mtt.a20) * det;
	ii.a22 =  (mtt.a00 * mtt.a11 - mtt.a01 * mtt.a10) * det;

	VEC3D Qth = transpose(n, G);

	VEC3D w_bc = _G * dth;

	MAT33D _Gd(
		dth.y * cy * sz + dth.z * sy * cz, -dth.z * sz, 0.0,
		dth.y * cy * cz - dth.z * sy * sz, -dth.z * cz, 0.0,
		-dth.y * sy, 0.0, 0.0);

	VEC3D Itt_w = VEC3D(iner.x * w_bc.x, iner.y * w_bc.y, iner.z * w_bc.z);
	VEC3D _wXItt_w = w_bc.cross(Itt_w);
	VEC3D _Gd_dth = _Gd * dth;
	VEC3D Itt_Gd_dth = VEC3D(iner.x * _Gd_dth.x, iner.y * _Gd_dth.y, iner.z * _Gd_dth.z);
	VEC3D Qv_th = -transpose(_wXItt_w + Itt_Gd_dth, _G);

	acc.x = inv_m * f.x;
	acc.y = inv_m * f.y;
	acc.z = inv_m * f.z;
	ddth = ii * (Qth + Qv_th);
}

void particle_cluster::setTM()
{
	double sx = sin(th.x); double sy = sin(th.y); double sz = sin(th.z);
	double cx = cos(th.x); double cy = cos(th.y); double cz = cos(th.z);
	//VEC3D a = th;
	A.a00 = cx * cz - sx * cy * sz; A.a01 = -cx * sz - sx * cy * cz; A.a02 = sx * sy;
	A.a10 = sx * cz + cx * cy * sz; A.a11 = -sx * sz + cx * cy * cz; A.a12 = -cx * sy;
	A.a20 = sy * sz;                A.a21 = sy * cz;                 A.a22 = cy;
}

VEC3D particle_cluster::angularVelocity_in_globalCoordinate()
{
	MAT33D G(
		0.0, cos(th.x), sin(th.y) * sin(th.x),
		0.0, sin(th.x), -sin(th.y) * cos(th.x),
		1.0, 0.0, cos(th.y));

	return G * dth;
}