#include "plane.h"
#include "modeler.h"
#include "mphysics_cuda_dec.cuh"
//#include <helper_math.h>

plane::plane()
	:object()
	, l1(0)
	, l2(0)
	, dpi(NULL)
{

}

plane::plane(modeler *_md, QString& _name, tMaterial _mat, tRoll _roll)
	: object(_md, _name, PLANE, _mat, _roll)
	, l1(0)
	, l2(0)
	, dpi(NULL)
{

}

plane::plane(const plane& _plane)
	: object(_plane)
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
	, dpi(NULL)
{

}

plane::~plane()
{
//	if (dpi) checkCudaErrors(cudaFree(dpi));
}

bool plane::define(vector3<float>& _xw, vector3<float>& _pa, vector3<float>& _pc, vector3<float>& _pb)
{
	w2 = _pa;
	w3 = _pc;
	w4 = _pb;

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

bool plane::define(vector3<float>& _xw, vector3<float>& _pa, vector3<float>& _pb)
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

unsigned int plane::makeParticles(float _rad, float _spacing, bool isOnlyCount, VEC4F_PTR pos, unsigned int sid)
{
	return 0;
}

void plane::cuAllocData(unsigned int _np)
{
	device_plane_info *_dpi = new device_plane_info;
	_dpi->l1 = l1;
	_dpi->l2 = l2;
	_dpi->xw = make_float3(xw.x, xw.y, xw.z);
	_dpi->uw = make_float3(uw.x, uw.y, uw.z);
	_dpi->u1 = make_float3(u1.x, u1.y, u1.z);
	_dpi->u2 = make_float3(u2.x, u2.y, u2.z);
	_dpi->pa = make_float3(pa.x, pa.y, pa.z);
	_dpi->pb = make_float3(pb.x, pb.y, pb.z);
	_dpi->w2 = make_float3(w2.x, w2.y, w2.z);
	_dpi->w3 = make_float3(w3.x, w3.y, w3.z);
	_dpi->w4 = make_float3(w4.x, w4.y, w4.z);
	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info)));
	checkCudaErrors(cudaMemcpy(dpi, _dpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
	object::setCudaRelativeImpactVelocity(_np);
	delete _dpi;
}

void plane::updateMotion(float dt, tSolveDevice tsd)
{
	if (tsd == CPU){
		xw += dt * VEC3F(0.0f, 0.0f, 0.0f);	//plane motion setting, m/s, CPU code
		w2 += dt * VEC3F(0.0f, 0.0f, 0.0f);
		w3 += dt * VEC3F(0.0f, 0.0f, 0.0f);
		w4 += dt * VEC3F(0.0f, 0.0f, 0.0f);
	}
	else if(tsd == GPU){
		device_plane_info *h_dpi = new device_plane_info;
		checkCudaErrors(cudaMemcpy(h_dpi, dpi, sizeof(device_plane_info), cudaMemcpyDeviceToHost));
		// plane motion setting, m/s, GPU code, mde���Ͽ��� plane �������� �������� '1'�� �ٲٱ�
		h_dpi->xw = make_float3(h_dpi->xw.x + dt * 0.0f, h_dpi->xw.y + dt *  0.2f, h_dpi->xw.z + dt *  0.0f);
		h_dpi->w2 = make_float3(h_dpi->w2.x + dt * 0.0f, h_dpi->w2.y + dt *  0.2f, h_dpi->w2.z + dt *  0.0f);
		h_dpi->w3 = make_float3(h_dpi->w3.x + dt * 0.0f, h_dpi->w3.y + dt *  0.2f, h_dpi->w3.z + dt *  0.0f);
		h_dpi->w4 = make_float3(h_dpi->w4.x + dt * 0.0f, h_dpi->w4.y + dt *  0.2f, h_dpi->w4.z + dt *  0.0f);
		checkCudaErrors(cudaMemcpy(dpi, h_dpi, sizeof(device_plane_info), cudaMemcpyHostToDevice));
		delete h_dpi;
	}
	
}

void plane::save_shape_data(QTextStream& ts) const
{
	//QTextStream ts(&(md->modelStream()));
	bool isExistMass = ms ? true : false;

	ts << "OBJECT PLANE " << id << " " << name << " " << roll_type << " " << mat_type << " " << (int)_expression << " " << isExistMass << endl
		<< xw.x << " " << xw.y << " " << xw.z << endl;
	VEC3F ap = xw + pa;
	ts << ap.x << " " << ap.y << " " << ap.z << endl;
	VEC3F cp = xw + pb + l1 * u1;
	ts << cp.x << " " << cp.y << " " << cp.z << endl;
	VEC3F bp = xw + pb;
	ts << bp.x << " " << bp.y << " " << bp.z << endl;

	if (isExistMass)
	{
		save_mass_data(ts);
	}
}