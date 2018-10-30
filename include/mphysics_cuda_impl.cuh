#include "mphysics_cuda_dec.cuh"
#include <helper_math.h>

__constant__ device_parameters cte;
__constant__ device_parameters_f cte_f;

inline __device__ int sign(float L)
{
	return L < 0 ? -1 : 1;
}

inline __device__ int sign(double L)
{
	return L < 0 ? -1 : 1;
}

inline __device__ double dot(double3& v1, double3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ double3 operator-(double3& v1, double3& v2)
{
	return make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
inline __device__ double3 operator-(double3& v1)
{
	return make_double3(-v1.x, -v1.y, -v1.z);
}

inline __device__ double3 operator*(double v1, double3& v2)
{
	return make_double3(v1 * v2.x, v1 * v2.y, v1 * v2.z);
}

inline __device__ double3 operator+(double3& v1, double3& v2)
{
	return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __host__ __device__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ double3 operator/(double3& v1, double v2)
{
	return make_double3(v1.x / v2, v1.y / v2, v1.z / v2);
}

inline __device__ double length(double3& v1)
{
	return sqrt(dot(v1, v1));
}

inline __device__ double3 cross(double3 a, double3 b)
{
	return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__
uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (cte.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (cte.grid_size.y - 1);
	gridPos.z = gridPos.z & (cte.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, cte.grid_size.y), cte.grid_size.x) + __umul24(gridPos.y, cte.grid_size.x) + gridPos.x;
}

// calculate position in uniform grid
__device__
int3 calcGridPos(double3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - cte.world_origin.x) / cte.cell_size);
	gridPos.y = floor((p.y - cte.world_origin.y) / cte.cell_size);
	gridPos.z = floor((p.z - cte.world_origin.z) / cte.cell_size);
	return gridPos;
}

__global__ void vv_update_position_kernel(double4* pos, double3* vel, double3* acc, unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;

	double3 _p = cte.dt * vel[id] + cte.half2dt * acc[id];
	pos[id].x += _p.x;
	pos[id].y += _p.y;
	pos[id].z += _p.z;

}

__global__ void vv_update_velocity_kernel(
	double3* vel,
	double3* acc,
	double3* omega,
	double3* alpha,
	double3* force,
	double3* moment,
	double* mass,
	double* iner,
	unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double m = mass[id];
	double3 v = vel[id];
	//double3 L = acc[id];
	double3 av = omega[id];
	//double3 aa = alpha[id];
	double3 a = (1.0 / m) * force[id];
	double3 in = (1.0 / iner[id]) * moment[id];
	v += 0.5 * cte.dt * (acc[id] + a);
	av += 0.5 * cte.dt * (alpha[id] + in);// aa;
	//L = (1.0 / m) * force[id];
	//aa = (1.0 / in) * moment[id];
	//v += 0.5 * cte.dt * L;
	//av += 0.5 * cte.dt * aa;
	// 	if(id == 0){
	// 		printf("Velocity --- > id = %d -> [%f.6, %f.6, %f.6]\n", id, v.x, v.y, v.z);
	// 	}
	force[id] = m * cte.gravity;
	moment[id] = make_double3(0.0, 0.0, 0.0);
	vel[id] = v;
	omega[id] = av;
	acc[id] = a;
	alpha[id] = in;
}


__global__ void calculateHashAndIndex_kernel(unsigned int* hash, unsigned int* index, double4* pos, unsigned int np)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (np)) return;
	volatile double4 p = pos[id];

	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned _hash = calcGridHash(gridPos);
	/*if(_hash >= cte.ncell)
	printf("Over limit - hash number : %d", _hash);*/
	hash[id] = _hash;
	index[id] = id;
}

__global__ void calculateHashAndIndexForPolygonSphere_kernel(
	unsigned int* hash, unsigned int* index, 
	unsigned int sid, unsigned int nsphere, double4* sphere)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= nsphere) return;
	volatile double4 p = sphere[id];
	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned int _hash = calcGridHash(gridPos);
	hash[sid + id] = _hash;
	index[sid + id] = sid + id;
}

__global__ void reorderDataAndFindCellStart_kernel(
	unsigned int* hash, 
	unsigned int* index, 
	unsigned int* cstart, 
	unsigned int* cend, 
	unsigned int* sorted_index,
	unsigned int np)
{
	extern __shared__ uint sharedHash[];	//blockSize + 1 elements
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned _hash;

	//unsigned int tnp = ;// cte.np + cte.nsphere;

	if (id < np)
	{
		_hash = hash[id];

		sharedHash[threadIdx.x + 1] = _hash;

		if (id > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[id - 1];
		}
	}
	__syncthreads();

	if (id < np)
	{
		if (id == 0 || _hash != sharedHash[threadIdx.x])
		{
			cstart[_hash] = id;

			if (id > 0)
				cend[sharedHash[threadIdx.x]] = id;
		}

		if (id == np - 1)
		{
			cend[_hash] = id + 1;
		}

		unsigned int sortedIndex = index[id];
		sorted_index[id] = sortedIndex;
	}
}

__device__ device_force_constant getConstant(
	int tcm, double ir, double jr, double im, double jm,
	double iE, double jE, double ip, double jp,
	double iG, double jG, double rest,
	double fric, double rfric, double sratio)
{
	device_force_constant dfc = { 0, 0, 0, 0, 0, 0 };
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	switch (tcm)
	{
	case 0:{
		double Geq = (iG * jG) / (iG*(2 - jp) + jG*(2 - ip));
		double ln_e = log(rest);
		double xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
		dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		dfc.vn = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.kn * Meq);
		dfc.ks = 8.0 * Geq * sqrt(Req);
		dfc.vs = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.ks * Meq);
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	case 1:{
		double beta = (M_PI / log(rest));
		dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		dfc.vn = sqrt((4.0 * Meq * dfc.kn) / (1.0 + beta * beta));
		dfc.ks = dfc.kn * sratio;
		dfc.vs = dfc.vn * sratio;
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	}
	
// 	dfc.kn = /*(16.f / 15.f)*sqrt(er) * eym * pow((T)((15.f * em * 1.0f) / (16.f * sqrt(er) * eym)), (T)0.2f);*/ (4.0f / 3.0f)*sqrt(er)*eym;
// 	dfc.vn = sqrt((4.0f*em * dfc.kn) / (1 + beta * beta));
// 	dfc.ks = dfc.kn * ratio;
// 	dfc.vs = dfc.vn * ratio;
// 	dfc.mu = fric;
	return dfc;
}

// ref. Three-dimensional discrete element modelling (DEM) of tillage: Accounting for soil cohesion and adhesion
__device__ double cohesionForce(
	double ri,
	double rj,
	double Ei,
	double Ej,
	double pri,
	double prj,
	double coh,
	double Fn)
{
	double cf = 0.f;
	if (coh){
		double req = (ri * rj / (ri + rj));
		double Eeq = ((1.0 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej);
		double rcp = (3.0 * req * (-Fn)) / (4.0 * (1.0 / Eeq));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = coh * Ac;
	}
	return cf;
}

// __device__ bool calForce(
// 	float ir,
// 	float jr,
// 	float im,
// 	float jm,
// 	float rest,
// 	float sh,
// 	float fric,
// 	float rfric,
// 	float E,
// 	float pr,
// 	float coh,
// 	float4 ipos,
// 	float4 jpos,
// 	float3 ivel,
// 	float3 jvel,
// 	float3 iomega,
// 	float3 jomega,
// 	float3& force,
// 	float3& moment
// 	/*float *riv*/)
// {
// 	float3 relative_pos = make_float3(jpos - ipos);
// 	float dist = length(relative_pos);
// 	float collid_dist = (ir + jr) - dist;
// 	float3 shear_force = make_float3(0.f);
// 	if (collid_dist <= 0){
// 		//*riv = 0.f;
// 		return false;
// 	}
// 	else{
// 		float rcon = ir - 0.5f * collid_dist;
// 		float3 unit = relative_pos / dist;
// 		float3 relative_vel = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
// 		//*riv = abs(length(relative_vel));
// 		device_force_constant<float> c = getConstant<float>(ir, jr, im, jm, E, E, pr, pr, sh, sh, rest, fric, rfric);
// 		float fsn = -c.kn * pow(collid_dist, 1.5f);
// 		float fca = cohesionForce(ir, jr, E, E, pr, pr, coh, fsn);
// 		float fsd = c.vn * dot(relative_vel, unit);
// 		float3 single_force = (fsn + fca + fsd) * unit;
// 		//float3 single_force = (-c.kn * pow(collid_dist, 1.5f) + c.vn * dot(relative_vel, unit)) * unit;
// 		float3 single_moment = make_float3(0, 0, 0);
// 		float3 e = relative_vel - dot(relative_vel, unit) * unit;
// 		float mag_e = length(e);
// 		if (mag_e){
// 			float3 s_hat = e / mag_e;
// 			float ds = mag_e * cte.dt;
// 			float fst = -c.ks * ds;
// 			float fdt = c.vs * dot(relative_vel, s_hat);
// 			shear_force = (fst + fdt) * s_hat;
// 			if (length(shear_force) >= c.mu * length(single_force))
// 				shear_force = c.mu * fsn * s_hat;
// 			single_moment = cross(rcon * unit, shear_force);
// 			if (length(iomega)){
// 				float3 on = iomega / length(iomega);
// 				single_moment += -rfric * fsn * rcon * on;
// 			}
// 			//shear_force = min(c.ks * ds + c.vs * (dot(relative_vel, s_hat)), c.mu * length(single_force)) * s_hat;
// 			//single_moment = cross(ir * unit, shear_force);
// 		}
// 		force += single_force + shear_force;
// 		moment += single_moment;
// 	}
// 	
// 	return true;
// }
__device__ void HMCModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double rcon, double cdist, double3 iomega,
	double3 dv, double3 unit, double3& Ft, double3& Fn, double3& M)
{
// 	if (coh && cdist < 1.0E-8)
// 		return;

	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
// 	if ((fsn + fca + fdn) < 0 && ir)
// 		return;
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e){
		double3 s_hat = -(e / mag_e);
		double ds = mag_e * cte.dt;
		double fst = -c.ks * ds;
		double fdt = c.vs * dot(dv, s_hat);
		Ft = (fst + fdt) * s_hat;
		if (length(Ft) >= c.mu * length(Fn))
			Ft = c.mu * fsn * s_hat;
		M = cross(ir * unit, Ft);
		if (length(iomega)){
			double3 on = iomega / length(iomega);
			M += c.ms * fsn * rcon * on;
		}
	}
}

__device__ void DHSModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double rcon, double cdist, double3 iomega,
	double3 dv, double3 unit, double3& Ft, double3& Fn, double3& M)
{
	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e)
	{
		double3 sh = e / mag_e;
		double ds = mag_e * cte.dt;
		Ft = min(c.ks * ds + c.vs * (dot(dv, sh)), c.mu * length(Fn)) * sh;
		M = cross(ir * unit, Ft);
		/*if (length(iomega)){
			double3 on = iomega / length(iomega);
			M += c.ms * fsn * rcon * on;
		}*/
	}
}

template <int TCM>
__global__ void calculate_p2p_kernel(
	double4* pos, double3* vel,
	double3* omega,	double3* force, 
	double3* moment, double* mass,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;

	double4 ipos = pos[id];
	double4 jpos = make_double4(0, 0, 0, 0);
	double3 ivel = vel[id];
	double3 jvel = make_double3(0, 0, 0);
	double3 iomega = omega[id];
	double3 jomega = make_double3(0, 0, 0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));

	double ir = ipos.w; double jr = 0;
	double im = mass[id]; double jm = 0;
	double3 Ft = make_double3(0, 0, 0);
	double3 Fn = make_double3(0, 0, 0);// [id] * cte.gravity;
	double3 M = make_double3(0, 0, 0);
	double3 sumF = make_double3(0, 0, 0);
	double3 sumM = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)
							continue;
						jpos = pos[k]; jvel = vel[k]; jomega = omega[k];
						jr = jpos.w; jm = mass[k];
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0){
							double rcon = ir - 0.5 * cdist;
							double3 unit = rp / dist;
							double3 rv = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
							device_force_constant c = getConstant(
								TCM, ir, jr, im, jm, cp->Ei, cp->Ej, 
								cp->pri, cp->prj, cp->Gi, cp->Gj,
								cp->rest, cp->fric, cp->rfric, cp->sratio);
							switch (TCM)
							{
							case 0:
								HMCModel(
									c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega, 
									rv, unit, Ft, Fn, M);
								break;
							case 1:
								DHSModel(
									c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega,
									rv, unit, Ft, Fn, M);
								break;
							}
							sumF += Fn + Ft;
							sumM += M;
						}
					}
				}
			}
		}
	}
	force[id] += sumF;
	moment[id] += sumM;
}

__device__ double particle_plane_contact_detection(
	device_plane_info *pe, double3& xp, double3& wp, double3& u, double r)
{
	double a_l1 = pow(wp.x - pe->l1, 2.0);
	double b_l2 = pow(wp.y - pe->l2, 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r*r;

	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe->l1) && (wp.y > 0 && wp.y < pe->l2)){
		double3 dp = xp - pe->xw;
		double3 uu = pe->uw / length(pe->uw);
		int pp = -sign(dot(dp, pe->uw));// dp.dot(pe->UW()));
		u = pp * uu;
		double collid_dist = r - abs(dot(dp, u));// dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		double3 Xsw = xp - pe->xw;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		double3 Xsw = xp - pe->w2;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->l1 && wp.y > pe->l2 && (a_l1 + b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe->w3;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe->l2 && (sqa + b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe->w4;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe->l1) && wp.y < 0 && (sqb + sqc) < sqr){
		double3 Xsw = xp - pe->xw;
		double3 wj_wi = pe->w2 - pe->xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe->l1) && wp.y > pe->l2 && (b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe->w4;
		double3 wj_wi = pe->w3 - pe->w4;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x < 0 && (sqr + sqc) < sqr){
		double3 Xsw = xp - pe->xw;
		double3 wj_wi = pe->w4 - pe->xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw,us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x > pe->l1 && (a_l1 + sqc) < sqr){
		double3 Xsw = xp - pe->w2;
		double3 wj_wi = pe->w3 - pe->w2;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw,us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}


	return -1.0;
}

template <int TCM>
__global__ void plane_contact_force_kernel(
	device_plane_info *plane,
	double4* pos, double3* vel, double3* omega, 
	double3* force, double3* moment,
	device_contact_property *cp, double* mass, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	double m = mass[id];
	double4 ipos = pos[id];
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double r = ipos.w;
	double3 ivel = vel[id];
	double3 iomega = omega[id];

	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);
	double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - plane->xw;
	double3 unit = make_double3(0, 0, 0);
	double3 wp = make_double3(dot(dp, plane->u1), dot(dp, plane->u2), dot(dp, plane->uw));
	
	double cdist = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
	if (cdist > 0){
		double rcon = r - 0.5 * cdist;
		double3 dv = -(ivel + cross(iomega, r * unit));
		device_force_constant c = getConstant(
			TCM, r, 0.0, m, 0.0, cp->Ei, cp->Ej,
			cp->pri, cp->prj, cp->Gi, cp->Gj,
			cp->rest, cp->fric, cp->rfric, cp->sratio);
		switch (TCM)
		{
		case 0: 
			HMCModel(
				c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
				iomega, dv, unit, Ft, Fn, M); 
			break;
		case 1:
			DHSModel(
				c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
				iomega, dv, unit, Ft, Fn, M);
			break;
		}
	}
	force[id] += Fn + Ft;// m_force + shear_force;
	moment[id] += M;
}


__device__ double3 makeTFM_1(double4& ep)
{
	return make_double3(2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5), 2.0 * (ep.y*ep.z - ep.x*ep.w), 2.0 * (ep.y*ep.w + ep.x*ep.z));
}

__device__ double3 makeTFM_2(double4& ep)
{
	return make_double3(2.0 * (ep.y*ep.z + ep.x*ep.w), 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5), 2.0 * (ep.z*ep.w - ep.x*ep.y));
}

__device__ double3 makeTFM_3(double4& ep)
{
	return make_double3(2.0 * (ep.y*ep.w - ep.x*ep.z), 2.0 * (ep.z*ep.w + ep.x*ep.y), 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5));
}

__device__ double3 toLocal(double3& A_1, double3& A_2, double3& A_3, double3& v)
{
	return make_double3
		(A_1.x * v.x + A_2.x * v.y + A_3.x * v.z, 
		 A_1.y * v.x + A_2.y * v.y + A_3.y * v.z,
		 A_1.z * v.x + A_2.z * v.y + A_3.z * v.z);
}

__device__ double3 toGlobal(double3& A_1, double3& A_2, double3& A_3, double3& v)
{
	return make_double3(dot(A_1, v), dot(A_2, v), dot(A_3, v));
}

__device__ float particle_cylinder_contact_detection(
	device_cylinder_info* cy, double4& pt, double3& u, double3& cp, unsigned int id = 0)
{
	double dist = -1.0;
	double3 ab = make_double3(cy->ptop.x - cy->pbase.x, cy->ptop.y - cy->pbase.y, cy->ptop.z - cy->pbase.z);
	double3 p = make_double3(pt.x, pt.y, pt.z);
	double t = dot(p - cy->pbase, ab) / dot(ab, ab);
	double3 _cp = make_double3(0.0, 0.0, 0.0);
	if (t >= 0 && t <= 1){
		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		u = (_cp - p) / dist;
		cp = _cp - cy->rbase * u;
		return cy->rtop + pt.w - dist;
	}
	else{
		
		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		if (dist < cy->rbase){
			double3 OtoCp = cy->origin - _cp;
			double OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
			cp = _cp - cy->rbase * u;
			return cy->len * 0.5 + pt.w - OtoCp_;
		}
		double3 A_1 = makeTFM_1(cy->ep);
		double3 A_2 = makeTFM_2(cy->ep);
		double3 A_3 = makeTFM_3(cy->ep);
		double3 _at = p - cy->ptop;
		double3 at = toLocal(A_1, A_2, A_3, _at);
		double r = length(at);
		cp = cy->ptop;
		if (abs(at.y) > cy->len){
			_at = p - cy->pbase;
			at = toLocal(A_1, A_2, A_3, _at);
			cp = cy->pbase;
		}
		double pi = atan(at.x / at.z);
		if (pi < 0 && at.z < 0){
			_cp.x = cy->rbase * sin(-pi);
		}
		else if (pi > 0 && at.x < 0 && at.z < 0){
			_cp.x = cy->rbase * sin(-pi);
		}
		else{
			_cp.x = cy->rbase * sin(pi);
		}
		_cp.z = cy->rbase * cos(pi);
		if (at.z < 0 && _cp.z > 0){
			_cp.z = -_cp.z;
		}
		else if (at.z > 0 && _cp.z < 0){
			_cp.z = -_cp.z;
		}
		_cp.y = 0.;
		cp = cp + toGlobal(A_1, A_2, A_3, _cp);

		double3 disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < pt.w){
			return pt.w - dist;
		}
	}
	return -1.0;
}

template<int TCM>
__global__ void cylinder_hertzian_contact_force_kernel(
	device_cylinder_info *cy,
	double4* pos, double3* vel, double3* omega, 
	double3* force, double3* moment, device_contact_property *cp,
	double* mass, double3* mpos, double3* mf, double3* mm, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;

	*mf = make_double3(0.0, 0.0, 0.0);
	*mm = make_double3(0.0, 0.0, 0.0);
	double cdist = 0.0;
	double im = mass[id];
	double4 ipos = make_double4(pos[id].x, pos[id].y, pos[id].z, pos[id].w);
	double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
	double3 iomega = make_double3(omega[id].x, omega[id].y, omega[id].z);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	double3 cpt = make_double3(0.0, 0.0, 0.0);
	double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	cdist = particle_cylinder_contact_detection(cy, ipos, unit, cpt, id);
	double3 si = cpt - mp;
	double3 cy2cp = cpt - cy->origin;
	double3 Ft = make_double3(0.0, 0.0, 0.0);
	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 M = make_double3(0.0, 0.0, 0.0);
	if (cdist > 0)
	{
		double rcon = ipos.w - 0.5 * cdist;
		double3 dv = cy->vel + cross(cy->omega, cy2cp) - (ivel + cross(iomega, ipos.w * unit));
		device_force_constant c = getConstant(
			TCM, ipos.w, 0, im, 0, cp->Ei, cp->Ej,
			cp->pri, cp->prj, cp->Gi, cp->Gj,
			cp->rest, cp->fric, cp->rfric, cp->sratio);
		switch (TCM)
		{
		case 0: HMCModel(
				c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
				dv, unit, Ft, Fn, M); 
			break;
		case 1:
			DHSModel(
				c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
				dv, unit, Ft, Fn, M);
			break;
		}
	}
	double3 sum_f = Fn + Ft;
	force[id] += make_double3(sum_f.x, sum_f.y, sum_f.z);
	moment[id] += make_double3(M.x, M.y, M.z);
	mf[id] = -(Fn);
	mm[id] = cross(si, -Fn);
}

template <typename T, unsigned int blockSize>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	/*extern*/ __shared__ T sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	
	T mySum;// = make_double3(0, 0, 0);;
	mySum.x = 0.0;
	mySum.y = 0.0;
	mySum.z = 0.0;
	//sdata[tid] = make_double3(0, 0, 0);

	while (i < n)
	{
		//sdata[tid] += g_idata[i] + g_idata[i + blockSize]; 
		mySum += g_idata[i];
 		if (i + blockSize < n)
			mySum += g_idata[i + blockSize];
		i += gridSize;
	}
	sdata[tid] = mySum;
	__syncthreads();
	if ((blockSize >= 512) && (tid < 256)) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();
	if ((blockSize >= 256) && (tid < 128)) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
	if ((blockSize >= 128) && (tid <  64)) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads();
	if ((blockSize >= 64) && (tid < 32)){ sdata[tid] = mySum = mySum + sdata[tid + 32]; } __syncthreads();
	if ((blockSize >= 32) && (tid < 16)){ sdata[tid] = mySum = mySum + sdata[tid + 16];	} __syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	__syncthreads();

	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__device__ double3 closestPtPointTriangle(
	device_polygon_info& dpi,
	double3& p,
	double pr)
{
	double3 a = dpi.P;
	double3 b = dpi.Q;
	double3 c = dpi.R;
	double3 ab = b - a;
	double3 ac = c - a;
	double3 ap = p - a;

	double d1 = dot(ab, ap);
	double d2 = dot(ac, ap);
	if (d1 <= 0.0 && d2 <= 0.0){
	//	*wc = 0;
		return a;
	}

	double3 bp = p - b;
	double d3 = dot(ab, bp);
	double d4 = dot(ac, bp);
	if (d3 >= 0.0 && d4 <= d3){
	//	*wc = 0;
		return b;
	}
	double vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
	//	*wc = 1;
		double v = d1 / (d1 - d3);
		return a + v * ab;
	}

	double3 cp = p - c;
	double d5 = dot(ab, cp);
	double d6 = dot(ac, cp);
	if (d6 >= 0.0 && d5 <= d6){
	//	*wc = 0;
		return c;
	}

	double vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
	//	*wc = 1;
		double w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
	//	*wc = 1;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	//*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;

	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

__device__ bool checkConcave(device_polygon_info* dpi, unsigned int* tid, unsigned int kid, double4* dsph, unsigned int cnt)
{
	double3 p1 = make_double3(0, 0, 0);
	double3 u1 = make_double3(0, 0, 0);
	double3 p2 = make_double3(dsph[kid].x, dsph[kid].y, dsph[kid].z);
	double3 u2 = dpi[kid].N;
	for (unsigned int i = 0; i < cnt; i++){
		unsigned int id = tid[i];
		p1 = make_double3(dsph[id].x, dsph[id].y, dsph[id].z);
		u1 = dpi[id].N;
		double3 p2p1 = p2 - p1;
		double chk1 = dot(p2p1, u1);
		double chk2 = dot(-p2p1, u2);
		if (chk1 > 0 && chk2 > 0)
		{
			tid[cnt++] = id;
			return true;
		}
	}
	return false;
}

template<int TCM>
__global__ void particle_polygonObject_collision_kernel(
	device_polygon_info* dpi, double4* dsph, device_polygon_mass_info* dpmi,
	double4 *pos, double3 *vel, double3 *omega, double3 *force, double3 *moment,
	double* mass, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, 
	device_contact_property *cp, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//unsigned int np = _np;
	if (id >= cte.np)
		return;

	double cdist = 0.0;
	double im = mass[id];
	double3 ipos = make_double3(pos[id].x, pos[id].y, pos[id].z);
	double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
	double3 iomega = make_double3(omega[id].x, omega[id].y, omega[id].z);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	//double3 cpt = make_double3(0.0, 0.0, 0.0);
	//double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	double ir = pos[id].w;
	double3 M = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (k >= cte.np)
						{
							k -= cte.np;
							//printf("%d", k);
							double3 distVec;
							double dist;
							unsigned int pidx = dpi[k].id;
							device_contact_property cmp = cp[pidx];
							device_polygon_mass_info pmi = dpmi[pidx];
							double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir);
							double3 po2cp = cpt - pmi.origin;
							//double3 si = cpt - pmi.origin;
							distVec = ipos - cpt;
							dist = length(distVec);
							cdist = ir - dist;
							Fn = make_double3(0.0, 0.0, 0.0);
							if (cdist > 0)
							{
								double3 qp = dpi[k].Q - dpi[k].P;
								double3 rp = dpi[k].R - dpi[k].P;
								double rcon = ir - 0.5 * cdist;
								unit = -cross(qp, rp);// -dpi[k].N;
								unit = unit / length(unit);
								double3 dv = pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, ir * unit));
								device_force_constant c = getConstant(
									TCM, ir, 0, im, 0, cmp.Ei, cmp.Ej,
									cmp.pri, cmp.prj, cmp.Gi, cmp.Gj,
									cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
								switch (TCM)
								{
								case 0: 
									HMCModel(
										c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
										dv, unit, Ft, Fn, M); 
									break;
								case 1:
									DHSModel(
										c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
										dv, unit, Ft, Fn, M);
									break;
								}
								double3 sum_f = Fn;// +shear_force;
								force[id] += make_double3(sum_f.x, sum_f.y, sum_f.z);
								moment[id] += make_double3(M.x, M.y, M.z);
								dpmi[pidx].force += -(sum_f + Ft);// +make_double3(1.0, 5.0, 9.0);
								dpmi[pidx].moment += -cross(po2cp, sum_f + Ft);
							}			
						}
					}
				}
			}
		}
	}
}

// float
__device__
int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - cte_f.world_origin.x) / cte_f.cell_size);
	gridPos.y = floor((p.y - cte_f.world_origin.y) / cte_f.cell_size);
	gridPos.z = floor((p.z - cte_f.world_origin.z) / cte_f.cell_size);
	return gridPos;
}

__global__ void vv_update_position_kernel(float4* pos, float3* vel, float3* acc, unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;

	float3 _p = cte_f.dt * vel[id] + cte_f.half2dt * acc[id];
	pos[id].x += _p.x;
	pos[id].y += _p.y;
	pos[id].z += _p.z;

}

__global__ void vv_update_velocity_kernel(
	float3* vel,
	float3* acc,
	float3* omega,
	float3* alpha,
	float3* force,
	float3* moment,
	float* mass,
	float* iner,
	unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	float3 v = vel[id];
	float3 L = acc[id];
	float3 av = omega[id];
	float3 aa = alpha[id];
	float m = mass[id];
	float in = iner[id];
	v += 0.5f * cte_f.dt * L;
	av += 0.5f * cte_f.dt * aa;
	L = (1.0f / m) * force[id];
	aa = (1.0f / in) * moment[id];
	v += 0.5f * cte_f.dt * L;
	av += 0.5f * cte_f.dt * aa;
	// 	if(id == 0){
	// 		printf("Velocity --- > id = %d -> [%f.6, %f.6, %f.6]\n", id, v.x, v.y, v.z);
	// 	}
	force[id] = m * cte_f.gravity;
	moment[id] = make_float3(0.0f, 0.0f, 0.0f);
	vel[id] = v;
	omega[id] = av;
	acc[id] = L;
	alpha[id] = aa;
}


__global__ void calculateHashAndIndex_kernel(
	unsigned int* hash, unsigned int* index, float4* pos, unsigned int np)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (np)) return;
	volatile float4 p = pos[id];

	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	unsigned _hash = calcGridHash(gridPos);
	/*if(_hash >= cte_f.ncell)
	printf("Over limit - hash number : %d", _hash);*/
	hash[id] = _hash;
	index[id] = id;
}

__global__ void calculateHashAndIndexForPolygonSphere_kernel(
	unsigned int* hash, unsigned int* index, 
	unsigned int sid, unsigned int nsphere, float4* sphere)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= nsphere) return;
	volatile float4 p = sphere[id];
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	unsigned int _hash = calcGridHash(gridPos);
	hash[sid + id] = _hash;
	index[sid + id] = sid + id;
}

__global__ void reorderDataAndFindCellStart_kernel_f(
	unsigned int* hash,
	unsigned int* index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int* sorted_index,
	unsigned int np)
{
	extern __shared__ uint sharedHash[];	//blockSize + 1 elements
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned _hash;

	//unsigned int tnp = ;// cte_f.np + cte_f.nsphere;

	if (id < np)
	{
		_hash = hash[id];

		sharedHash[threadIdx.x + 1] = _hash;

		if (id > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[id - 1];
		}
	}
	__syncthreads();

	if (id < np)
	{
		if (id == 0 || _hash != sharedHash[threadIdx.x])
		{
			cstart[_hash] = id;

			if (id > 0)
				cend[sharedHash[threadIdx.x]] = id;
		}

		if (id == np - 1)
		{
			cend[_hash] = id + 1;
		}

		unsigned int sortedIndex = index[id];
		sorted_index[id] = sortedIndex;
	}
}

__device__ device_force_constant_f getConstant(
	int tcm, float ir, float jr, float im, float jm,
	float iE, float jE, float ip, float jp,
	float iG, float jG, float rest,
	float fric, float rfric, float sratio)
{
	device_force_constant_f dfc = { 0, 0, 0, 0, 0, 0 };
	float Meq = jm ? (im * jm) / (im + jm) : im;
	float Req = jr ? (ir * jr) / (ir + jr) : ir;
	float Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	switch (tcm)
	{
	case 0:{
		float Geq = (iG * jG) / (iG*(2.0f - jp) + jG*(2.0f - ip));
		float ln_e = log(rest);
		float xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
		dfc.kn = (4.0f / 3.0f) * Eeq * sqrt(Req);
		dfc.vn = -2.0f * sqrt(5.0f / 6.0f) * xi * sqrt(dfc.kn * Meq);
		dfc.ks = 8.0f * Geq * sqrt(Req);
		dfc.vs = -2.0f * sqrt(5.0f / 6.0f) * xi * sqrt(dfc.ks * Meq);
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	case 1:{
		float beta = (M_PI / log(rest));
		dfc.kn = (4.0f / 3.0f) * Eeq * sqrt(Req);
		dfc.vn = sqrt((4.0 * Meq * dfc.kn) / (1.0 + beta * beta));
		dfc.ks = dfc.kn * sratio;
		dfc.vs = dfc.vn * sratio;
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	}

	// 	dfc.kn = /*(16.f / 15.f)*sqrt(er) * eym * pow((T)((15.f * em * 1.0f) / (16.f * sqrt(er) * eym)), (T)0.2f);*/ (4.0f / 3.0f)*sqrt(er)*eym;
	// 	dfc.vn = sqrt((4.0f*em * dfc.kn) / (1 + beta * beta));
	// 	dfc.ks = dfc.kn * ratio;
	// 	dfc.vs = dfc.vn * ratio;
	// 	dfc.mu = fric;
	return dfc;
}

// ref. Three-dimensional discrete element modelling (DEM) of tillage: Accounting for soil cohesion and adhesion
__device__ float cohesionForce(
	float ri,
	float rj,
	float Ei,
	float Ej,
	float pri,
	float prj,
	float coh,
	float Fn)
{
	float cf = 0.f;
	if (coh){
		float req = (ri * rj / (ri + rj));
		float Eeq = ((1.0f - pri * pri) / Ei) + ((1.0f - prj * prj) / Ej);
		float rcp = (3.0f * req * (-Fn)) / (4.0f * (1.0f / Eeq));
		float rc = pow(rcp, 1.0f / 3.0f);
		float Ac = M_PI * rc * rc;
		cf = coh * Ac;
	}
	return cf;
}

__device__ void HMCModel(
	device_force_constant_f c, float ir, float jr, float Ei, float Ej, float pri, float prj, float coh,
	float rcon, float cdist, float3 iomega,
	float3 dv, float3 unit, float3& Ft, float3& Fn, float3& M)
{
	float fsn = -c.kn * pow(cdist, 1.5f);
	float fdn = c.vn * dot(dv, unit);
	float fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	Fn = (fsn + fca + fdn) * unit;
	float3 e = dv - dot(dv, unit) * unit;
	float mag_e = length(e);
	if (mag_e){
		float3 s_hat = -(e / mag_e);
		float ds = mag_e * cte_f.dt;
		float fst = -c.ks * ds;
		float fdt = c.vs * dot(dv, s_hat);
		Ft = (fst + fdt) * s_hat;
		if (length(Ft) >= c.mu * length(Fn))
			Ft = c.mu * fsn * s_hat;
		M = cross(ir * unit, Ft);
		if (length(iomega)){
			float3 on = iomega / length(iomega);
			M += c.ms * fsn * rcon * on;
		}
	}
}

__device__ void DHSModel(
	device_force_constant_f c, float ir, float jr, float Ei, float Ej, float pri, float prj, float coh,
	float rcon, float cdist, float3 iomega,
	float3 dv, float3 unit, float3& Ft, float3& Fn, float3& M)
{
	float fsn = -c.kn * pow(cdist, 1.5f);
	float fdn = c.vn * dot(dv, unit);
	float fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	Fn = (fsn + fca + fdn) * unit;
	float3 e = dv - dot(dv, unit) * unit;
	float mag_e = length(e);
	if (mag_e)
	{
		float3 sh = e / mag_e;
		float ds = mag_e * cte_f.dt;
		Ft = min(c.ks * ds + c.vs * (dot(dv, sh)), c.mu * length(Fn)) * sh;
		M = cross(ir * unit, Ft);
	}
}

template <int TCM>
__global__ void calculate_p2p_kernel(
	float4* pos, float3* vel,
	float3* omega, float3* force,
	float3* moment, float* mass,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property_f* cp, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;

	float4 ipos = pos[id];
	float4 jpos = make_float4(0, 0, 0, 0);
	float3 ivel = vel[id];
	float3 jvel = make_float3(0, 0, 0);
	float3 iomega = omega[id];
	float3 jomega = make_float3(0, 0, 0);
	int3 gridPos = calcGridPos(make_float3(ipos.x, ipos.y, ipos.z));

	float ir = ipos.w; float jr = 0;
	float im = mass[id]; float jm = 0;
	float3 Ft = make_float3(0, 0, 0);
	float3 Fn = make_float3(0, 0, 0);// [id] * cte_f.gravity;
	float3 M = make_float3(0, 0, 0);
	float3 sumF = make_float3(0, 0, 0);
	float3 sumM = make_float3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)
							continue;
						jpos = pos[k]; jvel = vel[k]; jomega = omega[k];
						jr = jpos.w; jm = mass[k];
						float3 rp = make_float3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						float dist = length(rp);
						float cdist = (ir + jr) - dist;
						if (cdist > 0){
							float rcon = ir - 0.5f * cdist;
							float3 unit = rp / dist;
							float3 rv = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
							device_force_constant_f c = getConstant(
								TCM, ir, jr, im, jm, cp->Ei, cp->Ej,
								cp->pri, cp->prj, cp->Gi, cp->Gj,
								cp->rest, cp->fric, cp->rfric, cp->sratio);
							switch (TCM)
							{
							case 0:
								HMCModel(
									c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega,
									rv, unit, Ft, Fn, M);
								break;
							case 1:
								DHSModel(
									c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega,
									rv, unit, Ft, Fn, M);
								break;
							}
							sumF += Fn + Ft;
							sumM += M;
						}
					}
				}
			}
		}
	}
	force[id] += sumF;
	moment[id] += sumM;
}

__device__ float particle_plane_contact_detection(
	device_plane_info_f *pe, float3& xp, float3& wp, float3& u, float r)
{
	float a_l1 = pow(wp.x - pe->l1, 2.0f);
	float b_l2 = pow(wp.y - pe->l2, 2.0f);
	float sqa = wp.x * wp.x;
	float sqb = wp.y * wp.y;
	float sqc = wp.z * wp.z;
	float sqr = r*r;

	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe->l1) && (wp.y > 0 && wp.y < pe->l2)){
		float3 dp = xp - pe->xw;
		float3 uu = pe->uw / length(pe->uw);
		int pp = -sign(dot(dp, pe->uw));// dp.dot(pe->UW()));
		u = pp * uu;
		float collid_dist = r - abs(dot(dp, u));// dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		float3 Xsw = xp - pe->xw;
		float h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		float3 Xsw = xp - pe->w2;
		float h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->l1 && wp.y > pe->l2 && (a_l1 + b_l2 + sqc) < sqr){
		float3 Xsw = xp - pe->w3;
		float h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe->l2 && (sqa + b_l2 + sqc) < sqr){
		float3 Xsw = xp - pe->w4;
		float h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe->l1) && wp.y < 0 && (sqb + sqc) < sqr){
		float3 Xsw = xp - pe->xw;
		float3 wj_wi = pe->w2 - pe->xw;
		float3 us = wj_wi / length(wj_wi);// .length();
		float3 h_star = Xsw - (dot(Xsw, us)) * us;
		float h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe->l1) && wp.y > pe->l2 && (b_l2 + sqc) < sqr){
		float3 Xsw = xp - pe->w4;
		float3 wj_wi = pe->w3 - pe->w4;
		float3 us = wj_wi / length(wj_wi);// .length();
		float3 h_star = Xsw - (dot(Xsw, us)) * us;
		float h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x < 0 && (sqr + sqc) < sqr){
		float3 Xsw = xp - pe->xw;
		float3 wj_wi = pe->w4 - pe->xw;
		float3 us = wj_wi / length(wj_wi);// .length();
		float3 h_star = Xsw - (dot(Xsw, us)) * us;
		float h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x > pe->l1 && (a_l1 + sqc) < sqr){
		float3 Xsw = xp - pe->w2;
		float3 wj_wi = pe->w3 - pe->w2;
		float3 us = wj_wi / length(wj_wi);// .length();
		float3 h_star = Xsw - (dot(Xsw, us)) * us;
		float h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}


	return -1.0f;
}

template <int TCM>
__global__ void plane_contact_force_kernel(
	device_plane_info_f *plane,
	float4* pos, float3* vel, float3* omega,
	float3* force, float3* moment,
	device_contact_property_f *cp, float* mass, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	float m = mass[id];
	float4 ipos = pos[id];
	float3 ipos3 = make_float3(ipos.x, ipos.y, ipos.z);
	float r = ipos.w;
	float3 ivel = vel[id];
	float3 iomega = omega[id];

	float3 Fn = make_float3(0, 0, 0);
	float3 Ft = make_float3(0, 0, 0);
	float3 M = make_float3(0, 0, 0);
	float3 dp = make_float3(ipos.x, ipos.y, ipos.z) - plane->xw;
	float3 unit = make_float3(0, 0, 0);
	float3 wp = make_float3(dot(dp, plane->u1), dot(dp, plane->u2), dot(dp, plane->uw));

	float cdist = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
	if (cdist > 0){
		float rcon = r - 0.5f * cdist;
		float3 dv = -(ivel + cross(iomega, r * unit));
		device_force_constant_f c = getConstant(
			TCM, r, 0.f, m, 0.f, cp->Ei, cp->Ej,
			cp->pri, cp->prj, cp->Gi, cp->Gj,
			cp->rest, cp->fric, cp->rfric, cp->sratio);
		switch (TCM)
		{
		case 0:
			HMCModel(
				c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
				iomega, dv, unit, Ft, Fn, M);
			break;
		case 1:
			DHSModel(
				c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
				iomega, dv, unit, Ft, Fn, M);
			break;
		}
	}
	force[id] += Fn + Ft;// m_force + shear_force;
	moment[id] += M;
}


__device__ float3 makeTFM_1(float4& ep)
{
	return make_float3(2.0f * (ep.x*ep.x + ep.y*ep.y - 0.5f), 2.0f * (ep.y*ep.z - ep.x*ep.w), 2.0f * (ep.y*ep.w + ep.x*ep.z));
}

__device__ float3 makeTFM_2(float4& ep)
{
	return make_float3(2.0f * (ep.y*ep.z + ep.x*ep.w), 2.0f * (ep.x*ep.x + ep.z*ep.z - 0.5f), 2.0f * (ep.z*ep.w - ep.x*ep.y));
}

__device__ float3 makeTFM_3(float4& ep)
{
	return make_float3(2.0f * (ep.y*ep.w - ep.x*ep.z), 2.0f * (ep.z*ep.w + ep.x*ep.y), 2.0f * (ep.x*ep.x + ep.w*ep.w - 0.5f));
}

__device__ float3 toLocal(float3& A_1, float3& A_2, float3& A_3, float3& v)
{
	return make_float3
		(A_1.x * v.x + A_2.x * v.y + A_3.x * v.z,
		A_1.y * v.x + A_2.y * v.y + A_3.y * v.z,
		A_1.z * v.x + A_2.z * v.y + A_3.z * v.z);
}

__device__ float3 toGlobal(float3& A_1, float3& A_2, float3& A_3, float3& v)
{
	return make_float3(dot(A_1, v), dot(A_2, v), dot(A_3, v));
}

__device__ float particle_cylinder_contact_detection(
	device_cylinder_info_f* cy, float4& pt, float3& u, float3& cp, unsigned int id = 0)
{
	float dist = -1.0f;
	float3 ab = make_float3(cy->ptop.x - cy->pbase.x, cy->ptop.y - cy->pbase.y, cy->ptop.z - cy->pbase.z);
	float3 p = make_float3(pt.x, pt.y, pt.z);
	float t = dot(p - cy->pbase, ab) / dot(ab, ab);
	float3 _cp = make_float3(0.0f, 0.0f, 0.0f);
	if (t >= 0 && t <= 1){
		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		u = (_cp - p) / dist;
		cp = _cp - cy->rbase * u;
		return cy->rtop + pt.w - dist;
	}
	else{

		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		if (dist < cy->rbase){
			float3 OtoCp = cy->origin - _cp;
			float OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
			cp = _cp - cy->rbase * u;
			return cy->len * 0.5f + pt.w - OtoCp_;
		}
		float3 A_1 = makeTFM_1(cy->ep);
		float3 A_2 = makeTFM_2(cy->ep);
		float3 A_3 = makeTFM_3(cy->ep);
		float3 _at = p - cy->ptop;
		float3 at = toLocal(A_1, A_2, A_3, _at);
		float r = length(at);
		cp = cy->ptop;
		if (abs(at.y) > cy->len){
			_at = p - cy->pbase;
			at = toLocal(A_1, A_2, A_3, _at);
			cp = cy->pbase;
		}
		float pi = atan(at.x / at.z);
		if (pi < 0 && at.z < 0){
			_cp.x = cy->rbase * sin(-pi);
		}
		else if (pi > 0 && at.x < 0 && at.z < 0){
			_cp.x = cy->rbase * sin(-pi);
		}
		else{
			_cp.x = cy->rbase * sin(pi);
		}
		_cp.z = cy->rbase * cos(pi);
		if (at.z < 0 && _cp.z > 0){
			_cp.z = -_cp.z;
		}
		else if (at.z > 0 && _cp.z < 0){
			_cp.z = -_cp.z;
		}
		_cp.y = 0.;
		cp = cp + toGlobal(A_1, A_2, A_3, _cp);

		float3 disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < pt.w){
			return pt.w - dist;
		}
	}
	return -1.0f;
}

template<int TCM>
__global__ void cylinder_hertzian_contact_force_kernel(
	device_cylinder_info_f *cy,
	float4* pos, float3* vel, float3* omega,
	float3* force, float3* moment, device_contact_property_f *cp,
	float* mass, float3* mpos, float3* mf, float3* mm, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;

	*mf = make_float3(0.0f, 0.0f, 0.0f);
	*mm = make_float3(0.0f, 0.0f, 0.0f);
	float cdist = 0.0f;
	float im = mass[id];
	float4 ipos = make_float4(pos[id].x, pos[id].y, pos[id].z, pos[id].w);
	float3 ivel = make_float3(vel[id].x, vel[id].y, vel[id].z);
	float3 iomega = make_float3(omega[id].x, omega[id].y, omega[id].z);
	float3 unit = make_float3(0.0f, 0.0f, 0.0f);
	float3 cpt = make_float3(0.0f, 0.0f, 0.0f);
	float3 mp = make_float3(mpos->x, mpos->y, mpos->z);
	cdist = particle_cylinder_contact_detection(cy, ipos, unit, cpt, id);
	float3 si = cpt - mp;
	float3 cy2cp = cpt - cy->origin;
	float3 Ft = make_float3(0.0f, 0.0f, 0.0f);
	float3 Fn = make_float3(0.0f, 0.0f, 0.0f);
	float3 M = make_float3(0.0f, 0.0f, 0.0f);
	if (cdist > 0)
	{
		float rcon = ipos.w - 0.5f * cdist;
		float3 dv = cy->vel + cross(cy->omega, cy2cp) - (ivel + cross(iomega, ipos.w * unit));
		device_force_constant_f c = getConstant(
			TCM, ipos.w, 0, im, 0, cp->Ei, cp->Ej,
			cp->pri, cp->prj, cp->Gi, cp->Gj,
			cp->rest, cp->fric, cp->rfric, cp->sratio);
		switch (TCM)
		{
		case 0: HMCModel(
			c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
			dv, unit, Ft, Fn, M);
			break;
		case 1:
			DHSModel(
				c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
				dv, unit, Ft, Fn, M);
			break;
		}
	}
	float3 sum_f = Fn + Ft;
	force[id] += make_float3(sum_f.x, sum_f.y, sum_f.z);
	moment[id] += make_float3(M.x, M.y, M.z);
	mf[id] = -(Fn);
	mm[id] = cross(si, -Fn);
}

__device__ float3 closestPtPointTriangle(
	device_polygon_info_f& dpi,
	float3& p,
	float pr)
{
	float3 a = dpi.P;
	float3 b = dpi.Q;
	float3 c = dpi.R;
	float3 ab = b - a;
	float3 ac = c - a;
	float3 ap = p - a;

	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f){
		//	*wc = 0;
		return a;
	}

	float3 bp = p - b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3){
		//	*wc = 0;
		return b;
	}
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f){
		//	*wc = 1;
		float v = d1 / (d1 - d3);
		return a + v * ab;
	}

	float3 cp = p - c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6){
		//	*wc = 0;
		return c;
	}

	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f){
		//	*wc = 1;
		float w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f){
		//	*wc = 1;
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	//*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;

	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

__device__ bool checkConcave(device_polygon_info_f* dpi, unsigned int* tid, unsigned int kid, float4* dsph, unsigned int cnt)
{
	float3 p1 = make_float3(0, 0, 0);
	float3 u1 = make_float3(0, 0, 0);
	float3 p2 = make_float3(dsph[kid].x, dsph[kid].y, dsph[kid].z);
	float3 u2 = dpi[kid].N;
	for (unsigned int i = 0; i < cnt; i++){
		unsigned int id = tid[i];
		p1 = make_float3(dsph[id].x, dsph[id].y, dsph[id].z);
		u1 = dpi[id].N;
		float3 p2p1 = p2 - p1;
		float chk1 = dot(p2p1, u1);
		float chk2 = dot(-p2p1, u2);
		if (chk1 > 0 && chk2 > 0)
		{
			tid[cnt++] = id;
			return true;
		}
	}
	return false;
}

template<int TCM>
__global__ void particle_polygonObject_collision_kernel(
	device_polygon_info_f* dpi, float4* dsph, device_polygon_mass_info_f* dpmi,
	float4 *pos, float3 *vel, float3 *omega, float3 *force, float3 *moment,
	float* mass, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	device_contact_property_f *cp, unsigned int _np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;

	float cdist = 0.0f;
	float im = mass[id];
	float3 ipos = make_float3(pos[id].x, pos[id].y, pos[id].z);
	float3 ivel = make_float3(vel[id].x, vel[id].y, vel[id].z);
	float3 iomega = make_float3(omega[id].x, omega[id].y, omega[id].z);
	float3 unit = make_float3(0.0f, 0.0f, 0.0f);
	int3 gridPos = calcGridPos(make_float3(ipos.x, ipos.y, ipos.z));
	//float3 cpt = make_float3(0.0f, 0.0f, 0.0f);
	//float3 mp = make_float3(mpos->x, mpos->y, mpos->z);
	float ir = pos[id].w;
	float3 M = make_float3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	float3 Fn = make_float3(0, 0, 0);
	float3 Ft = make_float3(0, 0, 0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (k >= np)
						{
							k -= np;
							float3 distVec;
							float dist;
							unsigned int pidx = dpi[k].id;
							device_contact_property_f cmp = cp[pidx];
							device_polygon_mass_info_f pmi = dpmi[pidx];
							float3 cpt = closestPtPointTriangle(dpi[k], ipos, ir);
							float3 po2cp = cpt - pmi.origin;
							//float3 si = cpt - pmi.origin;
							distVec = ipos - cpt;
							dist = length(distVec);
							cdist = ir - dist;
							Fn = make_float3(0.0f, 0.0f, 0.0f);
							if (cdist > 0)
							{
								float3 qp = dpi[k].Q - dpi[k].P;
								float3 rp = dpi[k].R - dpi[k].P;
								float rcon = ir - 0.5 * cdist;
								unit = -cross(qp, rp);// -dpi[k].N;
								unit = unit / length(unit);
								float3 dv = pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, ir * unit));
								device_force_constant_f c = getConstant(
									TCM, ir, 0, im, 0, cmp.Ei, cmp.Ej,
									cmp.pri, cmp.prj, cmp.Gi, cmp.Gj,
									cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
								switch (TCM)
								{
								case 0:
									HMCModel(
										c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
										dv, unit, Ft, Fn, M);
									break;
								case 1:
									DHSModel(
										c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
										dv, unit, Ft, Fn, M);
									break;
								}
								float3 sum_f = Fn;// +shear_force;
								force[id] += make_float3(sum_f.x, sum_f.y, sum_f.z);
								moment[id] += make_float3(M.x, M.y, M.z);
								dpmi[pidx].force += -(sum_f + Ft);// +make_float3(1.0f, 5.0f, 9.0f);
								dpmi[pidx].moment += -cross(po2cp, sum_f + Ft);
							}
						}
					}
				}
			}
		}
	}
}