#include "mphysics_cuda_dec.cuh"
#include <helper_math.h>

__constant__ device_parameters cte;

inline __device__ int sign(float L)
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
int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - cte.world_origin.x) / cte.cell_size);
	gridPos.y = floor((p.y - cte.world_origin.y) / cte.cell_size);
	gridPos.z = floor((p.z - cte.world_origin.z) / cte.cell_size);
	return gridPos;
}

__global__ void vv_update_position_kernel(float4* pos, float3* vel, float3* acc)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= cte.np)
		return;
	/*double3 p = make_double3(pos[id]);
	double3 v = make_double3(vel[id]);
	double3 a = make_double3(acc[id]);*/

	float3 _p = cte.dt * vel[id] + cte.half2dt * acc[id];
	pos[id] += make_float4(_p, 0.f);
	// 	if(id == 31914){
	// 		p = make_double3(pos[id]);
	// 	}
	//pos[id] = make_double4(p.x, p.y, p.z, pos[id].w);
}

__global__ void vv_update_velocity_kernel(
	float3* vel,
	float3* acc,
	float3* omega,
	float3* alpha,
	float3* force,
	float3* moment,
	float* mass,
	float* iner)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= cte.np)
		return;
	float3 v = vel[id];
	float3 L = acc[id];
	float3 av = omega[id];
	float3 aa = alpha[id];
	float m = mass[id];
	float in = iner[id];

	v += 0.5f * cte.dt * L;
	av += 0.5f * cte.dt * aa;
	L = (1.f / m) * force[id];
	aa = (1.f / in) * moment[id];
	v += 0.5f * cte.dt * L;
	av += 0.5f * cte.dt * aa;
	// 	if(id == 0){
	// 		printf("Velocity --- > id = %d -> [%f.6, %f.6, %f.6]\n", id, v.x, v.y, v.z);
	// 	}
	vel[id] = v;
	omega[id] = av;
	acc[id] = L;
	alpha[id] = av;
}


__global__ void calculateHashAndIndex_kernel(unsigned int* hash, unsigned int* index, float4* pos)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (cte.np)) return;
	volatile float4 p = pos[id];

	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	unsigned _hash = calcGridHash(gridPos);
	/*if(_hash >= cte.ncell)
	printf("Over limit - hash number : %d", _hash);*/
	hash[id] = _hash;
	index[id] = id;
}

__global__ void calculateHashAndIndexForPolygonSphere_kernel(unsigned int* hash, unsigned int* index, unsigned int sid, unsigned int nsphere, double4* sphere)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= nsphere) return;
	volatile double4 p = sphere[id];
	int3 gridPos = calcGridPos(make_float3((float)p.x, (float)p.y, (float)p.z));
	unsigned int _hash = calcGridHash(gridPos);
	hash[sid + id] = _hash;
	index[sid + id] = sid + id;
}

__global__ void reorderDataAndFindCellStart_kernel(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index)
{
	extern __shared__ uint sharedHash[];	//blockSize + 1 elements
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned _hash;

	unsigned int tnp = cte.np + cte.nsphere;

	if (id < tnp)
	{
		_hash = hash[id];

		sharedHash[threadIdx.x + 1] = _hash;

		if (id > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[id - 1];
		}
	}
	__syncthreads();

	if (id < tnp)
	{
		if (id == 0 || _hash != sharedHash[threadIdx.x])
		{
			cstart[_hash] = id;

			if (id > 0)
				cend[sharedHash[threadIdx.x]] = id;
		}

		if (id == tnp - 1)
		{
			cend[_hash] = id + 1;
		}

		unsigned int sortedIndex = index[id];
		sorted_index[id] = sortedIndex;
	}
}

template<typename T>
__device__ device_force_constant<T> getConstant(
	T ir,
	T jr,
	T im,
	T jm,
	T iE,
	T jE,
	T ip,
	T jp,
	T rest,
	T ratio,
	T fric
	/*T riv*/)
{
	device_force_constant<T> dfc = { 0, 0, 0, 0, 0 };
	T em = jm ? (im * jm) / (im + jm) : im;
	T er = jr ? (ir * jr) / (ir + jr) : ir;
	T eym = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	T beta = ((T)M_PI / log(rest));
	dfc.kn = /*(16.f / 15.f)*sqrt(er) * eym * pow((T)((15.f * em * 1.0f) / (16.f * sqrt(er) * eym)), (T)0.2f);*/ (4.0f / 3.0f)*sqrt(er)*eym;
	dfc.vn = sqrt((4.0f*em * dfc.kn) / (1 + beta * beta));
	dfc.ks = dfc.kn * ratio;
	dfc.vs = dfc.vn * ratio;
	dfc.mu = fric;
	return dfc;
}

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
		float Eeq_inv = ((1 - pri * pri) / Ei) + ((1 - prj * prj) / Ej);
		float rcp = (3.f * req * (-Fn)) / (4.f * (1 / Eeq_inv));
		float rc = pow(rcp, 1.0f / 3.0f);
		float Ac = M_PI * rc * rc;
		cf = coh * Ac;
	}
	return cf;
}

__device__ bool calForce(
	float ir,
	float jr,
	float im,
	float jm,
	float rest,
	float ratio,
	float fric,
	float E,
	float pr,
	float coh,
	float4 ipos,
	float4 jpos,
	float3 ivel,
	float3 jvel,
	float3 iomega,
	float3 jomega,
	float3& force,
	float3& moment
	/*float *riv*/)
{
	float3 relative_pos = make_float3(jpos - ipos);
	float dist = length(relative_pos);
	float collid_dist = (ir + jr) - dist;
	float3 shear_force = make_float3(0.f);
	if (collid_dist <= 0){
		//*riv = 0.f;
		return false;
	}
	else{
		float3 unit = relative_pos / dist;
		float3 relative_vel = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
		//*riv = abs(length(relative_vel));
		device_force_constant<float> c = getConstant<float>(ir, jr, im, jm, E, E, pr, pr, rest, ratio, fric/*, *riv*/);
		float fsn = -c.kn * pow(collid_dist, 1.5f);
		float fca = cohesionForce(ir, jr, E, E, pr, pr, coh, fsn);
		float fsd = c.vn * dot(relative_vel, unit);
		float3 single_force = (fsn + fca + fsd) * unit;
		//float3 single_force = (-c.kn * pow(collid_dist, 1.5f) + c.vn * dot(relative_vel, unit)) * unit;
		float3 single_moment = make_float3(0, 0, 0);
		float3 e = relative_vel - dot(relative_vel, unit) * unit;
		float mag_e = length(e);
		if (mag_e){
			float3 s_hat = e / mag_e;
			float ds = mag_e * cte.dt;
			shear_force = min(c.ks * ds + c.vs * (dot(relative_vel, s_hat)), c.mu * length(single_force)) * s_hat;
			single_moment = cross(ir * unit, shear_force);
		}
		force += single_force + shear_force;
		moment += single_moment;
	}
	
	return true;
}

__global__ void calculate_p2p_kernel(
	float4* pos,
	float3* vel,
	float3* acc,
	float3* omega,
	float3* alpha,
	float3* force,
	float3* moment,
	//float* rad,
	float* mass,
	float* iner,
	float* riv,
	float E,
	float pr,
	float rest,
	float ratio,
	float fric,
	float coh,
	unsigned int* sorted_index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int cRun = 0)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= cte.np)
		return;

	float4 ipos = pos[id];
	float4 jpos = make_float4(0, 0, 0, 0);
	float3 ivel = vel[id];
	float3 jvel = make_float3(0, 0, 0);
	float3 iomega = omega[id];
	float3 jomega = make_float3(0, 0, 0);
	int3 gridPos = calcGridPos(make_float3(ipos));

	float ir = ipos.w;
	float jr = 0;
	float im = mass[id];
	float jm = 0;
	float3 m_force = mass[id] * cte.gravity;
	float3 m_moment = make_float3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	//device_force_constant dfc = { 0, 0, 0, 0, 0 };
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	// 	if(id == 31914){
	// 		end_index = 0;
	// 	}
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
						if (id == k || k >= cte.np)
							continue;
						jpos = pos[k];
						jvel = vel[k];
						jomega = omega[k];
						jr = jpos.w;
						jm = mass[k];
						/*dfc = getConstant(ir, jr, im, jm, E, E, pr, pr, rest, ratio, fric);*/
						//unsigned int rid = id > k ? (id * 10 + k) : (k * 10 + id);
						if (!calForce(ir, jr, im, jm, rest, ratio, fric, E, pr, coh, ipos, jpos, ivel, jvel, iomega, jomega, m_force, m_moment/*, &(riv[rid])*/))
							continue;
					}
				}
			}
		}
	}
	//force[cte.np] = make_double3(0, 0, 0);
	force[id] = m_force;
// 	if (id == 0){
// 		id = 0;
// 	}
	// 	if(id == 60775){
	// 		printf("id = %d -> [%f.6, %f.6, %f.6]\n", id, force[id].x, force[id].y, force[id].z);
	// 	}
	moment[id] = m_moment;
}

__device__ float particle_plane_contact_detection(device_plane_info *pe, float3& xp, float3& wp, float3& u, float r)
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
		float3 h_star = Xsw - (dot(Xsw,us)) * us;
		float h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x > pe->l1 && (a_l1 + sqc) < sqr){
		float3 Xsw = xp - pe->w2;
		float3 wj_wi = pe->w3 - pe->w2;
		float3 us = wj_wi / length(wj_wi);// .length();
		float3 h_star = Xsw - (dot(Xsw,us)) * us;
		float h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}


	return -1.0f;
}

__global__ void plane_hertzian_contact_force_kernel(
	device_plane_info *plane,
	float E,
	float pr,
	float rest,
	float ratio,
	float fric,
	float4* pos,
	float3* vel,
	float3* omega,
	float3* force,
	float3* moment,
	//float* rad,
	float* mass,
	float* riv,
	float pE,
	float pPr)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= cte.np)
		return;

	//float r = rad[id];
	float m = mass[id];
	float4 ipos = pos[id];
	float3 ipos3 = make_float3(ipos);
	float r = ipos.w;
	float3 ivel = vel[id];
	float3 iomega = omega[id];

	float3 single_force = make_float3(0, 0, 0);
	float3 m_force = make_float3(0, 0, 0);
	float3 m_moment = make_float3(0, 0, 0);
	//for (int i = 0; i < 6; i++){
		//device_plane_info plane = planes[i];
	
	float3 dp = make_float3(ipos) - plane->xw;
	float3 shear_force = make_float3(0.f);
	float3 unit = make_float3(0.0f);
	float3 wp = make_float3(dot(dp, plane->u1), dot(dp, plane->u2), dot(dp, plane->uw));
	
	float collid_dist = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
	/*if (abs(wp.z) < r && (wp.x > 0 && wp.x < plane->l1) && (wp.y > 0 && wp.y < plane->l2)){
		float3 unit = -sign(dot(ipos - plane->xw, plane->uw)) * (plane->uw / length(plane->uw));
		float collid_dist = r - abs(dot(ipos - plane->xw, unit));*/
	if (collid_dist > 0){
		//1float collid_dist2 = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
		float3 dv = -(ivel + cross(iomega, r * unit));
// 		if (!riv[id])
// 			riv[id] = abs(length(dv));
		device_force_constant<float> c = getConstant<float>(r, 0.f, m, 0.f, pE, E, pPr, pr, rest, ratio, fric/*, riv[id]*/);
		float fsn = -c.kn * pow(collid_dist, 1.5f);
		single_force = (fsn + c.vn * dot(dv, unit)) * unit;
		//float fca = cohesionForce(r, 0.f, pE, E, pPr, pr, fsn);
		//single_force += fca * unit;
		float3 e = dv - dot(dv, unit) * unit;
		float mag_e = length(e);
		if (mag_e){
			float3 s_hat = e / mag_e;
			float ds = mag_e * cte.dt;
			shear_force = min(c.ks * ds + c.vs * dot(dv, s_hat), c.mu * length(single_force)) * s_hat;
			m_moment += cross(r * unit, shear_force);
		}
		m_force += single_force;
	}
	else{
		riv[id] = 0.f;
	}
	//}
	force[id] += m_force + shear_force;
	moment[id] += m_moment;
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
	//double3 p_pbase = make_double3(p.x - cy->pbase.x, p.y - cy->pbase.y, p.z - cy->pbase.z);
	double t = dot(p - cy->pbase, ab) / dot(ab, ab);
	double3 _cp = make_double3(0.0, 0.0, 0.0);
	//float th = 0.f;
	//float n;
	//n = dot(cy->ep, cy->ep);
	//float3 cp = make_float3(0.f);
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
		//float pi = 0.f;
		//float r = 0.f;
		double3 A_1 = makeTFM_1(cy->ep);
		double3 A_2 = makeTFM_2(cy->ep);
		double3 A_3 = makeTFM_3(cy->ep);
		double3 _at = p - cy->ptop;
		double3 at = toLocal(A_1, A_2, A_3, _at);
		double r = length(at);
		cp = cy->ptop;
// 		th = acos(at.y / r);
// 		pi = atan(at.x / at.z);
		if (abs(at.y) > cy->len){
			_at = p - cy->pbase;
			at = toLocal(A_1, A_2, A_3, _at);
			cp = cy->pbase;
/*			th = acos(-at.y / r);*/
		}
		//float r = length(at);
		//th = acos(abs( at.y) / r);
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

		//cp.y = 0.f;
		double3 disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < pt.w){
			return pt.w - dist;
		}
	}
	return -1.0f;
}

__global__ void cylinder_hertzian_contact_force_kernel(
	device_cylinder_info *cy,
	float E,
	float pr,
	float rest,
	float ratio,
	float fric,
	float4* pos,
	float3* vel,
	float3* omega,
	float3* force,
	float3* moment,
	//float* rad,
	float* mass,
	float pE,
	float pPr,
	double3* mpos,
	double3* mf,
	double3* mm)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= cte.np)
		return;

	*mf = make_double3(0.0, 0.0, 0.0);
	*mm = make_double3(0.0, 0.0, 0.0);
	double overlap = 0.f;
	double im = mass[id];
	double4 ipos = make_double4((double)pos[id].x, (double)pos[id].y, (double)pos[id].z, (double)pos[id].w);
	double3 ivel = make_double3((double)vel[id].x, (double)vel[id].y, (double)vel[id].z);
	double3 iomega = make_double3((double)omega[id].x, (double)omega[id].y, (double)omega[id].z);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	double3 cp = make_double3(0.0, 0.0, 0.0);
	double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	overlap = particle_cylinder_contact_detection(cy, ipos, unit, cp, id);
	double3 si = cp - mp;
	double3 cy2cp = cp - cy->origin;
	double3 single_force = make_double3(0.0, 0.0, 0.0);
	double3 shear_force = make_double3(0.0, 0.0, 0.0);
	double3 m_moment = make_double3(0.0, 0.0, 0.0);
	//float3 m_force = make_float3(0.f);
	if (overlap > 0)
	{
		double3 dv = cy->vel + cross(cy->omega, cy2cp) - (ivel + cross(iomega, ipos.w * unit));
		device_force_constant<double> c = getConstant<double>(ipos.w, 0, im, 0, pE, E, pPr, pr, rest, ratio, fric/*, riv[id]*/);
		double fsn = -c.kn * pow(overlap, 1.5);
		single_force = (fsn + c.vn * dot(dv, unit)) * unit;
		double3 e = dv - dot(dv, unit) * unit;
		double mag_e = length(e);
		if (mag_e){
			double3 s_hat = e / mag_e;
			double ds = mag_e * cte.dt;
			shear_force = min(c.ks * ds + c.vs * dot(dv, s_hat), c.mu * length(single_force)) * s_hat;
			m_moment = cross(ipos.w * unit, shear_force);
		}
		//m_force += single_force;
	}
	double3 sum_f = single_force;// +shear_force;
	force[id] += make_float3(sum_f.x, sum_f.y, sum_f.z);
	moment[id] += make_float3(m_moment.x, m_moment.y, m_moment.z);
	mf[id] = -(sum_f + shear_force);// +make_double3(1.0, 5.0, 9.0);
	mm[id] = -cross(si, sum_f + shear_force);
}

template <typename T, unsigned int blockSize>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	/*extern*/ __shared__ T sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	
	T mySum = make_double3(0, 0, 0);;
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
// 	if (tid < 32) {
// 		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
// 		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
// 		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
// 		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
// 		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
// 		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//	}
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

__global__ void particle_polygonObject_collision_kernel(
	device_polygon_info* dpi,
	double4* dsph,
	device_polygon_mass_info* dpmi,
	float E,
	float pr,
	float rest,
	float ratio,
	float fric,
	float4 *pos,
	float3 *vel,
	float3 *omega,
	float3 *force,
	float3 *moment,
	float* mass,
	float pE,
	float pPr,
	unsigned int* sorted_index,
	unsigned int* cstart,
	unsigned int* cend,
	double3* mpos,
	double3* mf,
	double3* mm)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= cte.np)
		return;

	double overlap = 0.f;
	double im = (double)mass[id];
	int3 gridPos = calcGridPos(make_float3(pos[id]));
	double3 ipos = make_double3((double)pos[id].x, (double)pos[id].y, (double)pos[id].z);
	double3 ivel = make_double3((double)vel[id].x, (double)vel[id].y, (double)vel[id].z);
	double3 iomega = make_double3((double)omega[id].x, (double)omega[id].y, (double)omega[id].z);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	double3 cp = make_double3(0.0, 0.0, 0.0);
	double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	double ir = pos[id].w;
//  	unsigned int tid[4] = { 0, };
//  	unsigned int tu_cnt = 0;
//	double jr = 0;
	//double im = mass[id];
	//double jm = 0;
	//float3 m_force = mass[id] * cte.gravity;
	double3 m_moment = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	double3 single_force = make_double3(0, 0, 0);
	double3 shear_force = make_double3(0, 0, 0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	//bool isc = false;
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
							//double3 sph = make_double3(dsph[k].x, dsph[k].y, dsph[k].z);
						//	double sr = dsph[k].w;
							double3 distVec;// = make_double3(sph.x - ipos.x, sph.y - ipos.y, sph.z - ipos.z);
							double dist;// = sqrt(distVec.x * distVec.x + distVec.y * distVec.y + distVec.z * distVec.z);
							//if ((sr + ir) - dist > 0){
								double3 cp = closestPtPointTriangle(dpi[k], ipos, ir);
								double3 po2cp = cp - dpmi->origin;
								double3 si = cp - mp;
								distVec = ipos - cp;
								dist = length(distVec);
								overlap = ir - dist;
								single_force = make_double3(0.0, 0.0, 0.0);
								if (overlap > 0)
								{
 									unit = -dpi[k].N;
//  									if (!tu_cnt)
//  										tid[tu_cnt++] = k;
//  									else{
// //  										isc = checkConcave(dpi, tid, k, dsph, tu_cnt);
// // 										if (!isc)
// // 											continue;
//  									}
									
									double3 dv = dpmi->vel + cross(dpmi->omega, po2cp) - (ivel + cross(iomega, ir * unit));
									device_force_constant<double> c = getConstant<double>(ir, 0, im, 0, pE, E, pPr, pr, rest, ratio, fric);
									double fsn = -c.kn * pow(overlap, 1.5);
									single_force = (fsn + c.vn * dot(dv, unit)) * unit;
// 									printf("%f, %f\n", c.kn, c.vn);
// 									printf("%d, %f, %f, %f\n",k, single_force.x, single_force.y, single_force.z);
									double3 e = dv - dot(dv, unit) * unit;
									double mag_e = length(e);
									if (mag_e){
										double3 s_hat = e / mag_e;
										double ds = mag_e * cte.dt;
										shear_force = min(c.ks * ds + c.vs * dot(dv, s_hat), c.mu * length(single_force)) * s_hat;
										m_moment = cross(ir * unit, shear_force);
									}
									double3 sum_f = single_force;// +shear_force;
									force[id] += make_float3(sum_f.x, sum_f.y, sum_f.z);
									moment[id] += make_float3(m_moment.x, m_moment.y, m_moment.z);
									mf[id] += -(sum_f + shear_force);// +make_double3(1.0, 5.0, 9.0);
									mm[id] += -cross(si, sum_f + shear_force);
									//return;
									//m_force += single_force;
								}
								
							//}
						}
					}
				}
			}
		}
	}
//	force[id] = m_force;
	//moment[id] = m_moment;
}