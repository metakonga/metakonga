#include "cu_dem_dec.cuh"
#include <cstdio>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_math.h>

__constant__ device_parameters cte;

inline __device__ int sign(double a)
{
	return a < 0 ? -1 : 1;
}

inline __device__ double3 make_double3(double4 v4){
	return make_double3(v4.x, v4.y, v4.z);
}

inline __device__ double3 operator*(double a, double3 b)
{
    return make_double3(a * b.x, a * b.y, a * b.z);
}

inline __device__ void operator-=(double3& a, double3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

inline __device__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ double3 operator-(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ double3 operator-(double3 a)
{
	return make_double3(-a.x, -a.y, -a.z);
}

inline __device__ double length(double3 v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __device__ double dot(double3 a, double3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ double3 cross(double3 a, double3 b)
{
	 return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __device__ double3 operator/(double3 a, double b)
{
	double inv = 1 / b;
	return make_double3( a.x * inv, a.y * inv, a.z * inv );
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

__global__ void vv_update_position_kernel(double4* pos, double4* vel, double4* acc)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(id >= cte.np) 
		return;
	double3 p = make_double3(pos[id]);
	double3 v = make_double3(vel[id]);
	double3 a = make_double3(acc[id]);
	
	p += cte.dt * v + cte.half2dt * a;
// 	if(id == 31914){
// 		p = make_double3(pos[id]);
// 	}
	pos[id] = make_double4(p.x, p.y, p.z, pos[id].w);
}

__global__ void vv_update_velocity_kernel(
	double4* vel,
	double4* acc,
	double4* omega,
	double4* alpha,
	double3* force,
	double3* moment)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(id >= cte.np) 
		return;
	double3 v = make_double3(vel[id]);
	double3 a = make_double3(acc[id]);
	double3 av = make_double3(omega[id]);
	double3 aa = make_double3(alpha[id]);
	double mass = acc[id].w;
	double inertia = alpha[id].w;

	v += 0.5 * cte.dt * a;
	av += 0.5 * cte.dt * aa;
	a = (1 / mass) * force[id];
	aa = (1 / inertia) * moment[id];
	v += 0.5 * cte.dt * a;
	av += 0.5 * cte.dt * aa;
// 	if(id == 0){
// 		printf("Velocity --- > id = %d -> [%f.6, %f.6, %f.6]\n", id, v.x, v.y, v.z);
// 	}
	vel[id] = make_double4(v.x, v.y, v.z, 0);
	omega[id] = make_double4(av.x, av.y, av.z, 0);
	acc[id] = make_double4(a.x, a.y, a.z, mass);
	alpha[id] = make_double4(aa.x, aa.y, aa.z, inertia);
}

__global__ void calculateHashAndIndex_kernel(unsigned int* hash, unsigned int* index, double4* pos)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(id >= (cte.np + cte.nsp)) return;
	volatile double4 p = pos[id];

	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned _hash=calcGridHash(gridPos);
	/*if(_hash >= cte.ncell) 
		printf("Over limit - hash number : %d", _hash);*/
	hash[id] = _hash;
	index[id] = id;
}

__global__ void reorderDataAndFindCellStart_kernel(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index)
{
	extern __shared__ uint sharedHash[];	//blockSize + 1 elements
	unsigned id = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	unsigned _hash;

	unsigned int tnp = cte.np + cte.nsp;

	if (id < tnp)
	{
		_hash = hash[id];

		sharedHash[threadIdx.x+1] = _hash;

		if (id > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[id-1];
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

__device__ bool calForce(
	double ir, 
	double jr,
	double3 ipos,
	double3 jpos, 
	double3 ivel, 
	double3 jvel,
	double3 iomega,
	double3 jomega,
	double3& force,
	double3& moment)
{
	double3 relative_pos = jpos - ipos;
	double dist = length(relative_pos);
	double collid_dist = (ir + jr) - dist;

	if(collid_dist <= 0){
		if(cte.cohesive == 0)
			return false;
		double tu = cte.cohesive / cte.mu;
		double3 unit = relative_pos / dist;
		double3 relative_vel = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
		double Fn = (-cte.kn * pow(-collid_dist, 1.5) + cte.vn * dot(relative_vel, unit));
		if(Fn < 0 && Fn > -tu){
			force += tu * unit;
		}
		double3 e = relative_vel - dot(relative_vel, unit) * unit;
		double mag_e = length(e);
		if(mag_e){
			double3 s_hat = e / mag_e;
			double ds = mag_e * cte.dt;
			double Fs = min(cte.ks * ds + cte.vs * (dot(relative_vel, s_hat)), cte.mu * Fn);
			if(Fs < Fn*cte.mu + cte.cohesive){
				moment += cross(ir * unit, (Fn*cte.mu + cte.cohesive )*s_hat);
			}
		}
		return false;
	}

	double3 unit = relative_pos / dist;
	double3 relative_vel = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
	double3 single_force = (-cte.kn * pow(collid_dist, 1.5) + cte.vn * dot(relative_vel, unit)) * unit;
	double3 single_moment = make_double3(0,0,0);
	double3 e = relative_vel - dot(relative_vel, unit) * unit;
	double mag_e = length(e);
	if(mag_e){
		double3 s_hat = e / mag_e;
		double ds = mag_e * cte.dt;
		double3 shear_force = min(cte.ks * ds + cte.vs * (dot(relative_vel, s_hat)), cte.mu * length(single_force)) * s_hat;
		single_moment = cross(ir * unit, shear_force);
	}
	force += single_force;
	moment += single_moment;
	return true;
}

__global__ void calculate_p2p_kernel(
	double4* pos,
	double4* vel,
	double4* acc,
	double4* omega,
	double4* alpha,
	double3* force,
	double3* moment,
	unsigned int* sorted_index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int cRun = 0)
{
	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if(id >= cte.np) 
		return;

	double3 ipos = make_double3(pos[id]);
	double3 jpos = make_double3(0, 0, 0);
	double3 ivel = make_double3(vel[id]);
	double3 jvel = make_double3(0, 0, 0);
	double3 iomega = make_double3(omega[id]);
	double3 jomega = make_double3(0, 0, 0);
	int3 gridPos = calcGridPos(ipos);

	double ir = pos[id].w;
	double jr = 0;
	double3 m_force = acc[id].w * cte.gravity;
	double3 m_moment = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
// 	if(id == 31914){
// 		grid_hash = 0;
// 	}
	unsigned int start_index = 0;
	unsigned int end_index = 0;
// 	if(id == 31914){
// 		end_index = 0;
// 	}
	for(int z = -1; z <= 1; z++){
		for(int y = -1; y <= 1; y++){
			for(int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
 				if(start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for(unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if(id == k || k >= cte.np)
							continue;
						jpos = make_double3(pos[k]);
						jvel = make_double3(vel[k]);
						jomega = make_double3(omega[k]);
						jr = pos[k].w;

						if(!calForce(ir, jr, ipos, jpos, ivel, jvel, iomega, jomega, m_force, m_moment))
							continue;
					}
				}
			}
		}
	}
	//force[cte.np] = make_double3(0, 0, 0);
	force[id] += m_force;
	if(id == 0){
		id = 0;
	}
// 	if(id == 60775){
// 		printf("id = %d -> [%f.6, %f.6, %f.6]\n", id, force[id].x, force[id].y, force[id].z);
// 	}
	moment[id] += m_moment;
}

__global__ void cube_hertzian_contact_force_kernel(
	device_plane_info *planes,
	double kn,
	double vn,
	double ks,
	double vs,
	double mu,
	bool* isLineContact,
	double4* pos,
	double4* vel,
	double4* omega,
	double3* force,
	double3* moment)
{
	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if(id >= cte.np) 
		return;
	
	double r = pos[id].w;
	double3 ipos = make_double3(pos[id]);
	double3 ivel = make_double3(vel[id]);
	double3 iomega = make_double3(omega[id]);

	double3 single_force = make_double3(0, 0, 0);
	double3 m_force = make_double3(0, 0, 0);
	double3 m_moment = make_double3(0, 0, 0);
	for(int i = 0; i < 6; i++){
		device_plane_info plane = planes[i];
		double3 dp = ipos - plane.xw;
		double3 wp = make_double3(dot(dp, plane.u1), dot(dp, plane.u2), dot(dp, plane.uw));
		if(abs(wp.z) < r && (wp.x > 0 && wp.x < plane.l1) && (wp.y > 0 && wp.y < plane.l2)){
			double3 unit = -sign(dot(ipos - plane.xw, plane.uw)) * (plane.uw / length(plane.uw));
			double collid_dist = r - abs(dot(ipos - plane.xw, unit));
			double3 dv = -(ivel + cross(iomega, r * unit));
			single_force = (-kn * pow(collid_dist, 1.5) + vn * dot(dv, unit)) * unit;
			double3 e = dv - dot(dv, unit) * unit;
			double mag_e = length(e);
			if(mag_e){
				double3 s_hat = e / mag_e;
				double ds = mag_e * cte.dt;
				double3 shear_force = min(ks * ds + vs * dot(dv, s_hat), mu * length(single_force)) * s_hat;
				m_moment += cross(r * unit, shear_force);
			}
			m_force += single_force;
		}
	}
	force[id] += m_force;
	moment[id] += m_moment;
}

__device__ double3 ClosestPtPointTriangle(
	double3 p,
	double3 a,
	double3 b,
	double3 c,
	int *wc)
{
	double3 ab = b - a;
	double3 ac = c - a;
	double3 ap = p - a;

	double d1 = dot(ab, ap);
	double d2 = dot(ac, ap);
	if(d1 <= 0.0 && d2 <= 0.0){
		*wc = 0;
		return a;
	}

	double3 bp = p - b;
	double d3 = dot(ab, bp);
	double d4 = dot(ac, bp);
	if(d3 >= 0.0 && d4 <= d3){
		*wc = 0;
		return b;
	}
	double vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
		*wc = 1;
		double v = d1 / (d1 - d3);
		return a + v * ab;
	}

	double3 cp = p - c;
	double d5 = dot(ab, cp);
	double d6 = dot(ac, cp);
	if(d6 >= 0.0 && d5 <= d6){
		*wc = 0;
		return c;
	}

	double vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
		*wc = 1;
		double w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if(va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
		*wc = 1;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;

	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

__device__ bool isIntersect2Lines(double x1, double x2, double x3, double x4, double y1, double y2, double y3, double y4)
{
	double div = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1);
	double t = ( (x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3) ) / div;
	double s = ( (x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3) ) / div;
	if(t >= 0 && t <= 1 || s >= 0 && s <= 1)
		return true;

	return false;
}

__device__ bool calculateTriangleParticleContactForce(
	double3* plane_cps,
	double3* plane_normals,
	int &plane_contact_count,
	unsigned int id,
	double r,
	double kn,
	double vn,
	double ks,
	double vs,
	double mu,
	double3 pos,
	double3 vel,
	double3 omega,
	double3& force,
	double3& moment,
	cu_polygon* polygons,
	unsigned int* id_set,
	unsigned int* poly_start,
	unsigned int* poly_end,
	cu_mass_info* mi)
{
	cu_polygon poly;
//	cu_contact_info cinfo;
//	bool is_line_contact = false;
	int wc = -1;
	unsigned int poly_id = 0;
	double3 sp, contact_point, unit, p_unit;//, s_unit;
	double3 single_force, dv, e, s_hat, shear_force;
	double dist, collid_dist, ds, mag_e;
	for(unsigned int i = poly_start[id]; i < poly_end[id]; i++){
		poly_id = id_set[i];
		poly = polygons[poly_id];
		contact_point = ClosestPtPointTriangle(pos, poly.P, poly.Q, poly.R, &wc);
		sp = pos - contact_point;
		dist = length(sp);
		//s_unit = -sp / dist;
		p_unit = -poly.N / length(poly.N);
		//if(dot(s_unit, p_unit) <= 0)
			//continue;
		collid_dist = r - dist;
		if(collid_dist <= 0)
			continue;

		if(wc == 2)
		{
			if(plane_contact_count){
				double3 sp = plane_cps[plane_contact_count - 1];
				double3 pn = plane_normals[plane_contact_count - 1];
				double3 ep1 = sp + 10000 * pn;
				double3 ep2 = contact_point + 10000 * p_unit;
				if(!isIntersect2Lines(sp.x, ep1.x, contact_point.x, ep2.x, sp.y, ep1.y, contact_point.y, ep2.y))
					continue;
			}
			unit = p_unit;
			dv = mi->vel - (vel + cross(omega, r * unit));
			single_force = (-kn * pow(collid_dist, 1.5) + vn * dot(dv, unit)) * unit;
			e = dv - dot(dv,unit) * unit;
			mag_e = length(e);
			if(mag_e){
				s_hat = e / mag_e;
				ds = mag_e * cte.dt;
				shear_force = min(ks * ds + vs * dot(dv, s_hat), mu * length(single_force)) * s_hat;
				moment = cross(r * unit, shear_force);
			}
			force += single_force;
			plane_cps[plane_contact_count] = contact_point;
			plane_normals[plane_contact_count++] = unit;
			return true;
		}
	}
	return false;
}

__device__ bool calculateTriangleLineOrEdgeParticleContactForce(
	unsigned int id,
	double r,
	double kn,
	double vn,
	double ks,
	double vs,
	double mu,
	double3 pos,
	double3 vel,
	double3 omega,
	double3& line_force,
	double3& line_moment,
	cu_polygon* polygons,
	unsigned int* id_set,
	unsigned int* poly_start,
	unsigned int* poly_end,
	cu_mass_info* mi)
{
	cu_polygon poly;
	cu_contact_info cinfo;
	int wc = -1;
	unsigned int poly_id = 0;
	double3 sp, contact_point, unit/*t, p_unit*/, s_unit;
	double3 single_force, dv, e, s_hat, shear_force;
	double dist, collid_dist, ds, mag_e;
	for(unsigned int i = poly_start[id]; i < poly_end[id]; i++){
		poly_id = id_set[i];
		poly = polygons[poly_id];
		contact_point = ClosestPtPointTriangle(pos, poly.P, poly.Q, poly.R, &wc);
		sp = pos - contact_point;
		dist = length(sp);
		s_unit = -sp / dist;
		//p_unit = -poly.N / length(poly.N);
// 		if(dot(s_unit, p_unit) <= 0)
// 			continue;
		collid_dist = r - dist;
		if(collid_dist <= 0)
			continue;

		if(wc == 0 || wc == 1)
		{
			cinfo.penetration = collid_dist;
			cinfo.unit = s_unit;
			unit = cinfo.unit;	
			dv = mi->vel - (vel + cross(omega, r * unit));
			single_force = (-kn * pow(cinfo.penetration, 1.5) + vn * dot(dv, unit)) * unit;
			e = dv - dot(dv,unit) * unit;
			mag_e = length(e);
			if(mag_e){
				s_hat = e / mag_e;
				ds = mag_e * cte.dt;
				shear_force = min(ks * ds + vs * dot(dv, s_hat), mu * length(single_force)) * s_hat;
				line_moment = cross(r * unit, shear_force);
			}
			line_force = single_force;
			return true;
		}
	}
	return false;
}


__global__ void shape_line_or_edge_contact_force_kernel(
	cu_mass_info* mi,
	cu_polygon *polygons,
	unsigned int *id_set,
	unsigned int *poly_start, 
	unsigned int *poly_end,
	double kn,
	double vn,
	double ks,
	double vs,
	double mu,
	bool* isLineContact,
	double4 *pos,
	double4 *vel,
	double4 *omega,
	double3 *force,
	double3 *moment,
	unsigned int* sorted_id,
	unsigned int* cell_start,
	unsigned int* cell_end)
{
	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if(id >= cte.np) 
		return;

	if(isLineContact[id]){
		return;
	}

	double3 ipos = make_double3(pos[id]);
	double3 ivel = make_double3(vel[id]);
	double3 iomega = make_double3(omega[id]);
	int3 gridPos = calcGridPos(ipos);

	double ir = pos[id].w;
	/*isLineContact[id] = false;*/
//	double jr = 0.0;
	double3 l_force = make_double3(0, 0, 0);
	double3 l_moment = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int k = 0;
//	bool shape_contact_state = false;
	for(int z = -1; z <= 1; z++){
		for(int y = -1; y <= 1; y++){
			for(int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cell_start[grid_hash];
				if(start_index != 0xffffffff){
					end_index = cell_end[grid_hash];
					for(unsigned int j = start_index; j < end_index; j++){
						k = sorted_id[j];
						if(id == k)
							continue;

						///jr = pos[k].w;
						if(k >= cte.np/* && !shape_contact_state*/){
							    bool isLineCont = calculateTriangleLineOrEdgeParticleContactForce(
								k - cte.np,	ir,
								kn,	vn,	ks,	vs,	mu,
								ipos, ivel, iomega,
								l_force, l_moment,
								polygons,id_set, poly_start, poly_end, mi
								);
								if(isLineCont){
									isLineContact[id] = true;
									force[id] += l_force;
									moment[id] += l_moment;
									return;
								}
						}
					}
				}
			}
		}
	}
// 	
// 	force[id] += l_force;
// 	moment[id] += l_moment;
}

__global__ void shpae_hertzian_contact_force_kernel(
		cu_mass_info* mi,
		cu_polygon *polygons,
		unsigned int *id_set,
		unsigned int *poly_start, 
		unsigned int *poly_end,
		double kn,
		double vn,
		double ks,
		double vs,
		double mu,
		bool* isLineContact,
		double4 *pos,
		double4 *vel,
		double4 *omega,
		double3 *force,
		double3 *moment,
		unsigned int* sorted_id,
		unsigned int* cell_start,
		unsigned int* cell_end)
{
 	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

 	if(id >= cte.np) 
 		return;

	//if(isLineContact[id])
	//	return;
// 	if(id == 44446){
// 		isLineContact[id] = false;
// 	}
	isLineContact[id] = false;
 	double3 ipos = make_double3(pos[id]);
 	double3 ivel = make_double3(vel[id]);
 	double3 iomega = make_double3(omega[id]);
 	int3 gridPos = calcGridPos(ipos);

 	double ir = pos[id].w;
//	double jr = 0.0;
	int plane_contact_count = 0;
	double3 plane_normals[10] = {0, };
	double3 plane_cps[10] = {0, };
	double3 m_force = make_double3(0, 0, 0);
 	double3 m_moment = make_double3(0, 0, 0);
 	int3 neighbour_pos = make_int3(0, 0, 0);
 	uint grid_hash = 0;
 	unsigned int start_index = 0;
 	unsigned int end_index = 0;
	unsigned int k = 0;
//	bool shape_contact_state = false;
 	for(int z = -1; z <= 1; z++){
 		for(int y = -1; y <= 1; y++){
 			for(int x = -1; x <= 1; x++){
 				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
 				grid_hash = calcGridHash(neighbour_pos);
 				start_index = cell_start[grid_hash];
 				if(start_index != 0xffffffff){
 					end_index = cell_end[grid_hash];
 					for(unsigned int j = start_index; j < end_index; j++){
 						k = sorted_id[j];
 						if(id == k)
 							continue;

 						//jr = pos[k].w;
						if(k >= cte.np/* && !shape_contact_state*/){
							bool shape_contact_state = calculateTriangleParticleContactForce(
													plane_cps, plane_normals, plane_contact_count, k - cte.np,	ir,
													kn,	vn,	ks,	vs,	mu,
													ipos, ivel, iomega,
													m_force, m_moment,
													polygons,id_set, poly_start, poly_end, mi
													);
							if(shape_contact_state){
								force[id] += m_force;
								//printf("%f", force[id].y);
								moment[id] += m_moment;
								isLineContact[id] = true;
								return;
							}
						}
 					}
 				}
 			}
 		}
 	}
	
}

__global__
	void update_polygons_position(double3* mpos, double* A, double3* local_vertice, unsigned int nv, double3 *pos)
{
	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

 	if(id >= nv) 
 		return;

	double3 lv = local_vertice[id];
	double tx = A[0] * lv.x + A[1] * lv.y + A[2] * lv.z;
	double ty = A[3] * lv.x + A[4] * lv.y + A[5] * lv.z;
	double tz = A[6] * lv.x + A[7] * lv.y + A[8] * lv.z;

	pos[id].x = mpos->x + tx;
	pos[id].y = mpos->y + ty;
	pos[id].z = mpos->z + tz;
}

__global__ void mergedata_kernel(double4* data, double4* pos, uint np, double3* vertice, uint snp)
{
	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if(id >= np + snp)
		return;

	if(id < np){
		data[id] = pos[id];
	}
	else{
		double3 v = vertice[id - np];
		data[id] = make_double4(v.x, v.y, v.z, -1.0);
	}
}

__global__
	void update_polygons_information(cu_polygon* polygons, uint3 *indice, double3* pos, unsigned int ni)
{
	unsigned id = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

 	if(id >= ni) 
 		return;

	cu_polygon po;

	uint3 idx = indice[id];
	po.P = make_double3(pos[idx.x].x, pos[idx.x].y, pos[idx.x].z);
	po.Q = make_double3(pos[idx.y].x, pos[idx.y].y, pos[idx.y].z);
	po.R = make_double3(pos[idx.z].x, pos[idx.z].y, pos[idx.z].z);
	po.V = po.Q - po.P;
	//double len = length(po.V);
	po.W = po.R - po.P;
	po.N = cross(po.V, po.W);

	polygons[id] = po;
}

template <typename T, unsigned int blockSize>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	extern __shared__ double3 sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = make_double3(0, 0, 0);
	while (i < n) 
	{ 
		sdata[tid] += g_idata[i];
		if(i + blockSize < n)
			sdata[tid] += g_idata[i+blockSize]; 
		i += gridSize; 
	}
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}