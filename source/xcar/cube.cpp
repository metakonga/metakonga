#include "cube.h"
#include "mphysics_cuda_dec.cuh"

unsigned int cube::nCube = 0;

cube::cube(QString& _name, geometry_use _roll)
	: pointMass(_name, CUBE, _roll)
	, planes(NULL)
{
	
}

cube::cube(const cube& _cube)
	: planes(NULL)
	, ori(_cube.origin())
	, min_p(_cube.min_point())
	, max_p(_cube.max_point())
	, size(_cube.cube_size())
	, pointMass(_cube)
{
	if (!planes)
		planes = new plane[6];
	memcpy(planes, _cube.Planes(), sizeof(plane) * 6);
}

cube::~cube()
{
	if (planes) delete[] planes; planes = NULL;
	nCube--;
}

// void cube::cuAllocData(unsigned int _np)
// {
// 	device_plane_info *_dpi = new device_plane_info[6];
// 	for (unsigned int i = 0; i < 6; i++)
// 	{
// 		_dpi[i].l1 = planes[i].L1();
// 		_dpi[i].l2 = planes[i].L2();;
// 		_dpi[i].xw = make_double3(planes[i].XW().x, planes[i].XW().y, planes[i].XW().z);
// 		_dpi[i].uw = make_double3(planes[i].UW().x, planes[i].UW().y, planes[i].UW().z);
// 		_dpi[i].u1 = make_double3(planes[i].U1().x, planes[i].U1().y, planes[i].U1().z);
// 		_dpi[i].u2 = make_double3(planes[i].U2().x, planes[i].U2().y, planes[i].U2().z);
// 		_dpi[i].pa = make_double3(planes[i].PA().x, planes[i].PA().y, planes[i].PA().z);
// 		_dpi[i].pb = make_double3(planes[i].PB().x, planes[i].PB().y, planes[i].PB().z);
// 		_dpi[i].w2 = make_double3(planes[i].W2().x, planes[i].W2().y, planes[i].W2().z);
// 		_dpi[i].w3 = make_double3(planes[i].W3().x, planes[i].W3().y, planes[i].W3().z);
// 		_dpi[i].w4 = make_double3(planes[i].W4().x, planes[i].W4().y, planes[i].W4().z);
// 	}
// 	
// 	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info) * 6));
// 	checkCudaErrors(cudaMemcpy(dpi, _dpi, sizeof(device_plane_info) * 6, cudaMemcpyHostToDevice));
// 	delete _dpi;
// }

bool cube::define(VEC3D& min, VEC3D& max)
{
	if (!planes)
		planes = new plane[6];
	min_p = min;
	max_p = max;
	pointMass::pos = 0.5 * (min + max);
	size.x = (max_p - VEC3D(min_p.x, max_p.y, max_p.z)).length();
	size.y = (max_p - VEC3D(max_p.x, min_p.y, max_p.z)).length();
	size.z = (max_p - VEC3D(max_p.x, max_p.y, min_p.z)).length();

	planes[0].define(min_p, min_p + VEC3D(0, 0, size.z), min_p + VEC3D(size.x, 0, 0));
	planes[1].define(min_p, min_p + VEC3D(0, size.y, 0), min_p + VEC3D(0, 0, size.z));
	planes[2].define(min_p + VEC3D(size.x, 0, 0), min_p + VEC3D(size.x, 0, size.z), min_p + VEC3D(size.x, size.y, 0));
	planes[3].define(min_p, min_p + VEC3D(size.x, 0, 0), min_p + VEC3D(0, size.y, 0));
	planes[4].define(min_p + VEC3D(0, 0, size.z), min_p + VEC3D(0, size.y, size.z), min_p + VEC3D(size.x, 0, size.z));
	planes[5].define(min_p + VEC3D(0, size.y, 0), min_p + VEC3D(size.x, size.y, 0), min_p + VEC3D(0, size.y, size.z));
	nCube++;
	//save_shape_data();
	return true;
}

// unsigned int cube::makeParticles(double rad, VEC3UI &_size, VEC3D &_spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos, unsigned int sid)
// {
// 	//unsigned int np = 0;
// 	if (isOnlyCount){
// 		vector3<unsigned int> dim3np(
// 			static_cast<unsigned int>(abs(size.x / (rad * 2)))
// 			, static_cast<unsigned int>(abs(size.y / (rad * 2)))
// 			, static_cast<unsigned int>(abs(size.z / (rad * 2))));
// 
// 		//VEC3F space;
// 		double dia = rad * 2.0;
// 		VEC3D space = rad * 0.1;
// 		double x_len = dim3np.x * dia + (dim3np.x + 1) * space.x;
// 		double y_len = dim3np.y * dia + (dim3np.y + 1) * space.y;
// 		double z_len = dim3np.z * dia + (dim3np.z + 1) * space.z;
// 		if (x_len > size.x){
// 			dim3np.x--;
// 			space.x = (size.x - dim3np.x * dia) / (dim3np.x + 1);
// 			//x_len = dim3np.x * dia + (dim3np.x + 1) * space.x;
// 		}
// 		if (y_len > size.y){
// 			dim3np.y--;
// 			space.y = (size.y - dim3np.y * dia) / (dim3np.y + 1);
// 		}
// 		if (z_len > size.z){
// 			dim3np.z--;
// 			space.z = (size.z - dim3np.z * dia) / (dim3np.z + 1);
// 		}
// 		//float spacing = rad * 2.f + _spacing;
// 		count = dim3np.x * dim3np.y * dim3np.z;
// 		_spacing = space;
// 		_size = dim3np;
// 	}
// 	else{
// 		srand(1976);
// 		double jitter = rad * 0.001;
// 		unsigned int cnt = 0;
// 		for (unsigned int i = 0; i <= nstack; i++){
// 			for (unsigned int z = 0; z < _size.z; z++){
// 				for (unsigned int y = 0; y < _size.y; y++){
// 					for (unsigned int x = 0; x < _size.x; x++){
// 						double p[3] = {
// 							(min_p.x + rad * (2.0 * x + 1) + (x + 1) * _spacing.x) + frand()*jitter,
// 							(min_p.y + rad * (2.0 * y + 1) + (y + 1) * _spacing.y) + frand()*jitter,
// 							(min_p.z + rad * (2.0 * z + 1) + (z + 1) * _spacing.z) + frand()*jitter };
// 
// 						pos[cnt].x = p[0];
// 						pos[cnt].y = p[1];
// 						pos[cnt].z = p[2];
// 						pos[cnt].w = rad;
// 						cnt++;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return count;
// }