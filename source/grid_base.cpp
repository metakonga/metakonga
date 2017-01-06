#include "grid_base.h"
#include "mphysics_cuda_dec.cuh"

VEC3F grid_base::wo;				// world origin
float grid_base::cs = 0.f;			// cell size
VEC3UI grid_base::gs;				// grid size

unsigned int* grid_base::sorted_id = NULL;
unsigned int* grid_base::cell_id = NULL;
unsigned int* grid_base::body_id = NULL;
unsigned int* grid_base::cell_start = NULL;
unsigned int* grid_base::cell_end = NULL;

grid_base::grid_base()
	: name("grid_base")
	, d_sorted_id(NULL)
	, d_cell_id(NULL)
	, d_body_id(NULL)
	, d_cell_start(NULL)
	, d_cell_end(NULL)
	, nse(0)
{

}

grid_base::grid_base(std::string _name, modeler* _md)
	: name(_name)
	, md(_md)
	, d_sorted_id(NULL)
	, d_cell_id(NULL)
	, d_body_id(NULL)
	, d_cell_start(NULL)
	, d_cell_end(NULL)
	, nse(0)
{

}

grid_base::~grid_base()
{
	clear();
}

void grid_base::clear()
{
	if (cell_id) delete[] cell_id; cell_id = NULL;
	if (body_id) delete[] body_id; body_id = NULL;
	if (sorted_id) delete[] sorted_id; sorted_id = NULL;
	if (cell_start) delete[] cell_start; cell_start = NULL;
	if (cell_end) delete[] cell_end; cell_end = NULL;

	if (d_cell_id) checkCudaErrors(cudaFree(d_cell_id)); d_cell_id = NULL;
	if (d_body_id) checkCudaErrors(cudaFree(d_body_id)); d_body_id = NULL;
	if (d_sorted_id) checkCudaErrors(cudaFree(d_sorted_id)); d_sorted_id = NULL;
	if (d_cell_start) checkCudaErrors(cudaFree(d_cell_start)); d_cell_start = NULL;
	if (d_cell_end) checkCudaErrors(cudaFree(d_cell_end)); d_cell_start = NULL;
}

void grid_base::allocMemory(unsigned int n)
{
	ng = gs.x * gs.y * gs.z;
	cell_id = new unsigned int[n]; memset(cell_id, 0, sizeof(unsigned int)*n);
	body_id = new unsigned int[n]; memset(body_id, 0, sizeof(unsigned int)*n);
	sorted_id = new unsigned int[n]; memset(sorted_id, 0, sizeof(unsigned int)*n);
	cell_start = new unsigned int[ng]; memset(cell_start, 0, sizeof(unsigned int)*ng);
	cell_end = new unsigned int[ng]; memset(cell_end, 0, sizeof(unsigned int)*ng);
	nse = n;
}

void grid_base::cuAllocMemory(unsigned int n)
{
	ng = gs.x * gs.y * gs.z;
	checkCudaErrors(cudaMalloc((void**)&d_cell_id, sizeof(unsigned int)*n));
	checkCudaErrors(cudaMalloc((void**)&d_body_id, sizeof(unsigned int)*n));
	checkCudaErrors(cudaMalloc((void**)&d_sorted_id, sizeof(unsigned int)*n));
	checkCudaErrors(cudaMalloc((void**)&d_cell_start, sizeof(unsigned int)*ng));
	checkCudaErrors(cudaMalloc((void**)&d_cell_end, sizeof(unsigned int)*ng));
}

VEC3I grid_base::getCellNumber(float x, float y, float z)
{
	return VEC3I(
		static_cast<int>(abs(std::floor((x - wo.x) / cs))),
		static_cast<int>(abs(std::floor((y - wo.y) / cs))),
		static_cast<int>(abs(std::floor((z - wo.z) / cs)))
		);
}

VEC3I grid_base::getCellNumber(double x, double y, double z)
{
	return VEC3I(
		static_cast<int>(abs(std::floor((x - (double)wo.x) / cs))),
		static_cast<int>(abs(std::floor((y - (double)wo.y) / cs))),
		static_cast<int>(abs(std::floor((z - (double)wo.z) / cs)))
		);
}

unsigned int grid_base::getHash(VEC3I& c3)
{
	VEC3I gridPos;
	gridPos.x = c3.x & (gs.x - 1);
	gridPos.y = c3.y & (gs.y - 1);
	gridPos.z = c3.z & (gs.z - 1);
	return (gridPos.z*gs.y) * gs.x + (gridPos.y*gs.x) + gridPos.x;
}
