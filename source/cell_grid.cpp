#include "parSIM.h"

#include "cu_dem_dec.cuh"

using namespace parSIM;

cell_grid::cell_grid(Simulation *Sim)
	: sim(Sim)
	, data(NULL)
	, d_data(NULL)
	, m_np(0)
	, m_snp(0)
	, cell_id(NULL), d_cell_id(NULL)
	, body_id(NULL), d_body_id(NULL)
	, sorted_id(NULL), d_sorted_id(NULL)
	, cell_start(NULL), d_cell_start(NULL)
	, cell_end(NULL), d_cell_end(NULL)
{

}

cell_grid::cell_grid(Simulation *demSim, Simulation *mbdSim)
	: dem_sim(demSim)
	, mbd_sim(mbdSim)
	, data(NULL)
	, d_data(NULL)
	, m_np(0)
	, m_snp(0)
	, cell_id(NULL), d_cell_id(NULL)
	, body_id(NULL), d_body_id(NULL)
	, sorted_id(NULL), d_sorted_id(NULL)
	, cell_start(NULL), d_cell_start(NULL)
	, cell_end(NULL), d_cell_end(NULL)
{

}

cell_grid::~cell_grid()
{
	if(data) delete [] data; data = NULL;
	if(cell_id) delete [] cell_id; cell_id = NULL;
	if(body_id) delete [] body_id; body_id = NULL;
	if(sorted_id) delete [] sorted_id; sorted_id = NULL;
	if(cell_start) delete [] cell_start; cell_start = NULL;
	if(cell_end) delete [] cell_end; cell_end = NULL;

	if(d_data) checkCudaErrors( cudaFree(d_data) ); d_data = NULL;
	if(d_cell_id) checkCudaErrors( cudaFree(d_cell_id) ); d_cell_id = NULL;
	if(d_body_id) checkCudaErrors( cudaFree(d_cell_id) ); d_body_id = NULL;

}

void cell_grid::initialize()
{
	std::cout << "Initialize of cell grid detection method." << std::endl;
	m_cellSize = particles::radius * 2;
	std::map<std::string, geometry*>::iterator Geo;
	if(dem_sim){
		m_np = dem_sim->getParticles()->Size();
		std::cout << "    The number of dem particle : " << m_np << std::endl;
	}

	if(mbd_sim){
		for(Geo = mbd_sim->getGeometries()->begin(); Geo != mbd_sim->getGeometries()->end(); Geo++){
			if(Geo->second->Geometry() == SHAPE){
				geo::shape *Shape = dynamic_cast<geo::shape*>(Geo->second);
				unsigned int snp = Shape->getNp();
 				//for(unsigned int i = 0; i < snp; i++){
 				//	data[m_np + m_snp + i] = vector4<double>(Shape->getVertice()(i).x, Shape->getVertice()(i).y, Shape->getVertice()(i).z, -1);
 				//}
				std::cout << "    The number of " << Shape->get_name() << " shape point : " << snp << std::endl;
				m_snp += snp;
			}
		}
	}

 	unsigned int tnp = m_np + m_snp;
// 	std::cout << "    The number of object : " << tnp << std::endl;
	nGrid = m_gs.x * m_gs.y * m_gs.z;
	data = new vector4<double>[tnp];

 	for(unsigned int i = 0; i < m_np; i++){
 		data[i] = dem_sim->getParticles()->Position()[i];
 	}

	if(mbd_sim){
		for(Geo = mbd_sim->getGeometries()->begin(); Geo != mbd_sim->getGeometries()->end(); Geo++){
			if(Geo->second->Geometry() == SHAPE){
				geo::shape *Shape = dynamic_cast<geo::shape*>(Geo->second);
				unsigned int snp = Shape->getNp();
				for(unsigned int i = 0; i < snp; i++){
					data[m_np + i] = vector4<double>(Shape->getVertice()(i).x, Shape->getVertice()(i).y, Shape->getVertice()(i).z, -1);
				}
				//std::cout << "    The number of " << Shape->get_name() << " shape point : " << snp << std::endl;
				//m_snp += snp;
			}
		}
	}
	

	cell_id = new size_t[tnp];		memset(cell_id, 0, sizeof(size_t)*tnp);
	body_id = new size_t[tnp];		memset(body_id, 0, sizeof(size_t)*tnp);
	sorted_id = new size_t[tnp];	memset(sorted_id, 0, sizeof(size_t)*tnp);
	cell_start = new size_t[nGrid];	memset(cell_start, 0, sizeof(size_t)*nGrid);
	cell_end = new size_t[nGrid];	memset(cell_end, 0, sizeof(size_t)*nGrid);


	if(dem_sim->Device()==GPU){
		checkCudaErrors( cudaMalloc((void**)&d_data, sizeof(double)*tnp*4) );
		checkCudaErrors( cudaMalloc((void**)&d_cell_id, sizeof(unsigned int) * tnp) );
		checkCudaErrors( cudaMalloc((void**)&d_body_id, sizeof(unsigned int) * tnp) );
		checkCudaErrors( cudaMalloc((void**)&d_sorted_id, sizeof(unsigned int) * tnp) );
		checkCudaErrors( cudaMalloc((void**)&d_cell_start, sizeof(unsigned int) * nGrid) );
		checkCudaErrors( cudaMalloc((void**)&d_cell_end, sizeof(unsigned int) * nGrid) );

		checkCudaErrors( cudaMemcpy(d_data, data, sizeof(double) * tnp * 4, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_cell_id, cell_id, sizeof(unsigned int) * tnp, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_body_id, body_id, sizeof(unsigned int) * tnp, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_sorted_id, sorted_id, sizeof(unsigned int) * tnp, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_cell_start, cell_start, sizeof(unsigned int) * nGrid, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_cell_end, cell_end, sizeof(unsigned int) * nGrid, cudaMemcpyHostToDevice) );
	}

	

	std::cout << "done" << std::endl;
}

unsigned int cell_grid::calcGridHash(algebra::vector3<int>& cell3d)
{
	algebra::vector3<int> gridPos;
	gridPos.x = cell3d.x & (m_gs.x - 1);
	gridPos.y = cell3d.y & (m_gs.y - 1);
	gridPos.z = cell3d.z & (m_gs.z - 1);
	return (gridPos.z*m_gs.y) * m_gs.x + (gridPos.y*m_gs.x) + gridPos.x;
}

void cell_grid::reorderDataAndFindCellStart(size_t ID, size_t begin, size_t end)
{
	cell_start[ID] = begin;
	cell_end[ID] = end;
	size_t dim = 0, bid = 0;
	for(size_t i(begin); i < end; i++){
		sorted_id[i] = body_id[i];
	}
}

algebra::vector3<int> cell_grid::get_triplet(double r1, double r2, double r3)
{
	return algebra::vector3<int>(
		static_cast<int>( abs(std::floor((r1 - m_wo.x) / m_cellSize)) ),
		static_cast<int>( abs(std::floor((r2 - m_wo.y) / m_cellSize)) ),
		static_cast<int>( abs(std::floor((r3 - m_wo.z) / m_cellSize)) )
		);
}

algebra::vector3<int> cell_grid::get_triplet(vector3<double>& pos)
{
	return algebra::vector3<int>(
		static_cast<int>( abs(std::floor((pos.x - m_wo.x) / m_cellSize)) ),
		static_cast<int>( abs(std::floor((pos.y - m_wo.y) / m_cellSize)) ),
		static_cast<int>( abs(std::floor((pos.z - m_wo.z) / m_cellSize)) )
		);
}

void cell_grid::detection(vector4<double>* pos)
{
	algebra::vector3<int> cell3d;

	// Hash value calculation
	for(unsigned int i = 0; i < sim->getParticles()->Size() + m_snp; i++){
		cell3d = get_triplet(pos[i].x, pos[i].y, pos[i].z);
		cell_id[i] = calcGridHash(cell3d);
		body_id[i] = i;
	}

	// sorting by key
	thrust::sort_by_key(cell_id, cell_id + m_np + m_snp, body_id);
	memset(cell_start, 0xffffffff, sizeof(size_t)*nGrid);
	memset(cell_end, 0, sizeof(size_t)*nGrid);
	size_t begin = 0;
	size_t end = 0;
	size_t id = 0;
	bool ispass;


	while(end++ != m_np + m_snp){
		ispass = true;
		id = cell_id[begin];
		if(id != cell_id[end]){
			end - begin > 1 ? ispass = false : reorderDataAndFindCellStart(id, begin++, end);
		}
		if(!ispass){
			reorderDataAndFindCellStart(id, begin, end);
			begin = end;
		}
	}
}

void cell_grid::shape_detection(algebra::vector3<double>* vertice)
{
	//algebra::vector3<int> cell3d;
	//for(unsigned int i = )
}

void cell_grid::cu_detection(double* pos)
{
	cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, m_np + m_snp);
	cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, m_np + m_snp, nGrid);
}

void cell_grid::detection()
{
	std::map<std::string, geometry*>::iterator Geo;
	if(dem_sim->Device() == GPU){
		double3* vertice = NULL;
		if(mbd_sim){
			for(Geo = mbd_sim->getGeometries()->begin(); Geo != mbd_sim->getGeometries()->end(); Geo++){
				if(Geo->second->Geometry() == SHAPE){
					geo::shape *Shape = dynamic_cast<geo::shape*>(Geo->second);
					vertice = Shape->getCuVertice();
				}
			}
		}
		cu_mergedata(d_data, dem_sim->getParticles()->cu_Position(), m_np, vertice, m_snp);
		cu_calculateHashAndIndex(d_cell_id, d_body_id, d_data, m_np + m_snp);
		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, m_np + m_snp, nGrid);
	}
}