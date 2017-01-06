#ifndef CELL_GRID_H
#define CELL_GRID_H

#include "algebra.h"
#include "thrust/sort.h"

namespace parSIM
{
	class Simulation;
	class cell_grid
	{
	public:
		cell_grid() {}
		cell_grid(Simulation *Sim);
		cell_grid(Simulation *demSim, Simulation *mbdSim);

		~cell_grid();

		void initialize();
		void setWorldOrigin(double x, double y, double z) { m_wo = algebra::vector3<double>(x,y,z); }
		void setGridSize(size_t x, size_t y, size_t z) { m_gs = algebra::vector3<size_t>(x,y,z); }
		void setCellSize(double cs) { m_cellSize = cs; }
		double* getCuMergedData() { return d_data; }
		size_t getCellStart(size_t id) { return cell_start[id]; }
		size_t getCellEnd(size_t id) {return cell_end[id]; }
		size_t getSortedIndex(size_t id) { return sorted_id[id]; }
		double getCellSize() { return m_cellSize; }
		unsigned int getNumCell() { return nGrid; }
		unsigned int getNumShapeVertex() { return m_snp; }
		vector3<unsigned int>& GridSize() { return m_gs; }
		vector3<double>& WorldOrigin() { return m_wo; }

		unsigned int* cu_getSortedID() { return d_sorted_id; }
		unsigned int* cu_getCellID() { return d_cell_id; }
		unsigned int* cu_getBodyID() { return d_body_id; }
		unsigned int* cu_getCellStart() { return d_cell_start; }
		unsigned int* cu_getCellEnd() { return d_cell_end; }

		unsigned int calcGridHash(algebra::vector3<int>& cell3d);
		void reorderDataAndFindCellStart(size_t ID, size_t begin, size_t end);

		algebra::vector3<int> get_triplet(double r1, double r2, double r3);
		algebra::vector3<int> get_triplet(vector3<double> &pos);

		void detection(vector4<double> *pos);
		void detection();
		void shape_detection(algebra::vector3<double>* vertice);
		void cu_detection(double* pos);

	private:
		Simulation *sim;
		Simulation *dem_sim;
		Simulation *mbd_sim;

		algebra::vector4<double> *data;
		double *d_data;

		size_t *sorted_id;
		size_t *cell_id;
		size_t *body_id;
		size_t *cell_start;
		size_t *cell_end;

		unsigned int *d_sorted_id;
		unsigned int *d_cell_id;
		unsigned int *d_body_id;
		unsigned int *d_cell_start;
		unsigned int *d_cell_end;

		size_t m_np;		// the number of particle
		unsigned int m_snp;  // the number of shape particle
		size_t nGrid;
		double m_cellSize;
		double m_radius;
		algebra::vector3<size_t> m_gs;		// grid size
		algebra::vector3<double> m_wo;		// world origin
	};
}

#endif