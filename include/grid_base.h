#ifndef GRID_BASE_H
#define GRID_BASE_H

#include <iostream>
#include <string>
#include "mphysics_numeric.h"
#include "grid_base.cuh"

class modeler;

class grid_base
{
public:
	grid_base();
	grid_base(std::string _name, modeler* _md);
	virtual ~grid_base();

	void clear();
	virtual void detection() = 0;
	virtual void cuDetection() = 0;

	void setWorldOrigin(VEC3F _wo) { wo = _wo; }
	void setCellSize(float _cs) { cs = _cs; }
	void setGridSize(VEC3UI _gs) { gs = _gs; }
	unsigned int nCell() { return ng; }
	void allocMemory(unsigned int n);
	void cuAllocMemory(unsigned int n);


	static VEC3I getCellNumber(float x, float y, float z);
	static VEC3I getCellNumber(double x, double y, double z);
	static unsigned int getHash(VEC3I& c3);
	static unsigned int sortedID(unsigned int id) { return sorted_id[id]; }
	static unsigned int cellID(unsigned int id) { return cell_id[id]; }
	static unsigned int bodyID(unsigned int id) { return body_id[id]; }
	static unsigned int cellStart(unsigned int id) { return cell_start[id]; }
	static unsigned int cellEnd(unsigned int id) { return cell_end[id]; }

	unsigned int* cuSortedID() { return d_sorted_id; }
	unsigned int* cuCellStart() { return d_cell_start; }
	unsigned int* cuCellEnd() { return d_cell_end; }

	static VEC3F wo;			// world origin
	static float cs;			// cell size
	static VEC3UI gs;			// grid size

protected:
	std::string name;

	static unsigned int* sorted_id;
	static unsigned int* cell_id;
	static unsigned int* body_id;
	static unsigned int* cell_start;
	static unsigned int* cell_end;

	unsigned int *d_sorted_id;
	unsigned int *d_cell_id;
	unsigned int *d_body_id;
	unsigned int *d_cell_start;
	unsigned int *d_cell_end;

	unsigned int nse;   // number of sorting elements
	unsigned int ng;	// the number of grid

	modeler *md;
};

#endif