#ifndef GRID_BASE_H
#define GRID_BASE_H

#include <QString>
#include "vectorTypes.h"

class grid_base
{
public:
	enum Type{ NEIGHBORHOOD };
	grid_base();
	grid_base(Type t);
	virtual ~grid_base();

	void clear();
	//virtual void detection(VEC4D_PTR pos) = 0;
	void initialize(unsigned int np);
	virtual void detection(double *pos, unsigned int np) = 0;

	void setWorldOrigin(VEC3D _wo) { wo = _wo; }
	void setCellSize(float _cs) { cs = _cs; }
	void setGridSize(VEC3UI _gs) { gs = _gs; }
	unsigned int nCell() { return ng; }
	void allocMemory(unsigned int n);
	void cuAllocMemory(unsigned int n);
	//void cuResizeMemory(unsigned int n);


//	static VEC3I getCellNumber(double x, double y, double z);
	static VEC3I getCellNumber(double x, double y, double z);
	static unsigned int getHash(VEC3I& c3);
	unsigned int sortedID(unsigned int id) { return sorted_id[id]; }
	unsigned int cellID(unsigned int id) { return cell_id[id]; }
	unsigned int bodyID(unsigned int id) { return body_id[id]; }
	unsigned int cellStart(unsigned int id) { return cell_start[id]; }
	unsigned int cellEnd(unsigned int id) { return cell_end[id]; }
	unsigned int* sortedID() { return d_sorted_id; }
	unsigned int* cellStart() { return d_cell_start; }
	unsigned int* cellEnd() { return d_cell_end; }

	static VEC3D wo;			// world origin
	static double cs;			// cell size
	static VEC3UI gs;			// grid size

protected:
	Type type;
	unsigned int* sorted_id;
	unsigned int* cell_id;
	unsigned int* body_id;
	unsigned int* cell_start;
	unsigned int* cell_end;

	unsigned int *d_sorted_id;
	unsigned int *d_cell_id;
	unsigned int *d_body_id;
	unsigned int *d_cell_start;
	unsigned int *d_cell_end;

	unsigned int nse;   // number of sorting elements
	unsigned int ng;	// the number of grid
};

#endif