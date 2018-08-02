#ifndef NEIGHBORHOOD_CELL_H
#define NEIGHBORHOOD_CELL_H
 
#include "grid_base.h"
 
class neighborhood_cell : public grid_base
{
public:
	neighborhood_cell();
	virtual ~neighborhood_cell();

	virtual void detection(double *pos, unsigned int np);

	void reorderElements(bool isCpu);

private:
	void _detection(VEC4D_PTR pos, unsigned int np);
	void reorderDataAndFindCellStart(unsigned int id, unsigned int begin, unsigned int end);
};

#endif