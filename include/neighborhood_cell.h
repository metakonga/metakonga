#ifndef NEIGHBORHOOD_CELL_H
#define NEIGHBORHOOD_CELL_H
 
#include "grid_base.h"
 
class neighborhood_cell : public grid_base
{
public:
	neighborhood_cell();
	virtual ~neighborhood_cell();

	virtual void detection(double *pos = NULL, double* spos = NULL, unsigned int np = 0, unsigned int snp = 0);
	virtual void detection_f(float *pos = NULL, float* spos = NULL, unsigned int np = 0, unsigned int snp = 0);

	void reorderElements(double *pos, double *spos, unsigned int np, unsigned int snp);

private:
	void _detection(VEC4D_PTR pos, VEC4D_PTR spos, unsigned int np, unsigned int snp);
	void _detection_f(VEC4F_PTR pos, VEC4F_PTR spos, unsigned int np, unsigned int snp);
	void reorderDataAndFindCellStart(unsigned int id, unsigned int begin, unsigned int end);

	unsigned int *rearranged_id;
	unsigned int *d_rearranged_id;
};

#endif