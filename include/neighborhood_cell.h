#ifndef NEIGHBORHOOD_CELL
#define NEIGHBORHOOD_CELL

#include "grid_base.h"

class neighborhood_cell : public grid_base
{
public:
	neighborhood_cell();
	neighborhood_cell(std::string _name, modeler* _md);
	~neighborhood_cell();

	virtual void detection();
	virtual void cuDetection();

	void reorderElements(bool isCpu);

private:
	
	void reorderDataAndFindCellStart(unsigned int id, unsigned int begin, unsigned int end);
};

#endif