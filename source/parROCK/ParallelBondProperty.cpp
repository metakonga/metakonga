#include "ParallelBondProperty.h"

ParallelBondProperty::ParallelBondProperty()
	: kn(0)
	, ks(0)
	, normalStrength(0)
	, shearStrength(0)
	, thick(1.0)
	, radius(0.0)
	, broken(false)
	//, nforce(0.0)
	//, sforce(0.0)
{

}

ParallelBondProperty::~ParallelBondProperty()
{

}