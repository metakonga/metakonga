#include "s_particle.h"
#include "sphydrodynamics.h"

using namespace parsph;

s_particle::s_particle()
	: isInner(false)
	, isFreeSurface(false)
	, isFloating(false)
	, soundOfSpeed(0)
	, density_deriv(0)
	, eddyVisc(0.0)
{
	divP=0;
	mass=0;
	density=0;
	density_old = 0;
	density_temp = 0;
	pressure=0;
	pressure_old=0;
	pressure_temp=0;
	hydro_pressure = 0;
}

s_particle::~s_particle()
{

}