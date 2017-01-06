#ifndef PARSIM_H
#define PARSIM_H

static unsigned int cRun = 0;

#include <stdio.h>

// utilities
#include "algebra.h"
#include "checkErrors.h"
#include "Log.h"
#include "writer.h"
#include "timer.h"

// MBD
#include "constraintCalculator.h"
#include "massCalculator.h"
#include "forceCalculator.h"

// simulation
#include "Demsimulation.h"
#include "Mbdsimulation.h"
#include "geometry.h"
#include "particles.h"
#include "dem_force.h"
#include "cell_grid.h"
#include "rigid_body.h"
#include "kinematic_constraint.h"
#include "pointmass.h"

// loader
//#include "parSIM_xmlloader.h"

// integrator
#include "Euler_integrator.h"
#include "Verlet_integrator.h"
#include "HHT_integrator.h"

// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#endif