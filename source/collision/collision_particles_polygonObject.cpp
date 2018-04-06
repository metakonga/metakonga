#include "collision_particles_polygonObject.h"
#include "particle_system.h"
#include "polygonObject.h"
#include "mass.h"

collision_particles_polygonObject::collision_particles_polygonObject()
{

}

collision_particles_polygonObject::collision_particles_polygonObject(
	QString& _name, 
	modeler* _md, 
	particle_system *_ps, 
	polygonObject * _poly, 
	tContactModel _tcm)
	: collision(_name, _md, _ps->name(), _poly->objectName(), PARTICLES_POLYGONOBJECT, _tcm)
	, ps(_ps)
	, po(_poly)
{

}

collision_particles_polygonObject::~collision_particles_polygonObject()
{

}

bool collision_particles_polygonObject::collid(double dt)
{
	return true;
}

bool collision_particles_polygonObject::cuCollid(
	double *dpos, double *dvel,
	double *domega, double *dmass,
	double *dforce, double *dmoment, unsigned int np)
{
	double3 *mforce;
	double3 *mmoment;
	double3 *mpos;
	VEC3D _mp;
	double3 _mf = make_double3(0.0, 0.0, 0.0);
	double3 _mm = make_double3(0.0, 0.0, 0.0);
	if (po->pointMass())
		_mp = po->pointMass()->getPosition();
	checkCudaErrors(cudaMalloc((void**)&mpos, sizeof(double3)));
	checkCudaErrors(cudaMemcpy(mpos, &_mp, sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&mforce, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMalloc((void**)&mmoment, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMemset(mforce, 0, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMemset(mmoment, 0, sizeof(double3)*ps->numParticle()));

	switch (tcm)
	{
	case HMCM: 
		cu_particle_polygonObject_collision(
			0, po->devicePolygonInfo(), po->deviceSphereSet(), po->deviceMassInfo(), 
			dpos, dvel, domega,
			dforce, dmoment, dmass,
			gb->cuSortedID(), gb->cuCellStart(), gb->cuCellEnd(), dcp, 
			ps->numParticle(), mpos, mforce, mmoment, _mf, _mm); 
		break;
	}
	
	_mf = reductionD3(mforce, ps->numParticle());
	if (po->pointMass()){
		po->pointMass()->addCollisionForce(VEC3D(_mf.x, _mf.y, _mf.z));
	}
	_mm = reductionD3(mmoment, ps->numParticle());
	if (po->pointMass()){
		po->pointMass()->addCollisionMoment(VEC3D(_mm.x, _mm.y, _mm.z));
	}
	checkCudaErrors(cudaFree(mforce)); mforce = NULL;
	checkCudaErrors(cudaFree(mmoment)); mmoment = NULL;
	checkCudaErrors(cudaFree(mpos)); mpos = NULL;
	return true;
}

VEC3D collision_particles_polygonObject::particle_polygon_contact_detection(host_polygon_info& hpi, VEC3D& p, double pr)
{
	VEC3D a = hpi.P.To<double>();
	VEC3D b = hpi.Q.To<double>();
	VEC3D c = hpi.R.To<double>();
	VEC3D ab = b - a;
	VEC3D ac = c - a;
	VEC3D ap = p - a;

	double d1 = ab.dot(ap);// dot(ab, ap);
	double d2 = ac.dot(ap);// dot(ac, ap);
	if (d1 <= 0.0 && d2 <= 0.0){
		//	*wc = 0;
		return a;
	}

	VEC3D bp = p - b;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if (d3 >= 0.0 && d4 <= d3){
		//	*wc = 0;
		return b;
	}
	double vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
		//	*wc = 1;
		double v = d1 / (d1 - d3);
		return a + v * ab;
	}

	VEC3D cp = p - c;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if (d6 >= 0.0 && d5 <= d6){
		//	*wc = 0;
		return c;
	}

	double vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
		//	*wc = 1;
		double w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
		//	*wc = 1;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	//*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;

	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	//return 0.f;
}

bool collision_particles_polygonObject::collid_with_particle(unsigned int i, double dt)
{
	double overlap = 0.f;
	VEC4D ipos = ps->position()[i];
	VEC3D ivel = ps->velocity()[i];
	VEC3D iomega = ps->angVelocity()[i];
	double ir = ipos.w;
	VEC3D m_moment = 0.f;
	VEC3I neighbour_pos = 0;
	unsigned int grid_hash = 0;
	VEC3D single_force = 0.f;
	VEC3D shear_force = 0.f;
	VEC3I gridPos = gb->getCellNumber(ipos.x, ipos.y, ipos.z);
	unsigned int sindex = 0;
	unsigned int eindex = 0;
	VEC3D ip = VEC3D(ipos.x, ipos.y, ipos.z);
	double ms = ps->mass()[i];
	unsigned int np = md->numParticle();
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = VEC3I(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = gb->getHash(neighbour_pos);
				sindex = gb->cellStart(grid_hash);
				if (sindex != 0xffffffff){
					eindex = gb->cellEnd(grid_hash);
					for (unsigned int j = sindex; j < eindex; j++){
						unsigned int k = gb->sortedID(j);
						if (k >= np)
						{
							k -= np;
							VEC3D cp = particle_polygon_contact_detection(po->hostPolygonInfo()[k], ip, ir);
							VEC3D distVec = ip - cp;
							double dist = distVec.length();
							overlap = ir - dist;
							if (overlap > 0)
							{
								VEC3D unit = -po->hostPolygonInfo()[k].N;
								VEC3D dv = -(ivel + iomega.cross(ir * unit));
								constant c = getConstant(ir, 0, ms, 0, ps->youngs(), po->youngs(), ps->poisson(), po->poisson(), ps->shear(), po->shear());
								double fsn = -c.kn * pow(overlap, 1.5);
								single_force = (fsn + c.vn * dv.dot(unit)) * unit;
								//std::cout << k << ", " << single_force.x << ", " << single_force.y << ", " << single_force.z << std::endl;
								VEC3D e = dv - dv.dot(unit) * unit;
								double mag_e = e.length();
								if (mag_e){
									VEC3D s_hat = e / mag_e;
									double ds = mag_e * dt;
									shear_force = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * single_force.length()) * s_hat;
									m_moment = (ir*unit).cross(shear_force);
								}
								ps->force()[i] += single_force;
								ps->moment()[i] += m_moment;
							}
						}
					}
				}
			}
		}
	}
	return true;
}