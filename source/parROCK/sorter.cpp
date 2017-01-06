#include "sorter.h"
#include "Simulation.h"
#include "contact.h"
#include "thrust/sort.h"
#include "ball.h"

sorter::sorter(Simulation* baseSimulation)
	: sim(baseSimulation)
	, sorted_id(NULL)
	, cell_id(NULL)
	, body_id(NULL)
	, cell_start(NULL)
	, cell_end(NULL)
{

}

sorter::~sorter()
{
	if (cell_id) delete[] cell_id; cell_id = NULL;
	if (body_id) delete[] body_id; body_id = NULL;
	if (sorted_id) delete[] sorted_id; sorted_id = NULL;
	if (cell_start) delete[] cell_start; cell_start = NULL;
	if (cell_end) delete[] cell_end; cell_end = NULL;
}

vector3<int> sorter::get_triplet(double r1, double r2, double r3)
{
	return algebra::vector3<int>(
		static_cast<int>(abs(std::floor((r1 - m_wo.x) / csize))),
		static_cast<int>(abs(std::floor((r2 - m_wo.y) / csize))),
		static_cast<int>(abs(std::floor((r3 - m_wo.z) / csize)))
		);
}

vector3<int> sorter::get_triplet(vector3<double>& pos)
{
	return algebra::vector3<int>(
		static_cast<int>(abs(std::floor((pos.x - m_wo.x) / csize))),
		static_cast<int>(abs(std::floor((pos.y - m_wo.y) / csize))),
		static_cast<int>(abs(std::floor((pos.z - m_wo.z) / csize)))
		);
}

unsigned int sorter::calcGridHash(vector3<int>& cell3d)
{
	vector3<int> gridPos;
	gridPos.x = cell3d.x & (m_gs.x - 1);
	gridPos.y = cell3d.y & (m_gs.y - 1);
	gridPos.z = cell3d.z & (m_gs.z - 1);
	return (gridPos.z*m_gs.y) * m_gs.x + (gridPos.y*m_gs.x) + gridPos.x;
}

void sorter::reorderDataAndFindCellStart(unsigned int ID, unsigned int begin, unsigned int end)
{
	cell_start[ID] = begin;
	cell_end[ID] = end;
	unsigned int dim = 0, bid = 0;
	for (unsigned int i(begin); i < end; i++){
		sorted_id[i] = body_id[i];
	}
}

bool sorter::initialize()
{
	unsigned int nb = ball::nballs;
	ncells = m_gs.x * m_gs.y * m_gs.z;
	if (!nb || !ncells){
		std::cout << "The number of ball or cell is zero." << std::endl;
		return false;
	}

	for (std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin(); it != sim->Geometries().end(); it++){
		if (it->second->Shape() == SHAPE){
			geo::Shape *shape = dynamic_cast<geo::Shape*>(it->second);
			nb += shape->Vertice()->sizes();
		}
	}
	cell_id = new size_t[nb];		memset(cell_id, 0, sizeof(size_t)*nb);
	body_id = new size_t[nb];		memset(body_id, 0, sizeof(size_t)*nb);
	sorted_id = new size_t[nb];	memset(sorted_id, 0, sizeof(size_t)*nb);
	cell_start = new size_t[ncells];	memset(cell_start, 0, sizeof(size_t)*ncells);
	cell_end = new size_t[ncells];	memset(cell_end, 0, sizeof(size_t)*ncells);

	csize = sim->CalMaxRadius() * 2;
	npoint = nb;
	return true;
}

bool sorter::sort()
{
	vector3<int> cell3d;
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball* b = &sim->Balls()[i];
		cell3d = get_triplet(b->Position());
		cell_id[b->ID()] = calcGridHash(cell3d);
		body_id[b->ID()] = b->ID();
	}
	for (std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin(); it != sim->Geometries().end(); it++){
		if (it->second->Shape() == SHAPE){
			geo::Shape *shape = dynamic_cast<geo::Shape*>(it->second);
			vector3<double>* vertice = shape->Vertice()->get_ptr();
			for (unsigned int i = 0; i < npoint - ball::nballs; i++){
				cell3d = get_triplet(vertice[i]);
				cell_id[ball::nballs + i] = calcGridHash(cell3d);
				body_id[ball::nballs + i] = ball::nballs + i;
			}
		}
	}
	thrust::sort_by_key(cell_id, cell_id + npoint, body_id);
	memset(cell_start, 0xffffffff, sizeof(unsigned int)*ncells);
	memset(cell_end, 0, sizeof(unsigned int) * ncells);

	unsigned int begin = 0;
	unsigned int end = 0;
	unsigned int id = 0;
	bool ispass;

	while (end++ != npoint){
		ispass = true;
		id = cell_id[begin];
		if (id != cell_id[end]){
			end - begin > 1 ? ispass = false : reorderDataAndFindCellStart(id, begin++, end);
		}
		if (!ispass){
			reorderDataAndFindCellStart(id, begin, end);
			begin = end;
		}
	}

	return true;
}

bool sorter::detect()
{
	//sim->cmap.clear();
	vector3<int> gp, np;
	vector3<double> rpos;
	double dist, cdist;
	unsigned int hash, sidx, eidx;
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball* ib = &sim->Balls()[i];
		gp = get_triplet(ib->Position());
		for (int z = -1; z <= 1; z++){
			for (int y = -1; y <= 1; y++){
				for (int x = -1; x <= 1; x++){
					np = vector3<int>(gp.x + x, gp.y + y, gp.z + z);
					hash = calcGridHash(np);
					sidx = CellStart(hash);
					if (sidx != 0xffffffff){

						eidx = CellEnd(hash);
						for (unsigned int i = sidx; i < eidx; i++){
							unsigned int k = SortedIndex(i);
							ball *jb = &sim->Balls()[k];
							if (ib == jb)
								continue;
							else if (k >= ball::nballs){
								for (std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin(); it != sim->Geometries().end(); it++){
									if (it->second->Shape() == SHAPE){
										it->second->Collision(ib, k - ball::nballs);
									}
								}
							}
							else{
								rpos = jb->Position() - ib->Position();
								dist = rpos.length();

								cdist = (ib->Radius() + jb->Radius()) - dist;
								if (ib->Collision(jb, cdist, rpos / dist)){
									//std::cout << "c.info[" << "]" << " : balls[" << ib->ID() << ", " << jb->ID() << "]" << std::endl;
								}
							}

						}
					}
				}
			}
		}
		for (std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin(); it != sim->Geometries().end(); it++){
			if (it->second->Shape() == SHAPE)
				continue;
			it->second->Collision(ib);
		}
		for (std::map<std::string, Object*>::iterator obj = sim->Objects().begin(); obj != sim->Objects().end(); obj++){
			//std::cout << i << std::endl;
			obj->second->Collision(ib);
		}
	}
	sim->ContactList().clear();
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball* b = &sim->Balls()[i];
		for (std::map<ball*, ccontact>::iterator it = b->ContactPMap().begin(); it != b->ContactPMap().end(); it++){
			if (it->first->ID() > b->ID()){
				sim->ContactList().push_back(&(it->second));
			}
		}
		for (std::map<Geometry*, ccontact>::iterator it = b->ContactWMap().begin(); it != b->ContactWMap().end(); it++){
			sim->ContactList().push_back(&(it->second));
		}
		for (std::map<Object*, ccontact>::iterator it = b->ContactOMap().begin(); it != b->ContactOMap().end(); it++){
			sim->ContactList().push_back(&(it->second));
		}
	}
	//std::cout << sim->ContactList().size() << std::endl;
	return true;
}

bool sorter::detectOnlyShape()
{

	//sim->cmap.clear();
	vector3<int> gp, np;
	vector3<double> rpos;
	bool isContact = false;
	//double dist, cdist;
	unsigned int hash, sidx, eidx;
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball* ib = &sim->Balls()[i];
		ib->ContactSMap().clear();
		gp = get_triplet(ib->Position());
		for (int z = -1; z <= 1; z++){
			if (isContact) break;
			for (int y = -1; y <= 1; y++){
				if (isContact) break;
				for (int x = -1; x <= 1; x++){
					if (isContact) break;
					np = vector3<int>(gp.x + x, gp.y + y, gp.z + z);
					hash = calcGridHash(np);
					sidx = CellStart(hash);
					if (sidx != 0xffffffff){
						eidx = CellEnd(hash);
						for (unsigned int i = sidx; i < eidx; i++){
							unsigned int k = SortedIndex(i);
							ball *jb = &sim->Balls()[k];
							if (ib == jb)
								continue;
							else if (k >= ball::nballs){
								for (std::map<std::string, Geometry*>::iterator it = sim->Geometries().begin(); it != sim->Geometries().end(); it++){
									if (it->second->Shape() == SHAPE){
										isContact = it->second->Collision(ib, k - ball::nballs);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	//sim->ContactList().clear();
	sim->ShapeContactList().clear();
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball* b = &sim->Balls()[i];
		for (std::map<Geometry*, ccontact>::iterator it = b->ContactSMap().begin(); it != b->ContactSMap().end(); it++){
			if (it->first->Shape() == SHAPE){
				sim->ShapeContactList().push_back(&(it->second));
			}
		}
	}
	return true;
}