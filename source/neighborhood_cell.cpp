#include "neighborhood_cell.h"
#include "simulation.h"
#include "thrust/sort.h"
#include "mphysics_cuda_dec.cuh"

neighborhood_cell::neighborhood_cell()
	: grid_base(NEIGHBORHOOD)
{

}

neighborhood_cell::~neighborhood_cell()
{

}

void neighborhood_cell::reorderDataAndFindCellStart(unsigned int id, unsigned int begin, unsigned int end)
{
	cell_start[id] = begin;
	cell_end[id] = end;
	unsigned dim = 0, bid = 0;
	for (unsigned i(begin); i < end; i++){
		sorted_id[i] = body_id[i];
	}
}

void neighborhood_cell::_detection(VEC4D_PTR pos, unsigned int np)
{
	//VEC4D_PTR pos = md->particleSystem()->position();
	VEC4D *psph = NULL;
	VEC3I cell3d;
//	unsigned int _np = 0;
// 	if (md->particleSystem()->particleCluster().size())
// 		_np = md->particleSystem()->particleCluster().size() * particle_cluster::perCluster();
// 	else
// 		_np = md->numParticle();
	for (unsigned int i = 0; i < np; i++){
		cell3d = getCellNumber(pos[i].x, pos[i].y, pos[i].z);
		cell_id[i] = getHash(cell3d);
		body_id[i] = i;
	}
// 	unsigned int sid = md->numParticle();
// 	QList<polygonObject*> polys = md->polyObjects();
// 	for (unsigned int po = 0; po < md->numPoly(); po++){
// 		psph = polys.at(po)->hostSphereSet();
// 		for (unsigned int i = 0; i < polys.at(po)->numIndex(); i++){
// 			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
// 			cell_id[sid + i] = getHash(cell3d);
// 			body_id[sid + i] = sid + i;
// 		}
// 		sid += polys.at(po)->numIndex();
// 	}

	thrust::sort_by_key(cell_id, cell_id + np, body_id);
	memset(cell_start, 0xffffffff, sizeof(unsigned int) * ng);
	memset(cell_end, 0, sizeof(unsigned int)*ng);
	unsigned int begin = 0, end = 0, id = 0;
	bool ispass;
	while (end++ != np/*md->numParticle()*/){
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
}

void neighborhood_cell::detection(double *pos, unsigned int np)
{
	if (simulation::isGpu())
	{
		cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, np);
		//std::cout << "step 2" << std::endl;

// 		for (unsigned int i = 0; i < md->polyObjects().size(); i++){
// 			cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, np, md->polyObjects().at(i)->numIndex(), md->polyObjects().at(i)->deviceSphereSet());
// 		}
		//std::cout << "step 3" << std::endl;
		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, np,/* md->numPolygonSphere(),*/ ng);
		//std::cout << "step 4" << std::endl
	}
	else
		_detection((VEC4D_PTR)pos, np);
}

void neighborhood_cell::reorderElements(bool isCpu)
{
	
// 	VEC4D_PTR pos = md->particleSystem()->position();
// 	VEC4D *psph = NULL;
// 	unsigned int np = md->numParticle();
// 	allocMemory(np + md->numPolygonSphere());
// 	VEC3I cell3d;
// 	//bool isExistPolygonObject = false;
// 	for (unsigned int i = 0; i < np; i++){
// 		cell3d = getCellNumber(pos[i].x, pos[i].y, pos[i].z);
// 		cell_id[i] = getHash(cell3d);
// 		body_id[i] = i;
// 	}
// 	unsigned int sid = np;
// 	VEC4D *sphPos = NULL;
// 	host_polygon_info *hpi = NULL;
// 	unsigned int nsph = md->numPolygonSphere();
// 	if (nsph){
// 		sphPos = new VEC4D[nsph];
// 		hpi = new host_polygon_info[nsph];
// 	}
// 	QList<polygonObject*> polys = md->polyObjects();
// 	for (unsigned int j = 0; j < polys.size(); j++){
// 	//for (QMap<QString, polygonObject>::iterator po = md->objPolygon().begin(); po != md->objPolygon().end(); po++){
// 		psph = polys.at(j)->hostSphereSet();
// 		host_polygon_info* _hpi = polys.at(j)->hostPolygonInfo();
// 		for (unsigned int i = 0; i < polys.at(j)->numIndex(); i++){
// 			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
// 			cell_id[sid + i] = getHash(cell3d);
// 			body_id[sid + i] = sid + i;
// 			sphPos[sid + i - np] = psph[i];
// 			hpi[sid + i - np] = _hpi[i];
// 		}
// 		sid += polys.at(j)->numIndex();
// 	}
// 
// 	thrust::sort_by_key(cell_id, cell_id + nse/*md->numParticle()*/, body_id);
// 	VEC4D *pPos = new VEC4D[md->numParticle()];
// 	memcpy(pPos, pos, sizeof(VEC4D) * md->numParticle());
// // 	VEC4D *sphPos = NULL;
// // 	if (sid != md->numParticle())
// // 	{
// // 		sphPos = new VEC4D[nse - md->numParticle()];
// // 		memcpy(sphPos, )
// // 	}
// 	unsigned int pcnt = 0;
// 	unsigned int *pocnt = new unsigned int[polys.size()];
// 	memset(pocnt, 0, sizeof(unsigned int) * polys.size());
// 	for (unsigned int i = 0; i < nse; i++){
// 		if (body_id[i] < md->numParticle())
// 		{
// 			pos[pcnt] = pPos[body_id[i]];
// 			pcnt++;
// 		}
// 		else
// 		{
// 			unsigned int cnt = 0;
// 			unsigned int start = np;
// 			unsigned int bid = body_id[i];
// 			for (unsigned int j = 0; j < polys.size(); j++){
// 			//for (QMap<QString, polygonObject>::iterator po = md->objPolygon().begin(); po != md->objPolygon().end(); po++){
// 				if (bid < start + polys.at(j)->numIndex()){
// 					unsigned int rid = pocnt[cnt]++;
// 					polys.at(j)->hostPolygonInfo()[rid] = hpi[bid - np];
// 					polys.at(j)->hostSphereSet()[rid] = sphPos[bid - np];
// 				}
// 				cnt++;
// 				start += polys.at(j)->numIndex();
// 			}
// 		}
// 	}
// 	delete [] pocnt;
// 	for (unsigned int i = 0; i < polys.size(); i++){
// 	//for (QMap<QString, polygonObject>::iterator po = md->objPolygon().begin(); po != md->objPolygon().end(); po++){
// 		polys.at(i)->updateDeviceFromHost();
// 	}
// 	//md->objPolygon().begin().value().updateDeviceFromHost();
// // 	foreach (polygonObject value, md->objPolygon())
// // 	{
// // 		value.updateDeviceFromHost();
// // 	}
// // 	foreach(polygonObject value, md->objPolygon()){
// // 		psph = value.hostSphereSet();
// // 		for (unsigned int i = 0; i < value.numIndex(); i++){
// // 			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
// // 			cell_id[sid + i] = getHash(cell3d);
// // 			body_id[sid + i] = sid + i;
// // 			sphPos[sid + i] = psph[i];
// // 		}
// // 		sid += value.numIndex();
// // 	}
// 
// 	if (cell_id) delete[] cell_id; cell_id = NULL;
// 	if (body_id) delete[] body_id; body_id = NULL;
// 	if (sorted_id) delete[] sorted_id; sorted_id = NULL;
// 	if (cell_start) delete[] cell_start; cell_start = NULL;
// 	if (cell_end) delete[] cell_end; cell_end = NULL;
// 
// 	if (hpi)
// 		delete[] hpi;
// 	if (sphPos)
// 		delete[] sphPos;
// 	if (pPos)
// 		delete[] pPos;
 }