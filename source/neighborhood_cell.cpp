#include "neighborhood_cell.h"
#include "modeler.h"
#include "polygonObject.h"
#include "thrust/sort.h"
#include "mphysics_cuda_dec.cuh"

neighborhood_cell::neighborhood_cell()
	: grid_base()
{

}

neighborhood_cell::neighborhood_cell(std::string _name, modeler* _md)
	: grid_base(_name, _md)
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

void neighborhood_cell::detection()
{
	VEC4F_PTR pos = md->particleSystem()->position();
	VEC4D *psph = NULL;
	VEC3I cell3d;

	for (unsigned int i = 0; i < md->numParticle(); i++){
		cell3d = getCellNumber(pos[i].x, pos[i].y, pos[i].z);
		cell_id[i] = getHash(cell3d);
		body_id[i] = i;
	}
	unsigned int sid = md->numParticle();
	foreach(polygonObject value, md->objPolygon()){
		psph = value.hostSphereSet();
		for (unsigned int i = 0; i < value.numIndex(); i++){
			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
			cell_id[sid + i] = getHash(cell3d);
			body_id[sid + i] = sid + i;
		}
		sid += value.numIndex();
	}

	thrust::sort_by_key(cell_id, cell_id + nse/*md->numParticle()*/, body_id);
// 	std::fstream fs;
// 	fs.open("C:/C++/cpu_sorted_hash_index.txt", std::ios::out);
// 	for (unsigned int i = 0; i < nse; i++){
// 		fs << cell_id[i] << " " << body_id[i] << std::endl;
// 	}
// 	fs.close();
	memset(cell_start, 0xffffffff, sizeof(unsigned int) * ng);
	memset(cell_end, 0, sizeof(unsigned int)*ng);
	unsigned int begin = 0, end = 0, id = 0;
	bool ispass;
	while (end++ != nse/*md->numParticle()*/){
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

void neighborhood_cell::cuDetection()
{
	//std::cout << "step 1" << std::endl;
	cu_calculateHashAndIndex(d_cell_id, d_body_id, md->particleSystem()->cuPosition(), md->numParticle());
	//std::cout << "step 2" << std::endl;
	unsigned int sid = md->numParticle();
	for (QMap<QString, polygonObject>::iterator it = md->objPolygon().begin(); it != md->objPolygon().end(); it++){
		cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, sid, it.value().numIndex(), it.value().deviceSphereSet());
	}
	//std::cout << "step 3" << std::endl;
	cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, md->numParticle(), md->numPolygonSphere(), ng);
	//std::cout << "step 4" << std::endl;
}

void neighborhood_cell::reorderElements(bool isCpu)
{
	
	VEC4F_PTR pos = md->particleSystem()->position();
	VEC4D *psph = NULL;
	unsigned int np = md->numParticle();
	allocMemory(np + md->numPolygonSphere());
	VEC3I cell3d;
	//bool isExistPolygonObject = false;
	for (unsigned int i = 0; i < np; i++){
		cell3d = getCellNumber(pos[i].x, pos[i].y, pos[i].z);
		cell_id[i] = getHash(cell3d);
		body_id[i] = i;
	}
	unsigned int sid = np;
	VEC4D *sphPos = NULL;
	host_polygon_info *hpi = NULL;
	unsigned int nsph = md->numPolygonSphere();
	if (nsph){
		sphPos = new VEC4D[nsph];
		hpi = new host_polygon_info[nsph];
	}
	for (QMap<QString, polygonObject>::iterator po = md->objPolygon().begin(); po != md->objPolygon().end(); po++){
		psph = po.value().hostSphereSet();
		host_polygon_info* _hpi = po.value().hostPolygonInfo();
		for (unsigned int i = 0; i < po.value().numIndex(); i++){
			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
			cell_id[sid + i] = getHash(cell3d);
			body_id[sid + i] = sid + i;
			sphPos[sid + i - np] = psph[i];
			hpi[sid + i - np] = _hpi[i];
		}
		sid += po.value().numIndex();
	}

	thrust::sort_by_key(cell_id, cell_id + nse/*md->numParticle()*/, body_id);
	VEC4F *pPos = new VEC4F[md->numParticle()];
	memcpy(pPos, pos, sizeof(VEC4F) * md->numParticle());
// 	VEC4D *sphPos = NULL;
// 	if (sid != md->numParticle())
// 	{
// 		sphPos = new VEC4D[nse - md->numParticle()];
// 		memcpy(sphPos, )
// 	}
	unsigned int pcnt = 0;
	unsigned int *pocnt = new unsigned int[md->objPolygon().size()];
	memset(pocnt, 0, sizeof(unsigned int) * md->objPolygon().size());
	for (unsigned int i = 0; i < nse; i++){
		if (body_id[i] < md->numParticle())
		{
			pos[pcnt] = pPos[body_id[i]];
			pcnt++;
		}
		else
		{
			unsigned int cnt = 0;
			unsigned int start = np;
			unsigned int bid = body_id[i];
			for (QMap<QString, polygonObject>::iterator po = md->objPolygon().begin(); po != md->objPolygon().end(); po++){
				if (bid < start + po.value().numIndex()){
					unsigned int rid = pocnt[cnt]++;
					po.value().hostPolygonInfo()[rid] = hpi[bid - np];
					po.value().hostSphereSet()[rid] = sphPos[bid - np];
				}
				cnt++;
				start += po.value().numIndex();
			}
		}
	}
	delete [] pocnt;
	for (QMap<QString, polygonObject>::iterator po = md->objPolygon().begin(); po != md->objPolygon().end(); po++){
		po.value().updateDeviceFromHost();
	}
	//md->objPolygon().begin().value().updateDeviceFromHost();
// 	foreach (polygonObject value, md->objPolygon())
// 	{
// 		value.updateDeviceFromHost();
// 	}
// 	foreach(polygonObject value, md->objPolygon()){
// 		psph = value.hostSphereSet();
// 		for (unsigned int i = 0; i < value.numIndex(); i++){
// 			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
// 			cell_id[sid + i] = getHash(cell3d);
// 			body_id[sid + i] = sid + i;
// 			sphPos[sid + i] = psph[i];
// 		}
// 		sid += value.numIndex();
// 	}

	if (cell_id) delete[] cell_id; cell_id = NULL;
	if (body_id) delete[] body_id; body_id = NULL;
	if (sorted_id) delete[] sorted_id; sorted_id = NULL;
	if (cell_start) delete[] cell_start; cell_start = NULL;
	if (cell_end) delete[] cell_end; cell_end = NULL;

	if (hpi)
		delete[] hpi;
	if (sphPos)
		delete[] sphPos;
	if (pPos)
		delete[] pPos;
}