#include "neighborhood_cell.h"
#include "simulation.h"
#include "thrust/sort.h"
#include "mphysics_cuda_dec.cuh"
#include <QDebug>

neighborhood_cell::neighborhood_cell()
	: grid_base(NEIGHBORHOOD)
	, rearranged_id(NULL)
	, d_rearranged_id(NULL)
{

}

neighborhood_cell::~neighborhood_cell()
{
	if (rearranged_id) delete[] rearranged_id; rearranged_id = NULL;
	if (d_rearranged_id) checkCudaErrors(cudaFree(d_rearranged_id)); d_rearranged_id = NULL;
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

void neighborhood_cell::_detection(VEC4D_PTR pos, VEC4D_PTR spos, unsigned int np, unsigned int snp)
{
	//VEC4D_PTR pos = md->particleSystem()->position();
	VEC4D *psph = NULL;
	VEC3I cell3d;
//	unsigned int _np = 0;
// 	if (md->particleSystem()->particleCluster().size())
// 		_np = md->particleSystem()->particleCluster().size() * particle_cluster::perCluster();
// 	else
// 		_np = md->numParticle();
	for (unsigned int i = 0; i < np + snp; i++){
		unsigned int rid = rearranged_id[i];
		VEC4D p = rid >= np ? spos[rid - np] : pos[rid];
		cell3d = getCellNumber(p.x, p.y, p.z);
		cell_id[i] = getHash(cell3d);
		body_id[i] = rearranged_id[i];
	}
// 	if (spos)
// 	{
// 		for (unsigned int i = 0; i < snp; i++){
// 			cell3d = getCellNumber(spos[i].x, spos[i].y, spos[i].z);
// 			cell_id[np + i] = getHash(cell3d);
// 			body_id[np + i] = np + i;
// 		}
// 	}
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

	thrust::sort_by_key(cell_id, cell_id + np + snp, body_id);
	memset(cell_start, 0xffffffff, sizeof(unsigned int) * ng);
	memset(cell_end, 0, sizeof(unsigned int)*ng);
	unsigned int begin = 0, end = 0, id = 0;
	bool ispass;
	while (end++ != np + snp/*md->numParticle()*/){
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

void neighborhood_cell::_detection_f(VEC4F_PTR pos, VEC4F_PTR spos, unsigned int np, unsigned int snp)
{
	
}

void neighborhood_cell::detection(double *pos, double* spos, unsigned int np, unsigned int snp)
{
	if (simulation::isGpu())
	{
		cu_calculateHashAndIndex(d_cell_id, d_body_id, d_rearranged_id, pos, spos, np, snp);
	//	qDebug() << "detection0 done";
// 		if (snp && spos)
// 		{
// 			cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, np, snp, spos);
// 		//	qDebug() << "detection1 done";
// 		}
		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, np + snp,/* md->numPolygonSphere(),*/ ng);
		//qDebug() << "detection2 done";
	}
	else
		_detection((VEC4D_PTR)pos, (VEC4D_PTR)spos, np, snp);
}

void neighborhood_cell::detection_f(float *pos /*= NULL*/, float* spos /*= NULL*/, unsigned int np /*= 0*/, unsigned int snp /*= 0*/)
{
	if (simulation::isGpu())
	{
		cu_calculateHashAndIndex(d_cell_id, d_body_id, pos, np);
		if (snp && spos)
			cu_calculateHashAndIndexForPolygonSphere(d_cell_id, d_body_id, np, snp, spos);
		cu_reorderDataAndFindCellStart(d_cell_id, d_body_id, d_cell_start, d_cell_end, d_sorted_id, np + snp,/* md->numPolygonSphere(),*/ ng);
	}
	else
		_detection_f((VEC4F_PTR)pos, (VEC4F_PTR)spos, np, snp);
}

void neighborhood_cell::reorderElements(double *p, double *sp, unsigned int np, unsigned int snp)
{
	if (!rearranged_id)
		rearranged_id = new unsigned int[np + snp];
	VEC4D_PTR pos = (VEC4D_PTR)p;
	VEC4D_PTR psph = (VEC4D_PTR)sp;
	VEC3I cell3d;
	for (unsigned int i = 0; i < np; i++){
		cell3d = getCellNumber(pos[i].x, pos[i].y, pos[i].z);
		cell_id[i] = getHash(cell3d);
		rearranged_id[i] = i;
	}
	if (sp)
	{
		for (unsigned int i = 0; i < snp; i++){
			cell3d = getCellNumber(psph[i].x, psph[i].y, psph[i].z);
			cell_id[np + i] = getHash(cell3d);
			rearranged_id[np + i] = np + i;
		}
	}
	
	thrust::sort_by_key(cell_id, cell_id + np + snp, rearranged_id);
	if (simulation::isGpu())
	{
		checkCudaErrors(cudaMalloc((void**)&d_rearranged_id, sizeof(unsigned int) * (np + snp)));
		checkCudaErrors(cudaMemcpy(d_rearranged_id, rearranged_id, sizeof(unsigned int) * (np + snp), cudaMemcpyHostToDevice));
	}
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