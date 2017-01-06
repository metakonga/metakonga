#include "constraintCalculator.h"
#include "Simulation.h"

using namespace parSIM;

constraintCalculator::constraintCalculator()
{

}

constraintCalculator::~constraintCalculator()
{

}

// void constraintCalculator::constraintJacobian(matrix<double>& lhs, POINTMASS *masses, KCONSTRAINT *kconsts, DCONSTRAINT *dconsts /* = 0 */)
// {
// 	int sRow=0;
// 	int iCol=0;
// 	int jCol=0;
// 	pointmass *ib = NULL;
// 	pointmass *jb = NULL;
// 	vector3d dij = make_vector3d(0.0);
// 	vector4d ep = make_vector4d(0.0);
// 	KCONSTRAINT::iterator kit = kconsts->begin();
// 	while(kit != kconsts->end()){
// 		sRow = kit->sRow;
// 		ib = kit->i; iCol = kit->iCol;
// 		jb = kit->j; jCol = kit->jCol;
// 		switch(kit->type){
// 		case kinematicConstraint::REVOLUTE:
// 			if(ib->ID!=-1)
// 			{
// 				ep = ib->parameter;
// 				lhs(sRow+0, iCol+0) = lhs(sRow+1, iCol+1) = lhs(sRow+2, iCol+2) = -1;
// 				lhs.insert(sRow+0, iCol+3, POINTER(B(ep, -kit->sp_i)), MAT3X4);
// 				lhs.insert(sRow+3, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->f_i)).w, 4);
// 				lhs.insert(sRow+4, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->g_i)).w, 4);
// 			}
// 			if(jb->ID!=-1)
// 			{
// 				ep = jb->parameter;
// 				lhs(sRow+0, jCol+0) = lhs(sRow+1, jCol+1) = lhs(sRow+2, jCol+2) = 1;
// 				lhs.insert(sRow+0, jCol+3, POINTER(B(ep, kit->sp_j)), MAT3X4);
// 				lhs.insert(sRow+3, jCol+3, &transpose(ib->toGlobal(kit->f_i), B(ep, kit->h_j)).w, 4);
// 				lhs.insert(sRow+4, jCol+3, &transpose(ib->toGlobal(kit->g_i), B(ep, kit->h_j)).w, 4);
// 			}
// 			break;
// 		case kinematicConstraint::SPHERICAL:
// 			if(ib->ID!=-1)
// 			{
// 				ep = ib->parameter;
// 				lhs(sRow+0, iCol+0) = lhs(sRow+1, iCol+1) = lhs(sRow+2, iCol+2) = -1.0;
// 				lhs.insert(sRow+0, iCol+3, POINTER(B(ep, -kit->sp_i)), MAT3X4);
// 			}
// 			if(jb->ID!=-1)
// 			{
// 				ep = jb->parameter;
// 				lhs(sRow+0, jCol+0) = lhs(sRow+1, jCol+1) = lhs(sRow+2, jCol+2) = 1.0;
// 				lhs.insert(sRow+0, jCol+3, POINTER(B(ep, kit->sp_j)), MAT3X4);
// 			}
// 			break;
// 		case kinematicConstraint::CYLINDERICAL:
// 			dij = jb->pos + jb->toGlobal(kit->sp_j) - ib->pos - ib->toGlobal(kit->sp_i);
// 			if(ib->ID!=-1)
// 			{
// 				ep = ib->parameter;
// 				lhs.insert(sRow+0, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->f_i)).w, 4);
// 				lhs.insert(sRow+1, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->g_i)).w, 4);
// 				lhs.insert(sRow+2, iCol+0, &(-jb->toGlobal(kit->f_i)).x, &transpose(dij + jb->toGlobal(kit->sp_i), B(ep, kit->f_i)).w, 34);
// 				lhs.insert(sRow+3, iCol+0, &(-jb->toGlobal(kit->g_i)).x, &transpose(dij + jb->toGlobal(kit->sp_i), B(ep, kit->g_i)).w, 34);
// 			}
// 			if(jb->ID!=-1)
// 			{
// 				ep = jb->parameter;
// 				vector3d gfi = ib->toGlobal(kit->f_i);
// 				vector3d ggi = ib->toGlobal(kit->g_i);
// 				lhs.insert(sRow+0, jCol+3, &transpose(gfi, B(ep, kit->h_j)).w, 4);
// 				lhs.insert(sRow+1, jCol+3, &transpose(ggi, B(ep, kit->h_j)).w, 4);
// 				lhs.insert(sRow+2, jCol+0, &gfi.x, &transpose(gfi, B(ep, kit->sp_j)).w, 34);
// 				lhs.insert(sRow+3, jCol+0, &ggi.x, &transpose(ggi, B(ep, kit->sp_j)).w, 34);
// 			}
// 			break;
// 		case kinematicConstraint::DRIVING_CYLINDERICAL:
// 			if(ib->ID!=-1)
// 			{
// 				ep = ib->parameter;
// 				lhs.insert(sRow+0, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->f_i)).w, 4);
// 				lhs.insert(sRow+1, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->g_i)).w, 4);
// 			}
// 			if(jb->ID!=-1)
// 			{
// 				ep = jb->parameter;
// 				vector3d gfi = ib->toGlobal(kit->f_i);
// 				vector3d ggi = ib->toGlobal(kit->g_i);
// 				lhs.insert(sRow+0, jCol+3, &transpose(gfi, B(ep, kit->h_j)).w, 4);
// 				lhs.insert(sRow+1, jCol+3, &transpose(ggi, B(ep, kit->h_j)).w, 4);
// 			}
// 			break;
// 		}
// 		kit++;
// 	}
// 	DCONSTRAINT::iterator dit = dconsts->begin();
// 	while(dit != dconsts->end()){
// 		kinematicConstraint *kconst = dit->getTargetJoint();
// 		ib = kconst->i;
// 		jb = kconst->j;
// 		dij = jb->pos + jb->toGlobal(kconst->sp_j) - ib->pos - ib->toGlobal(kconst->sp_i);
// 	
// 		sRow = dit->getStartRow();
// 		if(ib->ID!=-1)
// 		{
// 			ep = ib->parameter;
// 			lhs.insert(sRow, kconst->iCol, &(-ib->toGlobal(kconst->h_i)).x, &transpose(dij + ib->toGlobal(kconst->sp_i), B(ep, kconst->h_i)).w, 34);
// 		}
// 		if(jb->ID!=-1)
// 		{
// 			ep = jb->parameter;
// 			vector3d ghi = ib->toGlobal(kconst->h_i);
// 			lhs.insert(sRow, kconst->jCol, POINTER3(ghi), &transpose(ghi, B(ep, kconst->sp_j)).w, 34);
// 		}
// 		
// 		switch (dit->dType)
// 		{
// 			case drivingConstraint::TRANSLATIONAL: sRow+=1; break;
// 		default:
// 			break;
// 		}
// 		dit++;
// 	}
// 	//sRow++;
// 	POINTMASS::iterator mass = masses->begin();
// 	if(mass->ID==-1) mass++;
// 	while(mass != masses->end()){
// 		lhs.insert(sRow++, mass->ID*7 + 3, &(2*mass->parameter).w, 4);
// 		mass++;
// 	}
// 	lhs.exportDataToText();
//}

 void constraintCalculator::constraintJacobian(matrix<double>& Pi, std::map<std::string, pointmass*> *masses, std::map<std::string, kinematicConstraint*> *kconsts, std::map<std::string, drivingConstraint*> *dconsts /* = 0 */)
 {
//  	int sRow=0;
//  	int iCol=0;
//  	int jCol=0;
//  	int ib = 0;
//  	int jb = 0;
// 	lhs.zeros();
//  	vector3d dij = make_vector3d(0.0);
//  	//vector4d ep = make_vector4d(0.0);
// 	double *ep = NULL;
//  	KCONSTRAINT::iterator kit = kconsts->begin();
//  	while(kit != kconsts->end()){
//  		sRow = kit->sRow;
//  		ib = kit->i; iCol = kit->iCol;
//  		jb = kit->j; jCol = kit->jCol;
//  		switch(kit->type){
//  		case kinematicConstraint::REVOLUTE:
//  			if(ib)
//  			{
//  				ep = masses->getParameterp(ib);//ib->parameter;
//  				lhs(sRow+0, iCol+0) = lhs(sRow+1, iCol+1) = lhs(sRow+2, iCol+2) = -1;
//  				lhs.insert(sRow+0, iCol+3, POINTER(B(ep, -kit->sp_i)), MAT3X4);
//  				lhs.insert(sRow+3, iCol+3, &transpose(masses->toGlobal(jb, kit->h_j), B(ep, kit->g_i)).w, 4);
//  				lhs.insert(sRow+4, iCol+3, &transpose(masses->toGlobal(jb, kit->h_j), B(ep, kit->f_i)).w, 4);
//  			}
//  			if(jb)
//  			{
//  				ep = masses->getParameterp(jb);
//  				lhs(sRow+0, jCol+0) = lhs(sRow+1, jCol+1) = lhs(sRow+2, jCol+2) = 1;
//  				lhs.insert(sRow+0, jCol+3, POINTER(B(ep, kit->sp_j)), MAT3X4);
//  				lhs.insert(sRow+3, jCol+3, &transpose(masses->toGlobal(ib, kit->g_i), B(ep, kit->h_j)).w, 4);
//  				lhs.insert(sRow+4, jCol+3, &transpose(masses->toGlobal(ib, kit->f_i), B(ep, kit->h_j)).w, 4);
//  			}
//  			break;
//  		case kinematicConstraint::SPHERICAL:
//  			if(ib)
//  			{
//  				ep = masses->getParameterp(ib);
//  				lhs(sRow+0, iCol+0) = lhs(sRow+1, iCol+1) = lhs(sRow+2, iCol+2) = -1.0;
//  				lhs.insert(sRow+0, iCol+3, POINTER(B(ep, -kit->sp_i)), MAT3X4);
//  			}
//  			if(jb)
//  			{
//  				ep = masses->getParameterp(jb);
//  				lhs(sRow+0, jCol+0) = lhs(sRow+1, jCol+1) = lhs(sRow+2, jCol+2) = 1.0;
//  				lhs.insert(sRow+0, jCol+3, POINTER(B(ep, kit->sp_j)), MAT3X4);
//  			}
//  			break;
//  		case kinematicConstraint::CYLINDERICAL:
//  			dij = masses->getPositionv(jb) + masses->toGlobal(jb, kit->sp_j) - masses->getPositionv(ib) - masses->toGlobal(ib, kit->sp_i);
//  			if(ib)
//  			{
//  				ep = masses->getParameterp(ib);
//  				lhs.insert(sRow+0, iCol+3, &transpose(masses->toGlobal(jb, kit->h_j), B(ep, kit->f_i)).w, 4);
//  				lhs.insert(sRow+1, iCol+3, &transpose(masses->toGlobal(jb, kit->h_j), B(ep, kit->g_i)).w, 4);
//  				lhs.insert(sRow+2, iCol+0, &(-masses->toGlobal(ib, kit->g_i)).x, &transpose(dij + masses->toGlobal(ib, kit->sp_i), B(ep, kit->g_i)).w, 34);
//  				lhs.insert(sRow+3, iCol+0, &(-masses->toGlobal(ib, kit->f_i)).x, &transpose(dij + masses->toGlobal(ib, kit->sp_i), B(ep, kit->f_i)).w, 34);
//  			}
//  			if(jb)
//  			{
//  				ep = masses->getParameterp(jb);
//  				vector3d gfi = masses->toGlobal(ib, kit->f_i);
//  				vector3d ggi = masses->toGlobal(ib, kit->g_i);
//  				lhs.insert(sRow+0, jCol+3, &transpose(ggi, B(ep, kit->h_j)).w, 4);
//  				lhs.insert(sRow+1, jCol+3, &transpose(gfi, B(ep, kit->h_j)).w, 4);
//  				lhs.insert(sRow+2, jCol+0, &gfi.x, &transpose(ggi, B(ep, kit->sp_j)).w, 34);
//  				lhs.insert(sRow+3, jCol+0, &ggi.x, &transpose(gfi, B(ep, kit->sp_j)).w, 34);
//  			}
//  			break;
//  		case kinematicConstraint::DRIVING_CYLINDERICAL:
// //  			if(ib->ID!=-1)
// //  			{
// //  				ep = ib->parameter;
// //  				lhs.insert(sRow+0, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->f_i)).w, 4);
// //  				lhs.insert(sRow+1, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->g_i)).w, 4);
// //  			}
// //  			if(jb->ID!=-1)
// //  			{
// //  				ep = jb->parameter;
// //  				vector3d gfi = ib->toGlobal(kit->f_i);
// //  				vector3d ggi = ib->toGlobal(kit->g_i);
// //  				lhs.insert(sRow+0, jCol+3, &transpose(gfi, B(ep, kit->h_j)).w, 4);
// //  				lhs.insert(sRow+1, jCol+3, &transpose(ggi, B(ep, kit->h_j)).w, 4);
// //  			}
//  			break;
//  		}
//  		kit++;
//  	}
//  	DCONSTRAINT::iterator dit = dconsts->begin();
//  	while(dit != dconsts->end()){
//  		kinematicConstraint *kconst = dit->getTargetJoint();
//  		ib = kconst->i;
//  		jb = kconst->j;
//  		dij = masses->getPositionv(jb) + masses->toGlobal(jb, kconst->sp_j) - masses->getPositionv(ib) - masses->toGlobal(ib, kconst->sp_i);
//  	
//  		sRow = dit->getStartRow();
//  		if(ib)
//  		{
//  			ep = masses->getParameterp(ib);
//  			lhs.insert(sRow, kconst->iCol, &(-masses->toGlobal(ib, kconst->h_i)).x, &transpose(dij + masses->toGlobal(ib, kconst->sp_i), B(ep, kconst->h_i)).w, 34);
//  		}
//  		if(jb)
//  		{
//  			ep = masses->getParameterp(jb);
//  			vector3d ghi = masses->toGlobal(ib, kconst->h_i);
//  			lhs.insert(sRow, kconst->jCol, POINTER3(ghi), &transpose(ghi, B(ep, kconst->sp_j)).w, 34);
//  		}
//  		
//  		//sRow++;
//  		dit++;
//  	}
//  	//sRow++;
// 	sRow = lhs.rows() - masses->size() + 1;
//  	//POINTMASS::iterator mass = masses->begin();
//  	//if(mass->ID==-1) mass++;
//  	//while(mass != masses->end()){
// 	for(int i(0); i < masses->size()-1; i++){
//  		lhs.insert(sRow++, i*7 + 3, &(2*masses->getParameterv(i+1)).w, 4);
//  	}
//  	//lhs.exportDataToText();
}

void constraintCalculator::constraintEquation(algebra::vector<double>& rhs, std::map<std::string, pointmass*> *masses, std::map<std::string, kinematicConstraint*> *kconsts, std::map<std::string, drivingConstraint*> *dconsts /* = 0 */, double t /* = 0 */)
{
//  	vector3d v3 = make_vector3d(0.0);
//  	int ib = 0;
//  	int jb = 0;
//  	KCONSTRAINT::iterator kit = kconsts->begin();
//  	while(kit != kconsts->end()){
//  		ib = kit->i;
//  		jb = kit->j;
//  		switch(kit->type){
//  		case kinematicConstraint::REVOLUTE:
//  			v3 = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kit->sp_i);
//  			rhs.insert(kit->sRow, POINTER3(v3), 3); 
//  			v3 = m->toGlobal(jb, kit->h_j);
//  			rhs(kit->sRow+3) = dot(v3, m->toGlobal(ib, kit->f_i));
//  			rhs(kit->sRow+4) = dot(v3, m->toGlobal(ib, kit->g_i));
//  			break;
//  		case kinematicConstraint::SPHERICAL:
//  			v3 = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kit->sp_i);
//  			rhs.insert(kit->sRow, POINTER3(v3), 3);
//  			break;
//  		case kinematicConstraint::CYLINDERICAL:
// 			v3 = m->toGlobal(jb, kit->h_j);
// 			rhs(kit->sRow+0) = dot(v3, m->toGlobal(ib, kit->f_i));
// 			rhs(kit->sRow+1) = dot(v3, m->toGlobal(ib, kit->g_i));
// 			v3 = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib);
// 			rhs(kit->sRow+2) = dot(v3, m->toGlobal(ib, kit->f_i)) - dot(kit->sp_i, kit->f_i);
// 			rhs(kit->sRow+3) = dot(v3, m->toGlobal(ib, kit->g_i)) - dot(kit->sp_i, kit->g_i);
// 			break;
// // 		case kinematicConstraint::DRIVING_CYLINDERICAL:
// // 			v3 = jb->toGlobal(kit->h_j);
// // 			rhs(kit->sRow+0) = dot(v3, ib->toGlobal(kit->f_i));
// // 			rhs(kit->sRow+1) = dot(v3, ib->toGlobal(kit->g_i));
// // 			break;
//  		}
//  		kit++;
//  	}
// 	DCONSTRAINT::iterator dit = dconsts->begin();
// 	while(dit != dconsts->end()){
// 		kinematicConstraint *kconst = dit->getTargetJoint();
// 		ib = kconst->i;
// 		jb = kconst->j;
// 		v3 = m->getPositionv(jb) + m->toGlobal(jb, kconst->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kconst->sp_i);
// 		double dd = dot(v3, m->toGlobal(ib, kconst->h_i));
// 		vector3d ghi = m->toGlobal(ib, kconst->h_i);
// 		rhs(dit->getStartRow()) = /*dot(v3, ib->toGlobal(kconst->h_i)) - dot(kconst->sp_i, kconst->h_i)*/dd + dit->driving(t);
// 		dit++;
// 	}
// 	int mRow = rhs.sizes() - m->size();
// 	for(int i(0); i < m->size(); i++){
// 		rhs(mRow++) = dot(m->getParameterv(i)) - 1;
// 	}	
}

void constraintCalculator::constraintEquation(double* rhs, std::map<std::string, pointmass*> &masses, std::map<std::string, kinematicConstraint*> &kconsts, std::map<std::string, drivingConstraint*> &dconsts /* = 0 */, int size/* =0 */, double t /* = 0 */, double mul/* =1.0 */)
{
	vector3d v3 = vector3<double>(0.0);
	int ib = 0;
	int jb = 0;
// 	for(std::map<std::string, kinematicConstraint*>::iterator kit = kconsts.begin(); kit != kconsts.end(); kit++){
// 		kinematicConstraint* kin = kit->second;
// 		ib = kin->i
// 	}
// 	KCONSTRAINT::iterator kit = kconsts->begin();
// 	while(kit != kconsts->end()){
// 		ib = kit->i;
// 		jb = kit->j;
// 		switch(kit->type){
// 		case kinematicConstraint::REVOLUTE:
// 			v3 = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kit->sp_i);
// 			//rhs.insert(kit->sRow, POINTER3(v3), 3); 
// 			rhs[kit->sRow+0] = mul*v3.x;
// 			rhs[kit->sRow+1] = mul*v3.y;
// 			rhs[kit->sRow+2] = mul*v3.z;
// 			v3 = m->toGlobal(jb, kit->h_j);
// 			//rhs(kit->sRow+3) = dot(v3, m->toGlobal(ib, kit->f_i));
// 			rhs[kit->sRow+3] = mul*dot(v3, m->toGlobal(ib, kit->g_i));
// 			rhs[kit->sRow+4] = mul*dot(v3, m->toGlobal(ib, kit->f_i));
// 			//rhs(kit->sRow+4) = dot(v3, m->toGlobal(ib, kit->g_i));
// 
// 			break;
// 		case kinematicConstraint::SPHERICAL:
// 			v3 = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kit->sp_i);
// 			//rhs.insert(kit->sRow, POINTER3(v3), 3);
// 			rhs[kit->sRow+0] = mul*v3.x;
// 			rhs[kit->sRow+1] = mul*v3.y;
// 			rhs[kit->sRow+2] = mul*v3.z;
// 			break;
// 		case kinematicConstraint::CYLINDERICAL:
// 			v3 = m->toGlobal(jb, kit->h_j);
// 			rhs[kit->sRow+0] = mul*dot(v3, m->toGlobal(ib, kit->g_i));
// 			rhs[kit->sRow+1] = mul*dot(v3, m->toGlobal(ib, kit->f_i));
// 
// 			v3 = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib);
// 			rhs[kit->sRow+2] = mul*(dot(v3, m->toGlobal(ib, kit->g_i)) - dot(kit->sp_i, kit->g_i)); 
// 			rhs[kit->sRow+3] = mul*(dot(v3, m->toGlobal(ib, kit->f_i)) - dot(kit->sp_i, kit->f_i));
// 
// 			//v3 = m->toGlobal(ib, kit->g_i);
// 			//double tmp = dot(v3, m->toGlobal(ib, kit->g_i));
// 			break;
// 			// 		case kinematicConstraint::DRIVING_CYLINDERICAL:
// 			// 			v3 = jb->toGlobal(kit->h_j);
// 			// 			rhs(kit->sRow+0) = dot(v3, ib->toGlobal(kit->f_i));
// 			// 			rhs(kit->sRow+1) = dot(v3, ib->toGlobal(kit->g_i));
// 			// 			break;
// 		}
// 		kit++;
// 	}
// 	DCONSTRAINT::iterator dit = dconsts->begin();
// 	while(dit != dconsts->end()){
// 		kinematicConstraint *kconst = dit->getTargetJoint();
// 		ib = kconst->i;
// 		jb = kconst->j;
// 		v3 = m->getPositionv(jb) + m->toGlobal(jb, kconst->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kconst->sp_i);
// 		double dd = dot(v3, m->toGlobal(ib, kconst->h_i));
// 		vector3d ghi = m->toGlobal(ib, kconst->h_i);
// 		rhs[dit->getStartRow()] = mul*(dd + dit->driving(t));
// 		dit++;
// 	}
 	int mRow = (masses.size()-1) * 7;
	for(std::map<std::string, pointmass*>::iterator mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		rhs[mRow++] = mul * (mass->Orientation().dot() - 1);
	}	
}

void constraintCalculator::constraintJacobian2(matrix<double>& Pi, std::map<std::string, pointmass*> *masses, std::map<std::string, kinematicConstraint*> *kconsts, std::map<std::string, drivingConstraint*> *dconsts /* = 0 */)
{
// 	int sRow=0;
// 	int iCol=0;
// 	int jCol=0;
// 	int ib = NULL;
// 	int jb = NULL;
// 	vector3d dij = make_vector3d(0.0);
// 	vector3d v3 = make_vector3d(0.0);
// 	vector4d v4 = make_vector4d(0.0);
// 	vector4d epi = make_vector4d(0.0);
// 	vector4d epj = make_vector4d(0.0);
// 	vector4d depi = make_vector4d(0.0);
// 	vector4d depj = make_vector4d(0.0);
// 	KCONSTRAINT::iterator kit = kconsts->begin();
// 	while(kit != kconsts->end()){
// 		sRow = kit->sRow;
// 		ib = kit->i; iCol = kit->iCol;
// 		jb = kit->j; jCol = kit->jCol;
// 		epi = m->getParameterv(ib);
// 		epj = m->getParameterv(jb);
// 		depi = m->getdParameterv(ib);
// 		depj = m->getdParameterv(jb);
// 		switch(kit->type){
// 		case kinematicConstraint::REVOLUTE:
// 			if(ib)
// 			{
// 				lhs.insert(sRow+0, iCol+3, POINTER(B(depi, -kit->sp_i)), MAT3X4);
// 				v4 = transpose(m->toGlobal(jb, kit->h_j), B(epi, kit->g_i)) + transpose(depj, transpose(B(epj, kit->h_j), B(epi, kit->g_i)));
// 				lhs.insert(sRow+3, iCol+3, POINTER4(v4), 4);
// 				v4 = transpose(m->toGlobal(jb, kit->h_j), B(depi, kit->f_i)) + transpose(depj, transpose(B(epj, kit->h_j), B(epi, kit->f_i)));
// 				lhs.insert(sRow+4, iCol+3, POINTER4(v4), 4);
// 			}
// 			if(jb)
// 			{
// 				lhs.insert(sRow+0, jCol+3, POINTER(B(depj, kit->sp_j)), MAT3X4);
// 				v4 = transpose(m->toGlobal(ib, kit->g_i), B(depj, kit->h_j)) + transpose(depi, transpose(B(epi, kit->g_i), B(epj, kit->h_j)));
// 				lhs.insert(sRow+3, jCol+3, POINTER4(v4), 4);
// 				v4 = transpose(m->toGlobal(ib, kit->f_i), B(depj, kit->h_j)) + transpose(depi, transpose(B(epi, kit->f_i), B(epj, kit->h_j)));
// 				lhs.insert(sRow+4, jCol+3, POINTER4(v4), 4);
// 			}
// 			break;
// 		case kinematicConstraint::SPHERICAL:
// 			if(ib)
// 				lhs.insert(sRow+0, iCol+3, POINTER(B(depi, -kit->sp_i)), MAT3X4);
// 			if(jb)
// 				lhs.insert(sRow+0, jCol+3, POINTER(B(depj, kit->sp_j)), MAT3X4);
// 			break;
// 		case kinematicConstraint::CYLINDERICAL:
// 			dij = m->getPositionv(jb) + m->toGlobal(jb, kit->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kit->sp_i);
// 			if(ib)
// 			{
// 				//ep = ib->parameter;
// 				v4 = transpose(m->toGlobal(jb, kit->h_j), B(depi, kit->f_i)) + transpose(depj, transpose(B(epj, kit->h_j), B(epi, kit->f_i)));
// 				lhs.insert(sRow+3, iCol+3, POINTER4(v4), 4);
// 				v4 = transpose(m->toGlobal(jb, kit->h_j), B(depi, kit->g_i)) + transpose(depj, transpose(B(epj, kit->h_j), B(epi, kit->g_i)));
// 				lhs.insert(sRow+4, iCol+3, POINTER4(v4), 4);
// 
// 				v3 = transpose2(-depi, B(epi, kit->f_i));
// 				v4 = transpose(B(epj, kit->sp_j)*depj + m->getVelocityv(jb) - m->getVelocityv(ib), B(epi, kit->f_i)) + transpose(dij + kit->sp_i, B(depi, kit->f_i));
// 				lhs.insert(sRow+2, iCol+0, POINTER3(v3), POINTER4(v4), 34);
// 				v3 = transpose2(-depi, B(epi, kit->g_i));
// 				v4 = transpose(B(epj, kit->sp_j)*depj + m->getVelocityv(jb) - m->getVelocityv(ib), B(epi, kit->g_i)) + transpose(dij + kit->sp_i, B(depi, kit->g_i));
// 				lhs.insert(sRow+3, iCol+0, POINTER3(v3), POINTER4(v4), 34);
// 			}
// 			if(jb)
// 			{
// 				//ep = jb->parameter;
// 				vector3d gfi = m->toGlobal(ib, kit->f_i);
// 				vector3d ggi = m->toGlobal(ib, kit->g_i);
// 				v4 = transpose(gfi, B(depj, kit->h_j)) + transpose(depi, transpose(B(epi, kit->f_i), B(epj, kit->h_j)));
// 				lhs.insert(sRow+3, jCol+3, POINTER4(v4), 4);
// 				v4 = transpose(ggi, B(depj, kit->h_j)) + transpose(depi, transpose(B(epi, kit->g_i), B(epj, kit->h_j)));
// 				lhs.insert(sRow+4, jCol+3, POINTER4(v4), 4);
// 
// 				v3 = transpose2(depi, B(epi, kit->f_i));
// 				v4 = transpose(depi, transpose(B(epi, kit->f_i), B(epj, kit->sp_j))) + transpose(gfi, B(depj, kit->sp_j));
// 				lhs.insert(sRow+2, jCol+0, POINTER3(v3), POINTER4(v4), 34);
// 				v3 = transpose2(depi, B(epi, kit->g_i));
// 				v4 = transpose(depi, transpose(B(epi, kit->g_i), B(epj, kit->sp_j))) + transpose(ggi, B(depj, kit->sp_j));
// 				lhs.insert(sRow+3, jCol+0, POINTER3(v3), POINTER4(v4), 34);
// 			}
// 			break;
// 		case kinematicConstraint::DRIVING_CYLINDERICAL:
// // 			if(ib->ID!=-1)
// // 			{
// // // 				ep = ib->parameter;
// // // 				lhs.insert(sRow+0, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->f_i)).w, 4);
// // // 				lhs.insert(sRow+1, iCol+3, &transpose(jb->toGlobal(kit->h_j), B(ep, kit->g_i)).w, 4);
// // 			}
// // 			if(jb->ID!=-1)
// // 			{
// // // 				ep = jb->parameter;
// // // 				vector3d gfi = ib->toGlobal(kit->f_i);
// // // 				vector3d ggi = ib->toGlobal(kit->g_i);
// // // 				lhs.insert(sRow+0, jCol+3, &transpose(gfi, B(ep, kit->h_j)).w, 4);
// // // 				lhs.insert(sRow+1, jCol+3, &transpose(ggi, B(ep, kit->h_j)).w, 4);
// // 			}
// 			break;
// 		}
// 		kit++;
// 	}
// 	DCONSTRAINT::iterator dit = dconsts->begin();
// 	while(dit != dconsts->end()){
// 		kinematicConstraint *kconst = dit->getTargetJoint();
// 		ib = kconst->i;
// 		jb = kconst->j;
// 		epi = m->getParameterv(ib);
// 		epj = m->getParameterv(jb);
// 		depi = m->getdParameterv(ib);
// 		depj = m->getdParameterv(jb);
// 		dij = m->getPositionv(jb) + m->toGlobal(jb, kconst->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kconst->sp_i);
// 
// 		sRow = dit->getStartRow();
// 		if(ib!=-1)
// 		{
// 			v3 = transpose2(-depi, B(epi, kconst->h_i));
// 			v4 = transpose(B(epj, kconst->sp_j)*depj + m->getVelocityv(jb) - m->getVelocityv(ib), B(epi, kconst->h_i)) + transpose(dij + kconst->sp_i, B(depi, kconst->h_i));
// 			lhs.insert(sRow, kconst->iCol+0, POINTER3(v3), POINTER4(v4), 34);
// 		}
// 		if(jb!=-1)
// 		{
// 			v3 = transpose2(depi, B(epi, kconst->h_i));
// 			v4 = transpose(depi, transpose(B(epi, kconst->h_i), B(epj, kconst->sp_j))) + transpose(m->toGlobal(ib, kconst->h_i), B(depj, kconst->sp_j));
// 			lhs.insert(sRow, kconst->jCol+0, POINTER3(v3), POINTER4(v4), 34);
// 		}
// 
// 		switch (dit->dType)
// 		{
// 		case drivingConstraint::TRANSLATIONAL: sRow+=1; break;
// 		default:
// 			break;
// 		}
// 		dit++;
// 	}
// 	sRow = lhs.rows() - m->size() + 1;
// 
// 	for(int i(0); i < m->size()-1; i++){
// 		lhs.insert(sRow++, i*7 + 3, &(2*m->getdParameterv(i+1)).w, 4);
// 	}
// // 	lhs.exportDataToText();
}

void constraintCalculator::sparseConstraintJacobian(int sr, sparse_matrix<double>& sjc, std::map<std::string, pointmass*> &masses, std::map<std::string, kinematicConstraint*> &kconsts, std::map<std::string, drivingConstraint*> &dconsts /* = 0 */)
{
	int sRow = 0;
	int iCol = 0;
	int jCol = 0;
	int ib=0; 
	int jb=0;
	int i = 0;
 	vector3<double> dij;
 	vector4<double> ep;
	std::map<std::string, kinematicConstraint*>::iterator kit;// = kconsts.begin();
 	sjc.zeroCount();
 	sRow = sjc.rows() - masses.size() + 1 + sr;
	std::map<std::string, pointmass*>::iterator mit;
	for(mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		euler_parameter<double> ep2 = 2 * mass->Orientation();
		sjc.extraction(sRow++, i * 7 + 3, ep2.Pointer(), 4);
		i++;
	}
	for(kit = kconsts.begin(); kit != kconsts.end(); kit++){
		kinematicConstraint* kconst = kit->second;
		sRow = kconst->sRow + sr;
		ib = kconst->i; iCol = kconst->iCol;
		jb = kconst->j; jCol = kconst->jCol;
//		switch(kconst->type)
//		{
// 		case kinematicConstraint::REVOLUTE:
// 			if(ib)
// 			{
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,iCol+i) = -1;
// 				ep = m->getParameterv(ib);
// 				sjc.extraction(sRow+0, iCol+3, POINTER(B(ep, -kconst->sp_i)), MAT3X4);
// 				sjc.extraction(sRow+3, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->g_i))), VEC4);
// 				sjc.extraction(sRow+4, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->f_i))), VEC4);
// 			}
// 			if(jb)
// 			{
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,jCol+i) = 1;
// 				ep = m->getParameterv(jb);
// 				sjc.extraction(sRow+0, jCol+3, POINTER(B(ep, kc->sp_j)), MAT3X4);	
// 				sjc.extraction(sRow+3, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->g_i), B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+4, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->f_i), B(ep, kc->h_j))), VEC4);
// 			}
// 			//std::cout << *sjc << std::endl;
// 			break;
// 		case kinematicConstraint::SPHERICAL:
// 			if(ib)
// 			{
// 				ep = m->getParameterv(ib);
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,iCol+i) = -1;
// 				sjc.extraction(sRow+0, iCol+3, POINTER(B(ep, -kc->sp_i)), MAT3X4);		
// 			}
// 			if(jb)
// 			{
// 				ep = m->getParameterv(jb);
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,jCol+i) = 1;
// 				sjc.extraction(sRow+0, jCol+3, POINTER(B(ep, kc->sp_j)), MAT3X4);
// 			}
// 			break;
// 		case kinematicConstraint::TRANSLATION:
// 			dij = m->getPositionv(jb) + m->toGlobal(jb, kc->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kc->sp_i);
// 			if(ib)
// 			{
// 				ep = m->getParameterv(ib);
// 				sjc.extraction(sRow+0, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->f_i))), VEC4);
// 				sjc.extraction(sRow+1, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->g_i))), VEC4);
// 				sjc.extraction(sRow+2, iCol+0, POINTER3((-kc->f_i)), POINTER4(transpose(dij + kc->sp_i, B(ep, kc->f_i))), VEC3_4);
// 				sjc.extraction(sRow+3, iCol+0, POINTER3((-kc->g_i)), POINTER4(transpose(dij + kc->sp_i, B(ep, kc->g_i))), VEC3_4);
// 				sjc.extraction(sRow+4, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->g_j), B(ep, kc->g_i))), VEC4);
// 			}
// 			if(jb)
// 			{
// 				ep = m->getParameterv(jb);
// 				sjc.extraction(sRow+0, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->f_i), B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+1, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->g_i), B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+2, jCol+0, POINTER3(kc->f_i), POINTER4(transpose(kc->f_i, B(ep, kc->sp_j))), VEC3_4);
// 				sjc.extraction(sRow+3, jCol+0, POINTER3(kc->g_i), POINTER4(transpose(kc->g_i, B(ep, kc->sp_j))), VEC3_4);
// 				sjc.extraction(sRow+4, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->g_i), B(ep, kc->g_j))), VEC4);
// 			}
// 			//std::cout << *sjc << std::endl;
// 			break;
// 		case kinematicConstraint::CYLINDERICAL:
// 			//vector3d yy = jb->Getpos() + jb->A()*joint->sp_j;
// 			dij = m->getPositionv(jb) + m->toGlobal(jb, kc->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kc->sp_i);
// 			if(ib)
// 			{
// 				ep = m->getParameterv(ib);
// 				sjc.extraction(sRow+0, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->g_i))), VEC4);
// 				sjc.extraction(sRow+1, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->f_i))), VEC4);
// 				sjc.extraction(sRow+2, iCol+0, POINTER3((-m->toGlobal(ib, kc->g_i))), POINTER4(transpose(dij + m->toGlobal(ib, kc->sp_i), B(ep, kc->g_i))), VEC3_4);
// 				sjc.extraction(sRow+3, iCol+0, POINTER3((-m->toGlobal(ib, kc->f_i))), POINTER4(transpose(dij + m->toGlobal(ib, kc->sp_i), B(ep, kc->f_i))), VEC3_4);
// 
// 			}
// 			if(jb)
// 			{
// 				ep = m->getParameterv(jb);
// 				vector3d gfi = m->toGlobal(ib, kc->f_i);
// 				vector3d ggi = m->toGlobal(ib, kc->g_i);
// 				sjc.extraction(sRow+0, jCol+3, POINTER4(transpose(ggi, B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+1, jCol+3, POINTER4(transpose(gfi, B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+2, jCol+0, POINTER3(ggi), POINTER4(transpose(ggi, B(ep, kc->sp_j))), VEC3_4);
// 				sjc.extraction(sRow+3, jCol+0, POINTER3(gfi), POINTER4(transpose(gfi, B(ep, kc->sp_j))), VEC3_4);
// 
// 			}
// 			break;
// 		}
//	}
// 
// 	while(kc != kconsts->end()){
// 		sRow = kc->sRow + sr;
// 		ib = kc->i; iCol = kc->iCol;
// 		jb = kc->j; jCol = kc->jCol;
// 		switch(kc->type)
// 		{
// 		case kinematicConstraint::REVOLUTE:
// 			if(ib)
// 			{
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,iCol+i) = -1;
// 				ep = m->getParameterv(ib);
// 				sjc.extraction(sRow+0, iCol+3, POINTER(B(ep, -kc->sp_i)), MAT3X4);
// 				sjc.extraction(sRow+3, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->g_i))), VEC4);
// 				sjc.extraction(sRow+4, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->f_i))), VEC4);
// 			}
// 			if(jb)
// 			{
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,jCol+i) = 1;
// 				ep = m->getParameterv(jb);
// 				sjc.extraction(sRow+0, jCol+3, POINTER(B(ep, kc->sp_j)), MAT3X4);	
// 				sjc.extraction(sRow+3, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->g_i), B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+4, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->f_i), B(ep, kc->h_j))), VEC4);
// 			}
// 			//std::cout << *sjc << std::endl;
// 			break;
// 		case kinematicConstraint::SPHERICAL:
// 			if(ib)
// 			{
// 				ep = m->getParameterv(ib);
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,iCol+i) = -1;
// 				sjc.extraction(sRow+0, iCol+3, POINTER(B(ep, -kc->sp_i)), MAT3X4);		
// 			}
// 			if(jb)
// 			{
// 				ep = m->getParameterv(jb);
// 				for(unsigned i(0); i < 3; i++) sjc(sRow+i,jCol+i) = 1;
// 				sjc.extraction(sRow+0, jCol+3, POINTER(B(ep, kc->sp_j)), MAT3X4);
// 			}
// 			break;
// 		case kinematicConstraint::TRANSLATION:
// 			dij = m->getPositionv(jb) + m->toGlobal(jb, kc->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kc->sp_i);
// 			if(ib)
// 			{
// 				ep = m->getParameterv(ib);
// 				sjc.extraction(sRow+0, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->f_i))), VEC4);
// 				sjc.extraction(sRow+1, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->g_i))), VEC4);
// 				sjc.extraction(sRow+2, iCol+0, POINTER3((-kc->f_i)), POINTER4(transpose(dij + kc->sp_i, B(ep, kc->f_i))), VEC3_4);
// 				sjc.extraction(sRow+3, iCol+0, POINTER3((-kc->g_i)), POINTER4(transpose(dij + kc->sp_i, B(ep, kc->g_i))), VEC3_4);
// 				sjc.extraction(sRow+4, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->g_j), B(ep, kc->g_i))), VEC4);
// 			}
// 			if(jb)
// 			{
// 				ep = m->getParameterv(jb);
// 				sjc.extraction(sRow+0, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->f_i), B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+1, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->g_i), B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+2, jCol+0, POINTER3(kc->f_i), POINTER4(transpose(kc->f_i, B(ep, kc->sp_j))), VEC3_4);
// 				sjc.extraction(sRow+3, jCol+0, POINTER3(kc->g_i), POINTER4(transpose(kc->g_i, B(ep, kc->sp_j))), VEC3_4);
// 				sjc.extraction(sRow+4, jCol+3, POINTER4(transpose(m->toGlobal(ib, kc->g_i), B(ep, kc->g_j))), VEC4);
// 			}
// 			//std::cout << *sjc << std::endl;
// 			break;
// 		case kinematicConstraint::CYLINDERICAL:
// 			//vector3d yy = jb->Getpos() + jb->A()*joint->sp_j;
// 			dij = m->getPositionv(jb) + m->toGlobal(jb, kc->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kc->sp_i);
// 			if(ib)
// 			{
// 				ep = m->getParameterv(ib);
// 				sjc.extraction(sRow+0, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->g_i))), VEC4);
// 				sjc.extraction(sRow+1, iCol+3, POINTER4(transpose(m->toGlobal(jb, kc->h_j), B(ep, kc->f_i))), VEC4);
// 				sjc.extraction(sRow+2, iCol+0, POINTER3((-m->toGlobal(ib, kc->g_i))), POINTER4(transpose(dij + m->toGlobal(ib, kc->sp_i), B(ep, kc->g_i))), VEC3_4);
// 				sjc.extraction(sRow+3, iCol+0, POINTER3((-m->toGlobal(ib, kc->f_i))), POINTER4(transpose(dij + m->toGlobal(ib, kc->sp_i), B(ep, kc->f_i))), VEC3_4);
// 				
// 			}
// 			if(jb)
// 			{
// 				ep = m->getParameterv(jb);
// 				vector3d gfi = m->toGlobal(ib, kc->f_i);
// 				vector3d ggi = m->toGlobal(ib, kc->g_i);
// 				sjc.extraction(sRow+0, jCol+3, POINTER4(transpose(ggi, B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+1, jCol+3, POINTER4(transpose(gfi, B(ep, kc->h_j))), VEC4);
// 				sjc.extraction(sRow+2, jCol+0, POINTER3(ggi), POINTER4(transpose(ggi, B(ep, kc->sp_j))), VEC3_4);
// 				sjc.extraction(sRow+3, jCol+0, POINTER3(gfi), POINTER4(transpose(gfi, B(ep, kc->sp_j))), VEC3_4);
// 				
// 			}
// 			break;
// 		}
// 		kc++;
}

	
	//std::cout << sjc;
// 	for(std::vector<drivingType>::iterator driving = drivings.begin(); driving != drivings.end(); driving++)
// 	{
// 		sCol = (driving->targetBody - 1) * 7;
// 		switch(driving->targetCoord)
// 		{
// 		case 'z':
// 			sjc->extraction(sRow, sCol+3, POINTER(vector4d(0.0,0.0,0.0,1.0)), VEC4);
// 			sRow+=1;
// 			break;
// 		}
// 
// 	}
}

void constraintCalculator::calculateSystemJacobian(matrix<double>& lhs, double* lag, std::map<std::string, pointmass*> &masses, std::map<std::string, kinematicConstraint*> &kconsts, double mul)
{
// 	int sRow = 0;
// 	int iCol = 0;
// 	int jCol = 0;
// 	int ib=0; 
// 	int jb=0;
// 	matrix<double> m_lhs(lhs.rows(), lhs.cols()); m_lhs.zeros();
//  	matrix4x4<double> Dv;
//  	matrix3x4<double> Bv;
//  	KCONSTRAINT::iterator kconst = kconsts->begin();
// 	while(kconst != kconsts->end()){
// 		ib = kconst->i; iCol = kconst->iCol;
// 		jb = kconst->j; jCol = kconst->jCol;
// 		switch(kconst->type){
// 		case kinematicConstraint::SPHERICAL:
// 			if(ib){
// 				Dv = -D(kconst->sp_i, make_vector3d(lag+sRow));
// 				m_lhs.plus(iCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 
// 			}
// 			if(jb){
// 				Dv = D(kconst->sp_j, make_vector3d(lag+sRow));
// 				m_lhs.plus(jCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 			}
// 			sRow += 3;
// 			break;
// 		case kinematicConstraint::REVOLUTE:
// 			if(ib){
// 				Dv = -D(kconst->sp_i, make_vector3d(lag+sRow));
// 				m_lhs.plus(iCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 
// 				Dv = lag[sRow+3]*D(kconst->g_i, m->toGlobal(jb, kconst->h_j)) + lag[sRow+4]*D(kconst->f_i, m->toGlobal(jb, kconst->h_j));
// 				m_lhs.plus(iCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 
// 				if(jb){
// 					Dv = lag[sRow+3]*transpose(B(m->getParameterv(jb), kconst->h_j), B(m->getParameterv(ib), kconst->g_i)) + lag[sRow+4]*transpose(B(m->getParameterv(jb), kconst->h_j), B(m->getParameterv(ib), kconst->f_i));
// 					m_lhs.plus(jCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 				}		
// 			}
// 			if(jb){
// 				Dv = D(kconst->sp_j, make_vector3d(lag+sRow));
// 				m_lhs.plus(jCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 
// 				if(ib){
// 					Dv = lag[sRow+3]*transpose(B(m->getParameterv(ib), kconst->g_i), B(m->getParameterv(jb), kconst->h_j)) + lag[sRow+4]*transpose(B(m->getParameterv(ib), kconst->f_i), B(m->getParameterv(jb), kconst->h_j));
// 					m_lhs.plus(iCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 				}
// 
// 				Dv = lag[sRow+3]*D(kconst->h_j, m->toGlobal(ib, kconst->g_i)) + lag[sRow+4]*D(kconst->h_j, m->toGlobal(ib, kconst->f_i));
// 				m_lhs.plus(jCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 			}
// 			sRow += 5;
// 			break;
// 		case kinematicConstraint::CYLINDERICAL:
// 			vector3d dij = m->getPositionv(jb) + m->toGlobal(jb, kconst->sp_j) - m->getPositionv(ib) - m->toGlobal(ib, kconst->sp_i);
// 			Bv = -lag[sRow+2]*B(m->getParameterv(ib), kconst->g_i) - lag[sRow+3]*B(m->getParameterv(ib), kconst->f_i);
// 			if(ib){
// 				Dv = lag[sRow+0]*D(kconst->g_i, m->toGlobal(jb, kconst->h_j)) + lag[sRow+1]*D(kconst->f_i, m->toGlobal(jb, kconst->h_j));
// 				m_lhs.plus(iCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 
// 				if(jb){
// 					Dv = lag[sRow+0]*transpose(B(m->getParameterv(jb), kconst->h_j), B(m->getParameterv(ib), kconst->g_i)) + lag[sRow+1]*transpose(B(m->getParameterv(jb), kconst->h_j), B(m->getParameterv(ib), kconst->f_i));
// 					m_lhs.plus(jCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 
// 					Bv = -Bv;
// 					Dv = lag[sRow+2]*transpose(B(m->getParameterv(jb), kconst->sp_j), B(m->getParameterv(ib), kconst->g_i)) + lag[sRow+3]*transpose(B(m->getParameterv(jb), kconst->sp_j), B(m->getParameterv(ib), kconst->f_i));
// 					m_lhs.plus(jCol+0, iCol+3, POINTER(Bv), MAT3X4);
// 					m_lhs.plus(jCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 				}
// 
// 				Bv = -lag[sRow+2]*B(m->getParameterv(ib), kconst->g_i) - lag[sRow+3]*B(m->getParameterv(ib), kconst->f_i);
// 				Dv = lag[sRow+2]*D(kconst->g_i, dij + m->toGlobal(ib, kconst->sp_i)) + lag[sRow+3]*D(kconst->f_i, dij + m->toGlobal(ib, kconst->sp_i));
// 				m_lhs.plus(iCol+0, iCol+3, POINTER(Bv), MAT3X4);
// 				m_lhs.plus(iCol+3, iCol+3, POINTER(Dv), MAT4x4);
// 				m_lhs.plus(iCol+3, iCol+0, POINTER(Bv), MAT4x3);
// 
// 				
// 			}
// 			if(jb){
// 				if(ib){
// 					Dv = lag[sRow+0]*transpose(B(m->getParameterv(ib), kconst->g_i), B(m->getParameterv(jb), kconst->h_j)) + lag[sRow+1]*transpose(B(m->getParameterv(ib), kconst->f_i), B(m->getParameterv(jb), kconst->h_j));
// 					m_lhs.plus(iCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 
// 					Bv = -Bv;
// 					Dv = lag[sRow+2]*transpose(B(m->getParameterv(ib), kconst->g_i), B(m->getParameterv(jb), kconst->sp_j)) + lag[sRow+3]*transpose(B(m->getParameterv(ib), kconst->f_i), B(m->getParameterv(ib), kconst->sp_j));
// 					m_lhs.plus(iCol+3, jCol+0, POINTER(Bv), MAT4x3);
// 					m_lhs.plus(iCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 				}
// 				Dv = lag[sRow+0]*D(kconst->h_j, m->toGlobal(ib, kconst->g_i)) + lag[sRow+1]*D(kconst->h_j, m->toGlobal(ib, kconst->f_i));
// 				m_lhs.plus(jCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 
// 				Dv = lag[sRow+2]*D(kconst->sp_j, m->toGlobal(ib, kconst->g_i)) + lag[sRow+3]*D(kconst->sp_j, m->toGlobal(ib, kconst->f_i));
// 				m_lhs.plus(jCol+3, jCol+3, POINTER(Dv), MAT4x4);
// 			}
// 			sRow += 4;
// 		}
// 		kconst++;
// 	}
// 
// 	lhs += mul*m_lhs;
}