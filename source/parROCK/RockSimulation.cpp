#include "RockSimulation.h"
#include "../timer.h"
#include <ctime>
#include <cmath>
#include <list>
#include <vector>
#include "contact.h"
#include "../algebra.h"

RockSimulation::RockSimulation(std::string bpath, std::string cname)
	: Simulation(bpath, cname)
	, isotropic_arrangement(false)
	, exist_bonds(false)
{

}

RockSimulation::~RockSimulation()
{

}

bool RockSimulation::Initialize()
{
	if (!MakeBalls()){
		std::cout << "Error : MakeBalls() function is returned the false value." << std::endl;
		return false;
	}

	if (!dt){
		dt = 1;
		for (unsigned int i = 0; i < nballs; i++){
			ball *b = balls + i;
			double temp = sqrt(b->Mass() / b->Kn());
			if (dt > temp)
				dt = temp;
		}
	}
	//for (int i = 0; i < 100; i++){
	//	std::cout << frand(10)-5 << std::endl;
	//}
	//dt = dt * 0.1;
	if (!sort->initialize())
	{
		std::cout << "ERROR : sort->initialize() function is returned the fals value." << std::endl;
		return false;
	}

	// rearrangement procedure
	IsotropicPacking();

	// install isotropic stress
	//InstallIsotropicStress();
	if (specificData != ""){
		std::fstream pf;
		pf.open(specificData, std::ios::in | std::ios::binary);
		if (pf.is_open()){
			while (!pf.eof()){
				int type;
				pf.read((char*)&type, sizeof(int));
				switch (type)
				{
				case -1:
					break;
				case PARTICLE:
					setSpecificDataFromFile(pf);
					break;
				default:
					break;
				}
			}
		}
	}
	FloaterElimination(true);

	ParallelBondProperties();

	Clustering();

	return true;
}

void RockSimulation::setSpecificDataFromFile(std::fstream& pf)
{
	float *tpos = new float[ball::nballs * 4];
	float *tvel = new float[ball::nballs * 4];

	pf.read((char*)tpos, sizeof(float)*ball::nballs * 4);
	pf.read((char*)tvel, sizeof(float)*ball::nballs * 4);

	for (unsigned int i = 0; i < ball::nballs; i++){
		balls[i].Position() = vector3<double>(
			static_cast<double>(tpos[i * 4 + 0]),
			static_cast<double>(tpos[i * 4 + 1]),
			static_cast<double>(tpos[i * 4 + 2]));
		balls[i].Radius() = static_cast<double>(tpos[i * 4 + 3]);
		balls[i].Velocity() = vector3<double>(
			static_cast<double>(tvel[i * 4 + 0]),
			static_cast<double>(tvel[i * 4 + 1]),
			static_cast<double>(tvel[i * 4 + 2]));
	}

	delete[] tpos;
	delete[] tvel;
}

void RockSimulation::ParallelBondProperties()
{
	//for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++){
	//	if ((*it)->Wall()){
	//		clist.erase(it);
	//	}
	//}
	for (unsigned int i = 0; i < nballs; i++){
		ball* b = balls + i;
		b->Velocity() = 0.0;
		b->Omega() = 0.0;
		b->Acceleration() = 0.0;
		b->Force() = 0.0;
		b->Moment() = 0.0;
		//b->ContactWMap().clear();
	}
	for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++){
		if ((*it)->Wall()){
			continue;
		}
		(*it)->PBondProps().NormalStrength() = CementElement::maxTensileStress + CementElement::tensileStdDeviation * GaussianNormalDistribution(frand(10) - 5);
		(*it)->PBondProps().ShearStrength() = CementElement::maxShearStress + CementElement::shearStdDeviation * GaussianNormalDistribution(frand(10) - 5);
		(*it)->PBondProps().Radius() = CementElement::brmul * min((*it)->IBall()->Radius(), (*it)->JBall()->Radius());
		double radsum = (*it)->IBall()->Radius() + (*it)->JBall()->Radius();
		(*it)->PBondProps().Kn() = CementElement::cyoungsModulus / radsum;
		(*it)->PBondProps().Ks() = (*it)->PBondProps().Kn() / CementElement::cstiffnessRatio;
		(*it)->SetNormalForce(vector3<double>(0.0));
		(*it)->SetNormalMoment(vector3<double>(0.0));
		(*it)->SetShearForce(vector3<double>(0.0));
		(*it)->SetShearMoment(vector3<double>(0.0));
	}
	exist_bonds = true;
}

ccontact* RockSimulation::b_clist(ball* b)
{
	for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++){
		if ((*it)->IBall() == b){
			return (*it);
		}
		if ((*it)->JBall() == b){
			return (*it);
		}
	}
	return NULL;
}

bool RockSimulation::b_extra(ball* bp1, ball* bpother)
{
	if (bp1 == bpother){
		return true;
	}
	std::map<ball*, Cluster>::iterator it = clusters.find(bp1);
	if (clusters.size() == 0 || it == clusters.end()){
		Cluster cluster;
		cluster.addBall(bpother);
		clusters[bp1] = cluster;
		return true;
	}

	if (it != clusters.end()){
		it->second.addBall(bpother);
	}
	if (it->second.Nballs() == 4)
		return false;
	return true;
}

void RockSimulation::Clustering()
{
	//ccontact *cps[4] = { 0, };
	ball* bpother = NULL;
	ball* tempball = NULL;
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball* bp1 = &balls[i];
		tempball = bp1;
		std::map<ball*, ccontact>::iterator it = bp1->ContactPMap().begin();
		ccontact* cp = &(it->second);
		//cps[0] = cp;
		while (1){
			if (cp->JBall() != NULL){
				//if (cp->IBall() == bp1){
				bpother = cp->JBall();
				//}
				//else{
				//	bpother = cp->IBall();
				//}
				if (!b_extra(bp1, bpother)){
					break;
				}
			}
			it++;
			if (it == tempball->ContactPMap().end()){
				it = bpother->ContactPMap().begin();
				tempball = bpother;
			}
			cp = &(it->second);
			/*if (cp->IBall() == bp1){
			cp = cp->c_b1clist();
			}
			else{
			cp = cp->c_b2clist();
			}*/
		}
	}
}

void RockSimulation::FloaterElimination(bool isInitRun)
{
	utility::writer::SetFileSystem("FloaterElimination");
	double sum = 0.0;
	unsigned int count = 0;
	unsigned int old_count = 0;
	double minRadius = 0.5 * (RockElement::maxDiameter / RockElement::diameterRatio);
	double maxRadius = 0.5 * RockElement::maxDiameter;
	ccontact::prod = 1.0;
	if (isInitRun)
		RunCycle(2000, 10);
	sort->sort();
	sort->detect();
	if (isotropic_arrangement){

		while (1){
			bool isDone = true;
			unsigned int nfloat = 0;
			double sum = 0;
			for (unsigned int i = 0; i < ball::nballs; i++){
				ball *b = balls + i;
				if ((b->ContactPMap().size() + b->ContactWMap().size()) < RSP::flt_def){
					isDone = false;
					b->Floater() = true;
					nfloat++;
				}
				else{
					b->Floater() = false;
				}
			}
			std::cout << "nfloat : " << nfloat << std::endl;
			if (!isDone){
				for (unsigned int i = 0; i < ball::nballs; i++){
					ball *b = balls + i;
					b->Radius() *= RSP::flt_r_mult;
				}
			}
			else{
				break;
			}
			RunCycle(100, 10);
			sort->sort();
			sort->detect();
		}
		return;
	}
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball *b = balls + i;
		b->Floater() = false;
		b->VelocityFixFlag() = true;
		b->OmegaFixFlag() = true;
		b->Color() = BLUE;
		b->Velocity() = 0.0;
		b->Omega() = 0.0;
	}
	for (unsigned int i = 0; i < ball::nballs; i++){
		ball *b = balls + i;
		bool isFloater = false;
		double mcf = 0.0;
		unsigned int n_cf = 0;
		unsigned int ncut = RSP::flt_def;
		for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++){
			mcf += abs((*it)->NormalForce().length());
			n_cf++;
		}
		if (n_cf > 0)
			mcf /= double(n_cf);
		double av_contact_force = mcf;
		double target = RSP::f_tol * mcf;
		if ((b->ContactPMap().size() + b->ContactWMap().size()) < RSP::flt_def){
			b->Floater() = true;
			b->VelocityFixFlag() = false;
			b->OmegaFixFlag() = false;
			b->Color() = RED;
			b->Friction() = 0.0;
			b->Radius() = b->Radius() * RSP::flt_r_mult;
			RunCycle(200, 10);

			double max_act_force = 0.0;
			unsigned int n_active = 0;
			while (1){
				sort->sort();
				sort->detect();
				sum = b->OnlyNormalForceBySumation();
				count = b->ContactPMap().size() + b->ContactWMap().size();
				if (!count){
					b->Radius() = b->Radius() * 1.1;
					RunCycle(100, 50);
					continue;
				}
				sum = sum / double(count);
				if (sum < target){
					if (count >= RSP::flt_def){
						b->Floater() = false;
						b->VelocityFixFlag() = true;
						b->Velocity() = 0.0;
						b->OmegaFixFlag() = true;
						b->Omega() = 0.0;
						break;
					}
					else{
						b->Radius() = b->Radius() * 1.1;
						RunCycle(100, 50);
					}
				}
				else{
					double delta_r = RSP::relax * (sum - RSP::hyst * target) / b->Kn();
					b->Radius() = b->Radius() - delta_r;
					RunCycle(100, 50);
				}
				max_act_force = max(max_act_force, sum);
			}
		}
	}
	//for (unsigned int outer = 0; outer < 1000; outer++){
	//	unsigned int nfloater = 0;
	//	ccontact::prod = 1.0;
	//	sort->sort();
	//	sort->detect();
	//	for (unsigned int i = 0; i < nballs; i++){
	//		ball* b = &balls[i];
	//		if ((b->ContactPMap().size() + b->ContactWMap().size()) < RSP::flt_def){
	//			nfloater++;
	//		}
	//	}
	//	std::cout << "The number of floater : " << nfloater << std::endl;
	//	if (nfloater <= 0){
	//		for (unsigned int i = 0; i < nballs; i++){
	//			ball* b = &balls[i];
	//			b->Floater() = false;
	//		}
	//		break;
	//	}

	//	double mcf = 0.0;
	//	unsigned int n_cf = 0;
	//	unsigned int ncut = RSP::flt_def;
	//	for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++){
	//		mcf += abs((*it)->NormalForce().length());
	//		n_cf++;
	//	}
	//	if (n_cf > 0)
	//		mcf /= double(n_cf);
	//	double av_contact_force = mcf;
	//	double target = RSP::f_tol * mcf;
	//	std::cout << "target : " << target << std::endl;
	//	unsigned int n_float = 0;
	//	for (unsigned int i = 0; i < nballs; i++){
	//		ball* b = &balls[i];
	//		b->Floater() = false;
	//		b->VelocityFixFlag() = true;
	//		b->OmegaFixFlag() = true;
	//		b->Color() = BLUE;
	//		b->Velocity() = 0.0;
	//		b->Omega() = 0.0;
	//		count = b->ContactPMap().size() + b->ContactWMap().size();
	//		if (count < ncut){
	//			b->VelocityFixFlag() = true;
	//			b->OmegaFixFlag() = true;
	//			b->Floater() = true;
	//			//b->Color() = RED;
	//			b->Friction() = 0.0;
	//			b->Radius() = b->Radius() * RSP::flt_r_mult;
	//			/*if (b->Radius() > maxRadius){
	//				b->Radius() = maxRadius;
	//			}*/
	//			n_float++;
	//		}
	//		else{
	//			//b->VelocityFixFlag() = true;
	//			//b->OmegaFixFlag() = true;
	//		}
	//	}
	//	if (outer > 0){
	//		if (n_float >= old_count){
	//			ncut = max(2, ncut - 1);
	//			RSP::flt_def = ncut;
	//		}
	//	}
	//	//ccontact::prod = 1.5;
	//	//RunCycle(200, 1);
	//	//ccontact::prod = 1.0;
	//	for (unsigned int i = 0; i < nballs; i++){
	//		ball* b = &balls[i];
	//		if (!b->Floater())
	//			continue;
	//		for (unsigned int n = 0; n < 100; n++){
	//			sort->sort();
	//			sort->detect();
	//			double max_act_force = 0.0;
	//			unsigned int n_active = 0;
	//			//for (ball* b = ball::BeginBall(); b != NULL; b = b->NextBall()){
	//			count = 0;
	//			sum = 0.0;
	//			if (b->Floater()){
	//				b->Velocity() = 0.0;
	//				b->Omega() = 0.0;
	//				b->VelocityFixFlag() = false;
	//				b->OmegaFixFlag() = false;
	//				b->Color() = RED;
	//				//RunCycle(1000, 2);
	//				sum = b->OnlyNormalForceBySumation();
	//				count = b->ContactPMap().size() + b->ContactWMap().size();
	//			}
	//			if (count > 1){
	//				sum = sum / double(count);
	//				double delta_r;
	//				if (sum > target){
	//					delta_r = RSP::relax * (sum - RSP::hyst * target) / b->Kn();
	//					b->Radius() = b->Radius() - delta_r;
	//					/*if (b->Radius() < minRadius){
	//					b->Radius() = minRadius;
	//					}*/

	//					//b->InertiaMass() = b->Mass() = RockElement::density * (dim == DIM_2 ? b->Radius() * b->Radius() * M_PI : (4 / 3) * M_PI * pow(b->Radius(), 3));
	//					//b->Inertia() = 2.0 * b->Mass() * pow(b->Radius(), 2) / 5.0;
	//					std::cout << "target : " << target << ", sum : " << sum << ", delta_r : " << delta_r << std::endl;
	//					n_active++;
	//					//break;
	//				}

	//			}
	//			max_act_force = max(max_act_force, sum);
	//			//}
	//			std::cout << "floaters : " << n_float << ", active : " << n_active << std::endl;
	//			if (n_active == 0){
	//				old_count = n_float;
	//				b->VelocityFixFlag() = true;
	//				b->OmegaFixFlag() = true;
	//				break;
	//			}
	//			RunCycle(100, 100);
	//		}
	//	}
	//}
	//RunCycle(200, 1);
	utility::writer::CloseFileSystem();
}

void RockSimulation::IsotropicPacking()
{
	double sumKn = 0;
	double sumKs = 0;
	for (unsigned int i = 0; i < nballs; i++){
		ball* b = &balls[i];
		sumKn += b->Kn();
		sumKs += b->Ks();
	}
	double avgKn = sumKn / ball::nballs;
	double avgKs = sumKs / ball::nballs;
	std::map<std::string, Geometry*>::iterator it = geometries.begin();
	for (; it != geometries.end(); it++){
		if (it->second->Type() == GEO_BOUNDARY){
			it->second->Kn() = WallElement::wYoungsFactor * avgKn;
			it->second->Ks() = 0.0;// WallElement::wYoungsFactor * avgKs;
			it->second->Friction() = 0.0;
			if (it->second->Shape() == CUBE){
				geo::Cube *cube = dynamic_cast<geo::Cube*>(it->second);
				cube->SetKnEachPlane();
			}
		}
		else{
			if (it->second->Shape() == SHAPE){
				geo::Shape *shape = dynamic_cast<geo::Shape*>(it->second);
				if (isotropic_arrangement){
					double ewpy = (shape->Youngs() * shape->Youngs()) / (shape->Youngs()*(1 - shape->Poisson()*shape->Poisson()) + shape->Youngs()*(1 - shape->Poisson()*shape->Poisson()));
					it->second->Kn() = (4 / 3)*sqrt(balls[0].Radius())*ewpy;
					it->second->Ks() = 0.0;
				}
			}
		}
	}
	std::map<std::string, Object*>::iterator ito = objects.begin();
	for (; ito != objects.end(); ito++){
		ito->second->Kn() = WallElement::wYoungsFactor * avgKn;
		ito->second->Ks() = 0.0;
		ito->second->Friction() = 0.0;
	}
	/*utility::writer::SetFileSystem("IsotropicPacking");

	it = geometries.find("specimen");
	if (it->second->Shape() == RECTANGLE){
	geo::Rectangle *rec = dynamic_cast<geo::Rectangle*>(it->second);
	for (int i = 0; i < 30; i++){
	RunCycle(5);
	for (unsigned int i = 0; i < nballs; i++){
	ball* b = &balls[i];
	bool inBound = true;
	if (b->Position().x < rec->StartPoint().x || b->Position().y < rec->StartPoint().y || b->Position().x > rec->EndPoint().x || b->Position().y > rec->EndPoint().y){
	b->Position().x = rec->StartPoint().x + frand() * rec->Sizex();
	b->Position().y = rec->StartPoint().y + frand() * rec->Sizey();
	}
	b->Velocity() = 0.0;
	b->Omega() = 0.0;
	}
	}
	}

	RunCycle(5000 - 150, 50);

	utility::writer::CloseFileSystem();*/
}

void RockSimulation::InstallIsotropicStress()
{

	double iso_str;
	double diso;
	double alpha;
	double vol;
	double tol;
	std::map<std::string, Geometry*>::iterator it = geometries.find("specimen");
	switch (it->second->Shape()){
	case RECTANGLE:
	{
		geo::Rectangle *rec = dynamic_cast<geo::Rectangle*>(it->second);
		vol = rec->Sizex() * rec->Sizey();
	}
	break;
	}
	while (1){
		sort->sort();
		sort->detect();
		double sumFn = 0;
		double sumFd = 0;
		int d = (int)dim;
		for (unsigned int i = 0; i < nballs; i++){
			ball* b = &balls[i];
			sumFn += b->GetNormalForceBySumation();
		}
		iso_str = -sumFn / (d * vol);

		diso = RSP::tm_req_isostr - iso_str;
		tol = abs(diso / RSP::tm_req_isostr);
		std::cout << "Tolerance of install isotropic stress procedure : " << tol;
		if (abs(tol) <= RSP::tm_req_isostr_tol)
			break;
		for (unsigned int i = 0; i < nballs; i++){
			ball* b = &balls[i];
			sumFd += b->DeltaIsotropicStress();
		}

		alpha = -(d * vol * diso) / sumFd;
		std::cout << ", Alpha : " << alpha << std::endl;
		for (unsigned int i = 0; i < nballs; i++){
			ball* b = &balls[i];
			b->Radius() = (1.0 + alpha) * b->Radius();
			b->InertiaMass() = b->Mass() = RockElement::density * (d == DIM_2 ? b->Radius() * b->Radius() * M_PI : (4 / 3) * M_PI * pow(b->Radius(), 3));
			b->Inertia() = 2.0 * b->Mass() * pow(b->Radius(), 2) / 5.0;
		}
		utility::writer::SetFileSystem("InstallIsotropicStress");
		Solve();
		utility::writer::CloseFileSystem();
	}
}

void RockSimulation::IsotropicMakeBalls()
{
	isotropic_arrangement = true;
	double rad = RockElement::maxDiameter * 0.5;
	std::map<std::string, Geometry*>::iterator it = geometries.find("specimen");
	switch (it->second->Shape()){
	case RECTANGLE:
	{
		geo::Rectangle *rec = dynamic_cast<geo::Rectangle*>(it->second);
		unsigned int nx = static_cast<unsigned int>((rec->Sizex() / RockElement::maxDiameter) + 1e-9);
		unsigned int ny = static_cast<unsigned int>((rec->Sizey() / RockElement::maxDiameter) + 1e-9);
		nballs = nx * ny;
		balls = new ball[nballs];
		unsigned int cnt = 0;
		vector2<double> sp = rec->StartPoint();
		for (unsigned int i = 0; i < nx; i++){
			for (unsigned int j = 0; j < ny; j++){
				ball* b = &balls[cnt];
				b->ID() = cnt;
				b->Radius() = rad;
				b->Position().x = sp.x + i*RockElement::maxDiameter + rad;
				b->Position().y = sp.y + j*RockElement::maxDiameter + rad;
				b->InertiaMass() = b->Mass() = RockElement::density * (dim == DIM_2 ? b->Radius() * b->Radius() * M_PI : (4 / 3) * M_PI * pow(b->Radius(), 3));
				b->Inertia() = 2.0 * b->Mass() * pow(b->Radius(), 2) / 5.0;
				cnt++;
			}
		}
	}
	case CUBE:
	{
		geo::Cube *cube = dynamic_cast<geo::Cube*>(it->second);
		unsigned int nx = static_cast<unsigned int>((cube->Sizex() / RockElement::maxDiameter) + 1e-9);
		unsigned int ny = static_cast<unsigned int>((cube->Sizey() / RockElement::maxDiameter) + 1e-9);
		unsigned int nz = static_cast<unsigned int>((cube->Sizez() / RockElement::maxDiameter) + 1e-9);
		nballs = nx * ny * nz;
		balls = new ball[nballs];
		unsigned int cnt = 0;
		vector3<double> temp;
		vector3<double> sp = cube->StartPoint();
		for (unsigned int x = 0; x < nx; x++){
			for (unsigned int y = 0; y < ny; y++){
				for (unsigned int z = 0; z < nz; z++){
					temp.x = sp.x + x * RockElement::maxDiameter + rad;
					temp.y = sp.y + y * RockElement::maxDiameter + rad;
					temp.z = sp.z + z * RockElement::maxDiameter + rad;
					if ((temp.x >= 0.0 && temp.x <= 0.02) && (temp.y <= 0.0 && temp.y >= -0.0096) && (temp.z >= -0.01 && temp.z <= 0.01))
					{
						continue;
					}
					ball* b = &balls[cnt];
					b->ID() = cnt;
					b->Radius() = rad;
					b->Position().x = sp.x + x * RockElement::maxDiameter + rad;
					b->Position().y = sp.y + y * RockElement::maxDiameter + rad;
					b->Position().z = sp.z + z * RockElement::maxDiameter + rad;
					b->InertiaMass() = b->Mass() = RockElement::density * (4 / 3) * M_PI * pow(b->Radius(), 3);
					b->Inertia() = 2.0 * b->Mass() * pow(b->Radius(), 2) / 5.0;
					cnt++;
				}
			}
		}
// 		std::fstream of;
// 		of.open("C:/C++/add_particle.bin", std::ios::out);
// 		for (unsigned int i = 0; i < cnt; i++){
// 			ball* b = &Balls()[i];
// 			of.write((char*)&b->Position(), sizeof(vector3<double>));
// 		}
// 		of.close();
	}
	}
}

bool RockSimulation::MakeBalls()
{
	std::map<std::string, Geometry*>::iterator it = geometries.find("specimen");
	if (it == geometries.end()){
		std::cout << "Error : No exist the specimen geometry." << std::endl;
		return false;
	}
	if (RockElement::diameterRatio == 1.0){
		IsotropicMakeBalls();
	}
	else{
		double minRadius = 0.5 * (RockElement::maxDiameter / RockElement::diameterRatio);
		double maxRadius = 0.5 * RockElement::maxDiameter;
		double ru = 0.5 * (maxRadius + minRadius);
		double area = 0.0;
		switch (it->second->Shape()){
		case RECTANGLE:
		{
			geo::Rectangle *rec = dynamic_cast<geo::Rectangle*>(it->second);
			area = rec->Area();
			nballs = static_cast<unsigned int>(area * (1 - RockElement::porosity) / (M_PI * ru * ru));
			if (!nballs){
				std::cout << "Error : The number of ball is zero." << std::endl;
				return false;
			}
			balls = new ball[nballs];
			srand(1973);
			double Ap = 0.0;
			for (ball *b = ball::BeginBall(); b != NULL; b = b->NextBall()){
				double radii = 0;
				while (radii <= minRadius){
					radii = maxRadius * frand();
				}
				b->Radius() = 0.5 * radii;
				Ap += M_PI * b->Radius() * b->Radius();
			}

			double n0 = (area - Ap) / area;
			double m = sqrt((1 - RockElement::porosity) / (1 - n0));
			for (unsigned int i = 0; i < nballs; i++){
				ball* b = &balls[i];
				b->Radius() *= m;
				b->Position().x = rec->StartPoint().x + frand() * rec->Sizex();
				b->Position().y = rec->StartPoint().y + frand() * rec->Sizey();
				/*b->Acceleration().x = 0.0;
				//b->*///Acceleration().y = -9.80665;
				b->InertiaMass() = b->Mass() = RockElement::density * (dim == DIM_2 ? b->Radius() * b->Radius() * M_PI : (4 / 3) * M_PI * pow(b->Radius(), 3));
				b->Inertia() = 2.0 * b->Mass() * pow(b->Radius(), 2) / 5.0;
			}
		}
		break;
		}
	}

	// set kn & ks
	for (unsigned int i = 0; i < nballs; i++){
		ball* b = &balls[i];
		if (Dimension() == DIM_2){
			b->Friction() = RockElement::friction;
			b->Kn() = 2 * 1 * RockElement::ryoungsModulus;
			b->Ks() = b->Kn() / RockElement::rstiffnessRatio;
		}
	}

	if (specificData != ""){
		std::fstream pf;
		pf.open(specificData, std::ios::in | std::ios::binary);
		if (pf.is_open()){
			int type;
			pf.read((char*)&type, sizeof(int));
			switch (type)
			{
			case PARTICLE:
			{
				float *tpos = new float[ball::nballs * 4];
				pf.read((char*)tpos, sizeof(float)*ball::nballs * 4);
				for (unsigned int i = 0; i < nballs; i++){
					ball* b = &balls[i];
					b->Position() = vector3<double>(tpos[b->ID() * 4 + 0], tpos[b->ID() * 4 + 1], tpos[b->ID() * 4 + 2]);
					//b->Radius() = tpos[b->ID() * 4 + 3];
				}
				delete[] tpos;
			}
			break;
			}
		}
		else{
			std::cout << "ERROR : No exist specific data. the path is " << specificData << std::endl;
			return false;
		}
	}
	return true;
}

void RockSimulation::ModifyInertiaMass()
{

}

void RockSimulation::Integration0()
{
	for (unsigned int i = 0; i < nballs; i++){
		ball* b = &balls[i];
		b->Position() += dt * b->Velocity() + 0.5 * dt * dt * b->Acceleration();
	}
}

void RockSimulation::Integration1()
{
	if (dim == DIM_2){
		if (exist_bonds){
			for (std::map<ball*, Cluster>::iterator it = clusters.begin(); it != clusters.end(); it++){
				ball* b = it->first;
				if (b->Broken()){
					double div_imass = 1 / b->InertiaMass();
					b->Acceleration() = div_imass * (b->Force() + b->Mass() * gravity);
					b->Velocity() += dt * b->Acceleration();
					b->Position() += dt * b->Velocity();
				}
				else{
					double div_imass = 1 / b->InertiaMass();
					b->Acceleration() = div_imass * (b->Force());
					b->Velocity() += dt * b->Acceleration();
				}
			}
		}
		else{
			for (unsigned int i = 0; i < nballs; i++){
				ball* b = &balls[i];
				double div_imass = 1 / b->InertiaMass();
				b->Acceleration() = div_imass * (b->Force()/* + b->Mass() * gravity*/);
				if (!exist_bonds){
					b->Acceleration() += div_imass * (b->Mass() * gravity);
				}
				if (!b->VelocityFixFlag())
					b->Velocity() += dt * b->Acceleration();

				if (!b->OmegaFixFlag())
					b->Omega() += dt * (b->Moment() / (0.4 * b->InertiaMass() * b->Radius() * b->Radius()));

				b->Position() += dt * b->Velocity();
			}
		}

		for (std::map<std::string, Object*>::iterator obj = objects.begin(); obj != objects.end(); obj++){
			obj->second->Update(times);
		}
		for (std::map<std::string, Geometry*>::iterator it = geometries.begin(); it != geometries.end(); it++){
			it->second->Update(times);
		}
	}
}

void RockSimulation::CalculateionForceAndMomentOfParallelBond()
{
	double pbrad2 = 0.0;
	double A, J, I;
	for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++)
	{
		ccontact* cc = (*it);
		double pbrad = pow((*it)->PBondProps().Radius(), 2.0);
		ball *ib = (*it)->IBall();
		ball *jb = (*it)->JBall();
		if (!ib || !jb)
			continue;
		if (ib->ContactSMap().size()==0)
			continue;
		if (ib->Broken())
			continue;
		A = PI * pbrad;
		J = 0.5 * PI * pbrad * pbrad;
		I = 0.25 * PI * pbrad * pbrad;

		vector3<double> Vi = (jb->Velocity() + jb->Omega().cross((*it)->ContactPoint() - jb->Position())) - (ib->Velocity() + ib->Omega().cross((*it)->ContactPoint() - ib->Position()));
		vector3<double> Vi_s = Vi - Vi.dot(cc->Normal()) * cc->Normal();
		vector3<double> dUi_s = Simulation::dt * Vi_s;
		double dUn = (Simulation::dt * Vi).dot(cc->Normal());
		vector3<double> O = jb->Omega() - ib->Omega();
		double dTn = (Simulation::dt * O).dot(cc->Normal());
		vector3<double> dTs = O - O.dot(cc->Normal()) * cc->Normal();

		vector3<double> dFn = -cc->PBondProps().Kn() * A * dUn * cc->Normal();
		vector3<double> dFs = -cc->PBondProps().Ks() * A * dUi_s;
		vector3<double> dMn = -cc->PBondProps().Ks() * J * dTn * cc->Normal();
		vector3<double> dMs = -cc->PBondProps().Kn() * I * dTs;

		cc->SetNormalForce(cc->ScalarNormalForce() * cc->Normal() + dFn);
		cc->SetScalarNormalForce(cc->NormalForce().dot(cc->Normal()));
		cc->SetShearForce(cc->ShearForce() + dFs);

		cc->SetNormalMoment(cc->ScalarNormalMoment() * cc->Normal() + dMn);
		cc->SetScalarNormalMoment(cc->NormalMoment().dot(cc->Normal()));
		cc->SetShearMoment(cc->ShearMoment() + dMs);

		cc->SetMaxTensileStress((-cc->ScalarNormalForce() / A) + (cc->ShearMoment().length() / I) * cc->PBondProps().Radius());
		cc->SetMaxShearStress((cc->ShearForce().length() / A) + (abs(cc->ScalarNormalMoment()) / J) * cc->PBondProps().Radius());

		if (cc->MaxTensileStress() >= cc->PBondProps().NormalStrength() || cc->MaxShearStress() >= cc->PBondProps().ShearStrength())
		{
			//std::cout << "Broken particle ID : " << ib->ID() << std::endl;
			ib->VelocityFixFlag() = false;
			ib->OmegaFixFlag() = false;
 			cc->PBondProps().isBroken() = true;
			ib->Broken() = true;
		}
	}
}

bool RockSimulation::CalculationForceAndMoment(bool calcRatio)
{
	double UF = 0;
	double CF = 0;
	double sumUF = 0;
	double maxUF = 0;
	double sumCF = 0;
	double maxCF = 0;
	avgRatio = 0.0;
	maxRatio = 0.0;
	for (unsigned int i = 0; i < nballs; i++){
		ball* b = &balls[i];
		vector3<double> Fi;
		vector3<double> Mi;
		double maxSforce;
		for (std::map<ball*, ccontact>::iterator it = b->ContactPMap().begin(); it != b->ContactPMap().end(); it++){
			maxSforce = b->Friction() * it->second.NormalForce().length();
			if (it->second.ShearForce().length() > maxSforce){
				it->second.ShearForce() = (maxSforce / it->second.ShearForce().length()) * it->second.ShearForce();
			}
			Fi += it->second.NormalForce() + it->second.ShearForce();
			Mi += (it->second.ContactPoint() - b->Position()).cross(it->second.ShearForce());
		}

		for (std::map<Geometry*, ccontact>::iterator it = b->ContactWMap().begin(); it != b->ContactWMap().end(); it++){
			maxSforce = WallElement::wfriction * it->second.NormalForce().length();
			if (it->second.ShearForce().length() > maxSforce){
				it->second.ShearForce() = (maxSforce / it->second.ShearForce().length()) * it->second.ShearForce();
			}
			Fi += it->second.NormalForce() + it->second.ShearForce();
			Mi += (it->second.ContactPoint() - b->Position()).cross(it->second.ShearForce());
		}

		for (std::map<Object*, ccontact>::iterator it = b->ContactOMap().begin(); it != b->ContactOMap().end(); it++){
			maxSforce = WallElement::wfriction * it->second.NormalForce().length();
			if (it->second.ShearForce().length() > maxSforce){
				it->second.ShearForce() = (maxSforce / it->second.ShearForce().length()) * it->second.ShearForce();
			}
			Fi += it->second.NormalForce() + it->second.ShearForce();
			Mi += (it->second.ContactPoint() - b->Position()).cross(it->second.ShearForce());
		}
		Fi.x -= 0.7 * abs(Fi.x) * sign(b->Velocity().x);
		Fi.y -= 0.7 * abs(Fi.y) * sign(b->Velocity().y);
		if (dim == DIM_3)
			Fi.z -= 0.7 * abs(Fi.z) * sign(b->Velocity().z);
		UF = Fi.length();
		sumUF += (b->Mass() * b->Acceleration()).length();
		maxUF = maxUF < UF ? UF : maxUF;
		b->Force() = -Fi;
		b->Moment() = -Mi;
	}

	if (calcRatio){
		for (std::list<ccontact*>::iterator it = clist.begin(); it != clist.end(); it++){
			CF = (*it)->NormalForce().length() + (*it)->ShearForce().length();
			sumCF += CF;
			maxCF = maxCF < CF ? CF : maxCF;
		}
		double avgUF = sumUF / ball::nballs;
		double avgCF = sumCF / clist.size();
		avgRatio = avgUF / avgCF;
		maxRatio = maxUF / maxCF;
		//std::cout << "Average ratio : " << ratio << ", Maximum ratio : " << maxUF / maxCF << std::endl;
		if (avgRatio <= 0.01 || maxRatio <= 0.01){
			return true;
		}
	}
	return false;
}

bool RockSimulation::CalculationForceAndMomentWithoutP2P()
{
	double UF = 0;
	double CF = 0;
	double sumUF = 0;
	double maxUF = 0;
	double sumCF = 0;
	double maxCF = 0;
	avgRatio = 0.0;
	maxRatio = 0.0;
	utility::writer::pick_force = 0.0;
	for (unsigned int i = 0; i < nballs; i++){
		ball* b = &balls[i];
		vector3<double> Fi;
		vector3<double> Mi;
		double maxSforce;

	/*	if (b->Broken()){
			for (std::map<ball*, ccontact>::iterator it = b->ContactPMap().begin(); it != b->ContactPMap().end(); it++){
				maxSforce = b->Friction() * it->second.NormalForce().length();
				if (it->second.ShearForce().length() > maxSforce){
					it->second.ShearForce() = (maxSforce / it->second.ShearForce().length()) * it->second.ShearForce();
				}
				Fi += it->second.NormalForce() + it->second.ShearForce();
				Mi += (it->second.ContactPoint() - b->Position()).cross(it->second.ShearForce());
			}
		}*/

		for (std::map<Geometry*, ccontact>::iterator it = b->ContactSMap().begin(); it != b->ContactSMap().end(); it++){
			maxSforce = WallElement::wfriction * it->second.NormalForce().length();
			if (it->second.ShearForce().length() > maxSforce){
				it->second.ShearForce() = (maxSforce / it->second.ShearForce().length()) * it->second.ShearForce();
			}
			Fi += it->second.NormalForce() + it->second.ShearForce();
			Mi += (it->second.ContactPoint() - b->Position()).cross(it->second.ShearForce());
			utility::writer::pick_force += Fi;
		}
		
		//for (std::map<Object*, ccontact>::iterator it = b->ContactOMap().begin(); it != b->ContactOMap().end(); it++){
		//	maxSforce = WallElement::wfriction * it->second.NormalForce().length();
		//	if (it->second.ShearForce().length() > maxSforce){
		//		it->second.ShearForce() = (maxSforce / it->second.ShearForce().length()) * it->second.ShearForce();
		//	}
		//	Fi += it->second.NormalForce() + it->second.ShearForce();
		//	Mi += (it->second.ContactPoint() - b->Position()).cross(it->second.ShearForce());
		//}

		Fi.x -= 0.7 * abs(Fi.x) * sign(b->Velocity().x);
		Fi.y -= 0.7 * abs(Fi.y) * sign(b->Velocity().y);
		Fi.z -= 0.7 * abs(Fi.z) * sign(b->Velocity().z);
		UF = Fi.length();
		//sumUF += (b->Mass() * b->Acceleration()).length();
		//maxUF = maxUF < UF ? UF : maxUF;
		b->Force() = -Fi;
		b->Moment() = -Mi;
	}

	return false;
}

bool RockSimulation::RunCycle(unsigned int cyc, unsigned int savecyc)
{
	parSIM::timer tmer;
	time_t t;
	tm date;
	time(&t);
	localtime_s(&date, &t);

	unsigned int curRun = 0;
	unsigned int eachRun = 0;
	times = curRun * dt;
	double elapsed_time = 0;
	std::cout << "---------------------------------------------------------------------------------" << std::endl
		<< "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |        Date       |" << std::endl
		<< "---------------------------------------------------------------------------------" << std::endl;
	std::ios::right;
	std::setprecision(6);
	if (utility::writer::Save(curRun)){
		std::cout << "| " << std::setw(9) << utility::writer::part - 1 << std::setw(12) << times << std::setw(10) << eachRun << std::setw(11) << curRun << std::setw(15) << tmer.GetElapsedTimeD() << std::setw(21) << make_date_form(date) << std::endl;
	}
	curRun++;
	//Integration();

	tmer.Start();
	while (curRun < cyc){
		times = curRun * dt;
		//Integration0();

		//std::cout << curRun << std::endl;
		//if (!exist_bonds){
		sort->sort();
		sort->detect();
		CalculationForceAndMoment();
		//}
		/*	if(exist_bonds)
		CalculateionForceAndMomentOfParallelBond();*/

		Integration1();
		if (!((curRun) % savecyc)){
			time(&t);
			localtime_s(&date, &t);
			tmer.Stop();
			if (utility::writer::Save(curRun)){
				std::cout << "| " << std::setw(9) << utility::writer::part - 1 << std::setw(12) << times << std::setw(10) << eachRun << std::setw(11) << curRun << std::setw(15) << tmer.GetElapsedTimeD() << std::setw(21) << make_date_form(date) << std::endl;
			}
			eachRun = 0;
			tmer.Start();
		}
		curRun++;
		eachRun++;
	}

	return true;
}

bool RockSimulation::Solve()
{
	unsigned int ncyc = 50000;
	unsigned int curRun = 0;
	while (curRun < ncyc){
		//Integration0();
		sort->sort();
		sort->detect();
		if (CalculationForceAndMoment(true))
			break;
		Integration1();
		if (!((curRun) % 100)){
			std::cout << "Current : " << curRun << ", Average ratio : " << avgRatio << ", Maximum ratio : " << maxRatio << std::endl;
			utility::writer::Save(curRun);
		}
		curRun++;
	}

	return true;
}

bool RockSimulation::RockRunCycle(unsigned int cyc, unsigned int savecyc)
{
	parSIM::timer tmer;
	time_t t;
	tm date;
	time(&t);
	localtime_s(&date, &t);

	unsigned int curRun = 0;
	unsigned int eachRun = 0;
	times = curRun * dt;
	std::cout << dt << std::endl;
	double elapsed_time = 0;
	std::cout << "---------------------------------------------------------------------------------" << std::endl
		<< "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |        Date       |" << std::endl
		<< "---------------------------------------------------------------------------------" << std::endl;
	std::ios::right;
	std::setprecision(6);
	if (utility::writer::Save(curRun)){
		std::cout << "| " << std::setw(9) << utility::writer::part - 1 << std::setw(12) << times << std::setw(10) << eachRun << std::setw(11) << curRun << std::setw(15) << tmer.GetElapsedTimeD() << std::setw(21) << make_date_form(date) << std::endl;
	}
	curRun++;
	//Integration();

	tmer.Start();
	while (curRun < cyc){
		times = curRun * dt;
		//for (unsigned int i = 0; i < ball::nballs; i++){
		//	ball* ib = &Balls()[i];
		//	
		//	for (std::map<std::string, Object*>::iterator obj = objects.begin(); obj != objects.end(); obj++){
		//		//std::cout << i << std::endl;
		//		obj->second->Collision(ib);
		//	}
		//}
		sort->sort();
		sort->detectOnlyShape();

		CalculationForceAndMomentWithoutP2P();
		Integration1();
		if (exist_bonds)
			CalculateionForceAndMomentOfParallelBond();


		if (!((curRun) % savecyc)){
			time(&t);
			localtime_s(&date, &t);
			tmer.Stop();
			if (utility::writer::Save(curRun)){
				std::cout << "| " << std::setw(9) << utility::writer::part - 1 << std::setw(12) << times << std::setw(10) << eachRun << std::setw(11) << curRun << std::setw(15) << tmer.GetElapsedTimeD() << std::setw(21) << make_date_form(date) << std::endl;
			}
			eachRun = 0;
			tmer.Start();
		}
		curRun++;
		eachRun++;
	}

	return true;
}