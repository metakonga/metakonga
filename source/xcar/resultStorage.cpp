#include "resultStorage.h"
#include <QDebug>
#include <QFile>

resultStorage *RS;

resultStorage::resultStorage()
	: isAllocMemory(false)
	, type(NULL)
	, pos(NULL)
	, vel(NULL)
	, press(NULL)
	, isf(NULL)
	, color(NULL)
	, time(NULL)
	, cramp(NULL)
	, pos_f(NULL)
	, vel_f(NULL)
	, press_f(NULL)
	, cpart(0)
	, np(0)
	, nfluid(0)
	, nfloat(0)
	, nbound(0)
	, ndummy(0)
	, maxPress(0)
	, pdata_type(3)
{
	RS = this;
}

resultStorage::~resultStorage()
{
	clearMemory();
}

resultStorage* resultStorage::RStorage()
{
	return RS;
}

unsigned int resultStorage::Np()
{
	return np;
}

unsigned int resultStorage::NFluid()
{
	return nfluid;
}

unsigned int resultStorage::NFloat()
{
	return nfloat;
}

unsigned int resultStorage::NBound()
{
	return nbound;
}

unsigned int resultStorage::NDummy()
{
	return ndummy;
}

ucolors::colorMap * resultStorage::ColorMap()
{
	return cramp;
}

double resultStorage::RequriedMemory(unsigned int n, unsigned int npart, solver_type ts)
{
	int pn = ts == DEM ? 4 : 3;
	if (isSingle)
	{
		float m_size = 0.0f;
		m_size += sizeof(float) * npart;
		m_size += sizeof(float) * n * pn * npart;
		return m_size;
	}
	double m_size = 0.0;
	m_size += sizeof(double) * npart;
	m_size += sizeof(double) * n * pn * npart;
	//m_size += sizeof(double) * n * 3 * npart;
	//m_size += sizeof(double) * n * 4 * npart;
	return m_size;
}

// void resultStorage::defineUserResult(QString nm)
// {
// 	
// }

void resultStorage::insertTimeData(double ct)
{
	time[cpart] = ct; 
}

void resultStorage::insertGLineData(QString tg, VEC3D& b)
{
	glLine[tg].push_back(b);
}

void resultStorage::insertPartName(QString nm)
{
	pList.push_back(nm);
}

void resultStorage::insertPointMassResult(QString& nm, pointMassResultData& pmrd)
{
	pmrs[nm].push_back(pmrd);
}

void resultStorage::insertReactionForceResult(QString& nm, reactionForceData& rfd)
{
	rfrs[nm].push_back(rfd);
}

double* resultStorage::getPartPosition(unsigned int pt)
{
	return pos + (np * pdata_type * pt);
}

double* resultStorage::getPartVelocity(unsigned int pt)
{
	return vel + (np * 3 * pt);
}

double* resultStorage::getPartPressure(unsigned int pt)
{
	return press + (np * pt);
}

float* resultStorage::getPartPosition_f(unsigned int pt)
{
	return pos_f + (np * pdata_type * pt);
}

float* resultStorage::getPartVelocity_f(unsigned int pt)
{
	return vel_f + (np * 3 * pt);
}

float* resultStorage::getPartPressure_f(unsigned int pt)
{
	return press_f + (np * pt);
}

double* resultStorage::getPartColor(unsigned int pt)
{
	return color + (np * 4 * pt);
}

particle_type* resultStorage::getParticleType()
{
	return type;
}

bool* resultStorage::getPartFreeSurface(unsigned int pt)
{
	return isf + (np * pt);
}

void resultStorage::clearMemory()
{
	if (time) delete[] time; time = NULL;
	if (type) delete[] type; type = NULL;
	if (pos) delete[] pos; pos = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (press) delete[] press; press = NULL;
	if (pos_f) delete[] pos_f; pos_f = NULL;
	if (vel_f) delete[] vel_f; vel_f = NULL;
	if (press_f) delete[] press_f; press_f = NULL;
	if (isf) delete[] isf; isf = NULL;
	if (color) delete[] color; color = NULL;
	if (cramp) delete cramp; cramp = NULL;
//	qDeleteAll(pmrd);
	isAllocMemory = false;
}

void resultStorage::allocMemory(unsigned int n, unsigned int npart, solver_type ts)
{
	int pn = ts == DEM ? 4 : 3;
	time = new double[npart]; memset(time, 0, sizeof(double)*npart);
	if (isSingle)
	{
		pdata_type = pn;
		pos_f = new float[(n * pn) * npart]; memset(pos_f, 0, sizeof(float) * n * pn * npart);
		if (ts == SPH)
		{
			type = new particle_type[n]; memset(type, 0, sizeof(particle_type)*n);
			press_f = new float[n * npart]; memset(press, 0, sizeof(float) * n * npart);
			isf = new bool[n * npart]; memset(isf, 0, sizeof(bool) * n * npart);
		}
	}
	else
	{		
		pos = new double[(n * pn) * npart]; memset(pos, 0, sizeof(double) * n * pn * npart);
		//vel = new double[(n * 3) * npart]; memset(vel, 0, sizeof(double) * n * 3 * npart);	
		//color = new double[(n * 4) * npart]; memset(color, 0, sizeof(double) * n * 4 * npart);
		//cramp = new ucolors::colorMap(npart);
		pdata_type = pn;
		if (ts == SPH)
		{
			type = new particle_type[n]; memset(type, 0, sizeof(particle_type)*n);
			press = new double[n * npart]; memset(press, 0, sizeof(double) * n * npart);
			isf = new bool[n * npart]; memset(isf, 0, sizeof(bool) * n * npart);
		}
		else if (ts == DEM)
		{
			//cramp->setTarget(ucolors::COLORMAP_VELOCITY_MAG);
		}
	}
	
	
// 	QStringList pm = pmrd.keys();
// 	for (unsigned int i = 0; i < pmrd.size(); i++)
// 	{
// 		pmrd[pm.at(i)] = new pointMassResultData[npart];
// 	}
	isAllocMemory = true;
}

void resultStorage::setResultMemorySPH(unsigned int npart)
{
// 	nfluid = nf;
// 	nfloat = nft;
// 	nbound = nb;
// 	ndummy = nd;
// 	np = nf + nft + nb + nd;
// // 	if (np != _np)
// 		return;
// 	double *t_time = new double[1];				memset(t_time, 0, sizeof(double));
// 	tParticle *t_type = new tParticle[np];		memset(t_type, 0, sizeof(tParticle) * np);
// 	double *t_pos = new double[(np * 3)];		memset(t_pos, 0, sizeof(double) * np * 3);
// 	double *t_vel = new double[(np * 3)];		memset(t_vel, 0, sizeof(double) * np * 3);
// 	double *t_press = new double[np];			memset(t_press, 0, sizeof(double) * np);
// 	bool *t_isf = new bool[np];					memset(t_isf, 0, sizeof(bool) * np);
// 	double* t_color = new double[(np * 4)];		memset(t_color, 0, sizeof(double) * np * 4);
// 
// 	memcpy(t_time, time, sizeof(double));
// 	memcpy(t_type, type, sizeof(tParticle) * np);
// 	memcpy(t_pos, pos, sizeof(double) * np * 3);
// 	memcpy(t_vel, vel, sizeof(double) * np * 3);
// 	memcpy(t_press, press, sizeof(double) * np);
// 	memcpy(t_isf, isf, sizeof(bool) * np);
// 	memcpy(t_color, color, sizeof(double) * np * 4);
	total_parts = npart;
	clearMemory();
	if (!isAllocMemory)
		allocMemory(np, total_parts);
// 	if (!cramp)
// 	{
// 		delete cramp;
// 	}
	//cramp = new ucolors::colorMap(total_parts);
// 	memcpy(time, t_time, sizeof(double));
// 	memcpy(type, t_type, sizeof(tParticle) * np);
// 	memcpy(pos, t_pos, sizeof(double) * np * 3);
// 	memcpy(vel, t_vel, sizeof(double) * np * 3);
// 	memcpy(press, t_press, sizeof(double) * np);
// 	memcpy(isf, t_isf, sizeof(bool) * np);
// 	memcpy(color, t_color, sizeof(double) * np * 4);
// 
// 	delete[] t_time; t_time = NULL;
// 	delete[] t_type; t_type = NULL;
// 	delete[] t_pos; t_pos = NULL;
// 	delete[] t_vel; t_vel = NULL;
// 	delete[] t_press; t_press = NULL;
// 	delete[] t_isf; t_isf = NULL;
// 	delete[] t_color; t_color = NULL;
}

void resultStorage::setResultMemoryDEM(bool _isSingle, unsigned int npart, unsigned int _np)
{
	np = _np;
	total_parts = npart;
	isSingle = _isSingle;
	clearMemory();
	if (!isAllocMemory)
	{
		allocMemory(np, total_parts, DEM);			
	}
}

void resultStorage::insertTimeDoubleResult(QString& nm, time_double& td)
{
	tds[nm].push_back(td);// td;
}

void resultStorage::definePartDatasSPH(bool isOnlyCountNumPart, int index)
{
	unsigned int _np = np;
// 	if (!cramp)
// 	{
// 		if (!cpart)
// 			cramp = new ucolors::colorMap(1);
// 		else
// 			cramp = new ucolors::colorMap(cpart);
// 	}

	if (!isOnlyCountNumPart)
	{
		unsigned int id = index > -1 ? index : cpart;
		VEC3D min_vel = DBL_MAX;
		VEC3D max_vel = -DBL_MAX;
		double min_pre = DBL_MAX;
		double max_pre = -DBL_MAX;
		for (unsigned int j = 0; j < _np; j++)
		{
			unsigned int idx = np * 3 * id + j * 3;
			max_vel.x = abs(vel[idx + 0]) > max_vel.x ? abs(vel[idx + 0]) : max_vel.x;
			max_vel.y = vel[idx + 1] > max_vel.y ? vel[idx + 1] : max_vel.y;
			max_vel.z = vel[idx + 2] > max_vel.z ? vel[idx + 2] : max_vel.z;

			min_vel.x = 0.0;// abs(vel[idx + 0]) < min_vel.x ? abs(vel[idx + 0]) : min_vel.x;
			min_vel.y = vel[idx + 1] < min_vel.y ? vel[idx + 1] : min_vel.y;
			min_vel.z = vel[idx + 2] < min_vel.z ? vel[idx + 2] : min_vel.z;

			max_pre = press[np * id + j] > max_pre ? press[np * id + j] : max_pre;
			min_pre = press[np * id + j] < min_pre ? press[np * id + j] : min_pre;

			max_pre = press[np * id + j] > max_pre ? press[np * id + j] : max_pre;
			min_pre = press[np * id + j] < min_pre ? press[np * id + j] : min_pre;
		}
		cramp->setMinMax(id, min_vel.x, min_vel.y, min_vel.z, max_vel.x, max_vel.y, max_vel.z, min_pre, max_pre);

		unsigned int idx = np * id;

		double grad = maxPress * 0.1f;
		double t = 0.f;
		for (unsigned int j = 0; j < _np; j++){

			unsigned int cidx = (idx * 4) + (j * 4);
			unsigned int pidx = (idx * 3) + (j * 3);
			if (isf[idx + j]){
				color[cidx + 0] = 1.0f;
				color[cidx + 1] = 1.0f;
				color[cidx + 2] = 1.0f;
				color[cidx + 3] = 1.0f;
				continue;
			}
			if (type[j] == FLOATING)
			{
				color[cidx + 0] = 0.0f;
				color[cidx + 1] = 0.0f;
				color[cidx + 2] = 0.0f;
				color[cidx + 3] = 1.0f;
				continue;
			}
			if (type[j] == FLOATING_DUMMY)
			{
				color[cidx + 0] = 0.0f;
				color[cidx + 1] = 1.0f;
				color[cidx + 2] = 1.0f;
				color[cidx + 3] = 1.0f;
				continue;
			}
		

			if (cramp->target() == ucolors::COLORMAP_PRESSURE)
				cramp->getColorRamp(id, press[idx + j], &(color[cidx]));
			else if (cramp->target() == ucolors::COLORMAP_VELOCITY_X)
				cramp->getColorRamp(id, abs(vel[pidx + 0]), &(color[cidx]));
			color[cidx + 3] = 1.0f;
		}
		if (index == -1){
			cpart++;
		}
	}
}

void resultStorage::definePartDatasDEM(bool isOnlyCountNumPart, int index /*= -1*/)
{
	if (!isOnlyCountNumPart)
	{
		unsigned int id = index > -1 ? index : cpart;
		VEC3D min_vel = DBL_MAX;
		VEC3D max_vel = -DBL_MAX;
		double min_pre = DBL_MAX;
		double max_pre = -DBL_MAX;
		for (unsigned int j = 0; j < np; j++)
		{
			unsigned int idx = np * 3 * id + j * 3;
			max_vel.x = abs(vel[idx + 0]) > max_vel.x ? abs(vel[idx + 0]) : max_vel.x;
			max_vel.y = vel[idx + 1] > max_vel.y ? vel[idx + 1] : max_vel.y;
			max_vel.z = vel[idx + 2] > max_vel.z ? vel[idx + 2] : max_vel.z;

			min_vel.x = 0.0;// abs(vel[idx + 0]) < min_vel.x ? abs(vel[idx + 0]) : min_vel.x;
			min_vel.y = vel[idx + 1] < min_vel.y ? vel[idx + 1] : min_vel.y;
			min_vel.z = vel[idx + 2] < min_vel.z ? vel[idx + 2] : min_vel.z;

// 			max_pre = press[np * id + j] > max_pre ? press[np * id + j] : max_pre;
// 			min_pre = press[np * id + j] < min_pre ? press[np * id + j] : min_pre;
// 
// 			max_pre = press[np * id + j] > max_pre ? press[np * id + j] : max_pre;
// 			min_pre = press[np * id + j] < min_pre ? press[np * id + j] : min_pre;
		}
		cramp->setMinMax(id, min_vel.x, min_vel.y, min_vel.z, max_vel.x, max_vel.y, max_vel.z, 0, 0);

		unsigned int idx = np * id;

		double grad = maxPress * 0.1f;
		double t = 0.f;
		for (unsigned int j = 0; j < np; j++){

			unsigned int cidx = (idx * 4) + (j * 4);
			unsigned int pidx = (idx * 3) + (j * 3);
			VEC3D v(vel[pidx + 0], vel[pidx + 1], vel[pidx + 2]);
			if (cramp->target() == ucolors::COLORMAP_PRESSURE)
				cramp->getColorRamp(id, press[idx + j], &(color[cidx]));
			else if (cramp->target() == ucolors::COLORMAP_VELOCITY_X)
				cramp->getColorRamp(id, abs(v.x), &(color[cidx]));
			else if (cramp->target() == ucolors::COLORMAP_VELOCITY_Y)
				cramp->getColorRamp(id, abs(v.y), &(color[cidx]));
			else if (cramp->target() == ucolors::COLORMAP_VELOCITY_Z)
				cramp->getColorRamp(id, abs(v.z), &(color[cidx]));
			else if (cramp->target() == ucolors::COLORMAP_VELOCITY_MAG)
				cramp->getColorRamp(id, abs(v.length()), &(color[cidx]));
			color[cidx + 3] = 1.0f;
		}
		if (index == -1){
			cpart++;
		}
	}
}

void resultStorage::setInformationSPH(unsigned int _np, unsigned int _nf, unsigned int _nft, unsigned int _nb, unsigned int _nd)
{
	np = _np;
	nfluid = _nf;
	nfloat = _nft;
	nbound = _nb;
	ndummy = _nd;
}

double resultStorage::getPartTime(unsigned int pt)
{
	return time[pt];
}

void resultStorage::setPartDataFromBinary(unsigned int pt, QString file)
{
	//QFile pf(rList.at(idx));
	QFile pf(file);
	pf.open(QIODevice::ReadOnly);
	double time = 0.0;
	unsigned int _np = 0;
	pf.read((char*)&time, sizeof(double));
	pf.read((char*)&_np, sizeof(unsigned int));
	//pf.read((char*)type, sizeof(particle_type) * np);
	int sz = isSingle ? sizeof(float) : sizeof(double);
	pf.read((char*)(pos + (_np * pdata_type * pt)), sz * np * 4);
	pf.close();
	//pos + (np * pdata_type * pt);
// 	char pname[256] = { 0, };
// 	QString part_name;
// 	part_name.sprintf("part%04d", pt);
	insertTimeData(time);
	insertPartName(file);
	pf.close();
}

void resultStorage::openSphResultFiles(QStringList& slist)
{
	rList = slist;
	clearMemory();
	if (!isAllocMemory)
		allocMemory(np, slist.size());
	unsigned int i = 0;
	foreach(QString str, slist)
	{
		int begin = str.lastIndexOf("/");
		int end = str.lastIndexOf(".");
		QString partName = str.mid(begin+1, end - begin - 1);
		insertPartName(partName);
		QString file = str;
		QFile pf(file);
		pf.open(QIODevice::ReadOnly);
		pf.read((char*)&(time[i]), sizeof(double));
		pf.read((char*)&np, sizeof(unsigned int));
		pf.read((char*)&nfluid, sizeof(unsigned int));
		pf.read((char*)&nfloat, sizeof(unsigned int));
		pf.read((char*)&nbound, sizeof(unsigned int));
		pf.read((char*)&ndummy, sizeof(unsigned int));
		pf.read((char*)type, sizeof(particle_type)*np);
		pf.read((char*)&(pos[(np * 3) * i]), sizeof(double) * np * 3);
		pf.read((char*)&(vel[(np * 3) * i]), sizeof(double) * np * 3);
		pf.read((char*)&(press[np * i]), sizeof(double) * np);
		pf.read((char*)&(isf[np * i]), sizeof(bool) * np);
		pf.close();
		definePartDatasSPH(false, i);
		i++;
		qDebug() << i;
	}
}

void resultStorage::insertDataSPH(particle_type* tp, double* _p, double* _v, double* _prs, bool isCalcContour /*= false*/)
{

}

void resultStorage::exportEachResult2TXT(QString path)
{
	QFile qf_list(path + "_result_file_list.rfl");
	qf_list.open(QIODevice::WriteOnly);
	QTextStream qts_list(&qf_list);
	QMapIterator<QString, QList<pointMassResultData>> m_pmrd(pmrs);
	while (m_pmrd.hasNext())
	{
		m_pmrd.next();
		QString file_name = path + "/" + m_pmrd.key() + ".txt";
		QFile qf(file_name);
		qf.open(QIODevice::WriteOnly);
		QTextStream qts(&qf);
		qts << "time "
			<< "px " << "py " << "pz " << "ep0 " << "ep1 " << "ep2 " << "ep3 "
			<< "vx " << "vy " << "vz " << "wx " << "wy " << "wz "
			<< "ax " << "ay " << "az " << "apx " << "apy " << "apz " 
			<< "ea0 " << "ea1 " << "ea2 " << "ea3 " 
			<< "tFx " << "tFy " << "tFz " 
			<< "cFx " << "cFy " << "cFz "
			<< "hFx " << "hFy " << "hFz " << endl;
		foreach(pointMassResultData p, m_pmrd.value())
		{
			qts << p.time
				<< " " << p.pos.x << " " << p.pos.y << " " << p.pos.z
				<< " " << p.ep.e0 << " " << p.ep.e1 << " " << p.ep.e2 << " " << p.ep.e3
				<< " " << p.vel.x << " " << p.vel.y << " " << p.vel.z
				<< " " << p.omega.x << " " << p.omega.y << " " << p.omega.z
				<< " " << p.acc.x << " " << p.acc.y << " " << p.acc.z
				<< " " << p.alpha.x << " " << p.alpha.y << " " << p.alpha.z 
				<< " " << p.ea.e0 << " " << p.ea.e1 << " " << p.ea.e2 << " " << p.ea.e3 
				<< " " << p.tforce.x << " " << p.tforce.y << " " << p.tforce.z
				<< " " << p.cforce.x << " " << p.cforce.y << " " << p.cforce.z
				<< " " << p.hforce.x << " " << p.hforce.y << " " << p.hforce.z << endl;
		}
		qf.close();
		qts_list << "point_mass_result " << file_name << endl;
	}
	QMapIterator<QString, QList<reactionForceData>> m_rfd(rfrs);
	while (m_rfd.hasNext())
	{
		m_rfd.next();
		QString file_name = path + "/" + m_rfd.key() + ".txt";
		QFile qf(file_name);
		qf.open(QIODevice::WriteOnly);
		QTextStream qts(&qf);
		qts << "time "
			<< "fix " << "fiy " << "fiz " << "ri0 " << "ri1 " << "ri2 " << "ri3 "
			<< "fjx " << "fjy " << "fjz " << "rj0 " << "rj1 " << "rj2 " << "rj3 " << endl;
		foreach(reactionForceData p, m_rfd.value())
		{
			qts << p.time
				<< " " << p.iAForce.x << " " << p.iAForce.y << " " << p.iAForce.z
				<< " " << p.iRForce.x << " " << p.iRForce.y << " " << p.iRForce.z << " " << p.iRForce.w
				<< " " << p.jAForce.x << " " << p.jAForce.y << " " << p.jAForce.z
				<< " " << p.jRForce.x << " " << p.jRForce.y << " " << p.jRForce.z << " " << p.jRForce.w << endl;
		}
		qf.close();
	}
	QMapIterator<QString, QList<time_double>> m_tds(tds);
	while (m_tds.hasNext())
	{
		m_tds.next();
		QString file_name = path + "/" + m_tds.key() + ".txt";
		QFile qf(file_name);
		qf.open(QIODevice::WriteOnly);
		QTextStream qts(&qf);
		qts << "time " << "power " << endl;
		foreach(time_double td, m_tds.value())
		{
			qts << td.time << " " << td.value << endl;
		}
		qf.close();
	}
	qts_list << "particle_result " << pList.size() << np << isSingle << endl;
	foreach(QString n, pList)
	{
		qts_list << "part_result_list " << n << endl;
	}
	qf_list.close();
}

void resultStorage::openResultList(QString f)
{
	QFile qf(f);
	qf.open(QIODevice::ReadOnly);
	QTextStream qts(&qf);
	QString ch;
	unsigned int pt = 0;
	while (!qf.atEnd())
	{
		qts >> ch;
		if (ch == "point_mass_result")
		{
			qts >> ch;
			QFile pmr(ch);
			int begin = ch.lastIndexOf("/");
			int end = ch.lastIndexOf(".");
			QString fn = ch.mid(begin + 1, end - begin);
			QTextStream qts_pm(&pmr);
			pointMassResultData p;
			while (!pmr.atEnd())
			{
				for (int i = 0; i < 33; i++)
					qts_pm >> ch;
				qts_pm
					>> p.time >> p.pos.x >> p.pos.y >> p.pos.z
					>> p.ep.e0 >> p.ep.e1 >> p.ep.e2 >> p.ep.e3
					>> p.vel.x >> p.vel.y >> p.vel.z
					>> p.omega.x >> p.omega.y >> p.omega.z
					>> p.acc.x >> p.acc.y >> p.acc.z
					>> p.alpha.x >> p.alpha.y >> p.alpha.z
					>> p.ea.e0 >> p.ea.e1 >> p.ea.e2 >> p.ea.e3
					>> p.tforce.x >> p.tforce.y >> p.tforce.z
					>> p.cforce.x >> p.cforce.y >> p.cforce.z
					>> p.hforce.x >> p.hforce.y >> p.hforce.z;
				insertPointMassResult(fn, p);
			}			
		}
		else if (ch == "particle_result")
		{
			unsigned int _npart = 0;
			unsigned int _np = 0;
			int iss = 0;
			qts >> _npart >> _np >> iss;
			setResultMemoryDEM(iss, _npart, _np);
		}
		else if (ch == "part_result_list")
		{
			qts >> ch;
			setPartDataFromBinary(pt, ch);
			pt++;
		}
	}
	qf.close();
}

QMap<QString, QList<VEC3D>>& resultStorage::linePointResults()
{
	return glLine;
}

// QMap<QString, pointMassResultData*>& resultStorage::PointMassResultData()
// {
// 	return pmrd;
// }
// 
// void resultStorage::preparePointMassResultDataMemory(QString nm)
// {
// 	pmrd[nm] = NULL;
// }
// 
// void resultStorage::exportPointMassResultData2TXT()
// {
// // 	QString file_name = model::path + model::name + "/" + name + ".txt";
// // 	QFile qf(file_name);
// // 	qf.open(QIODevice::WriteOnly);
// // 	QTextStream qts(&qf);
// }

QStringList& resultStorage::partList()
{
	return pList;
}
