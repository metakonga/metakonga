#include "startingModel.h"

startingModel::startingModel()
	: dem_pos(NULL)
	, dem_vel(NULL)
	, dem_avel(NULL)
	, rhs(NULL)
	, is_alloc_dem(false)
	, is_alloc_mbd(false)
{

}

startingModel::~startingModel()
{
	if (rhs) delete[] rhs; rhs = NULL;
	if (dem_pos) delete[] dem_pos; dem_pos = NULL;
	if (dem_vel) delete[] dem_vel; dem_vel = NULL;
	if (dem_avel) delete[] dem_avel; dem_avel = NULL;
}

void startingModel::copyDEMData(unsigned int np, double* p, double* v, double* av)
{
	if(p)
		memcpy(p, dem_pos, sizeof(double) * 4 * np);
	if(v)
		memcpy(v, dem_vel, sizeof(double) * 3 * np);
	if(av)
		memcpy(av, dem_avel, sizeof(double) * 3 * np);
}

void startingModel::setDEMData(unsigned int np, QFile& qf)
{
	int flag = 0;
	int precision = 0;
	qf.read((char*)&precision, sizeof(int));
	qf.read((char*)&flag, sizeof(int));
	if (flag == 1)
	{
		if (precision == 1)
		{
			if (!dem_pos_f)
				dem_pos_f = new float[np * 4];
			if (!dem_vel_f)
				dem_vel_f = new float[np * 3];
 			if (!dem_avel_f)
 				dem_avel_f = new float[np * 3];

			qf.read((char*)dem_pos_f, sizeof(float) * np * 4);
			qf.read((char*)dem_vel_f, sizeof(float) * np * 3);
			qf.read((char*)dem_avel_f, sizeof(float) * np * 3);
		}
		else
		{
			if (!dem_pos)
				dem_pos = new double[np * 4];
			if (!dem_vel)
				dem_vel = new double[np * 3];
			if (!dem_avel)
				dem_avel = new double[np * 3];

			qf.read((char*)dem_pos, sizeof(double) * np * 4);
			qf.read((char*)dem_vel, sizeof(double) * np * 3);
			qf.read((char*)dem_avel, sizeof(double) * np * 3);
		}		
	}
}

void startingModel::setMBDData(QFile& qf)
{
	int flag = 0;
	unsigned int sz = 0;
	qf.read((char*)&flag, sizeof(int));
	qf.read((char*)&sz, sizeof(unsigned int));
	if (flag == 2)
	{
		unsigned int szn = 0;
		char name[255] = {0, };
		resultStorage::pointMassResultData pmrd;
		for (unsigned int i = 0; i < sz; i++)
		{
			qf.read((char*)&szn, sizeof(unsigned int));
			qf.read(name, sizeof(char)*szn);
			qf.read((char*)&pmrd, sizeof(resultStorage::pointMassResultData));
			mbd_bodies[name] = pmrd;

		}
		unsigned int d = 0;
		qf.read((char*)&d, sizeof(unsigned int));
		rhs = new double[d];
		qf.read((char*)rhs, sizeof(double) * d);
	}
}

