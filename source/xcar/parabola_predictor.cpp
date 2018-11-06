#include "parabola_predictor.h"

utility::parabola_predictor::parabola_predictor()
	: data3(NULL)
{

}

utility::parabola_predictor::~parabola_predictor()
{
	if(data3) delete data3; data3 = NULL;
}

void utility::parabola_predictor::init(double* _data, int _dataSize)
{
	data = _data;
	dataSize = _dataSize;
	data3 = new vector<double>(dataSize*3);
}

bool utility::parabola_predictor::apply(unsigned int it)
{
	int insertID = (it - 1) % 3;
	for(int i(0); i < dataSize; i++) (*data3)(insertID * dataSize + i) = *(data + i);
	if(it < 3) return false;
	double cur_xp = dt * it;
	xp = vector3<double>((it - 3) * dt, (it - 2) * dt, (it - 1) * dt);
	switch(insertID)
	{
	case 2: idx = VEC3I(0, 1, 2); break;
	case 0: idx = VEC3I(1, 2, 0); break;
	case 1: idx = VEC3I(2, 0, 1); break;
	}
	A.set(xp.x * xp.x, xp.x, 1
		, xp.y * xp.y, xp.y, 1
		, xp.z * xp.z, xp.z, 1);

	A.inv();
	//fstream of;
	//of.open("C:/predictor_data.txt", ios::out);

	for(int i(0); i < dataSize; i++)
	{
		yp = vector3<double>((*data3)(idx.x * dataSize + i), (*data3)(idx.y * dataSize + i), (*data3)(idx.z * dataSize + i));
		//std::cout << yp << std::endl;
		coeff = A * yp;

		data[i] = coeff.x * cur_xp * cur_xp + coeff.y * cur_xp + coeff.z;
	}
	//of.close();

	return true;
}