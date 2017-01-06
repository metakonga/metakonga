#include "cube.h"
#include "modeler.h"


cube::cube(modeler* _md, QString& _name, tMaterial _mat, tRoll _roll)
	: object(_md, _name, CUBE, _mat, _roll)
{

}

cube::cube(const cube& _cube)
	: ori(_cube.origin())
	, min_p(_cube.min_point())
	, max_p(_cube.max_point())
	, size(_cube.cube_size())
	, object(_cube)
{
	for (int i = 0; i < 6; i++){
		planes[i] = _cube.planes_data(i);
	}
}

cube::~cube()
{

}

bool cube::define(vector3<float>& min, vector3<float>& max)
{
	min_p = min;
	max_p = max;

	size.x = (max_p - vector3<float>(min_p.x, max_p.y, max_p.z)).length();
	size.y = (max_p - vector3<float>(max_p.x, min_p.y, max_p.z)).length();
	size.z = (max_p - vector3<float>(max_p.x, max_p.y, min_p.z)).length();

	planes[0].define(min_p, min_p + vector3<float>(0, 0, size.z), min_p + vector3<float>(size.x, 0, 0));
	planes[1].define(min_p, min_p + vector3<float>(0, size.y, 0), min_p + vector3<float>(0, 0, size.z));
	planes[2].define(min_p + vector3<float>(size.x, 0, 0), min_p + vector3<float>(size.x, 0, size.z), min_p + vector3<float>(size.x, size.y, 0));
	planes[3].define(min_p, min_p + vector3<float>(size.x, 0, 0), min_p + vector3<float>(0, size.y, 0));
	planes[4].define(min_p + vector3<float>(0, 0, size.z), min_p + vector3<float>(0, size.y, size.z), min_p + vector3<float>(size.x, 0, size.z));
	planes[5].define(min_p + vector3<float>(0, size.y, 0), min_p + vector3<float>(size.x, size.y, 0), min_p + vector3<float>(0, size.y, size.z));

	//save_shape_data();
	return true;
}

unsigned int cube::makeParticles(float rad, float _spacing, bool isOnlyCount, VEC4F_PTR pos, unsigned int sid)
{
	vector3<unsigned int> dim3np(
		static_cast<unsigned int>(abs(size.x / (rad * 2)))
		, static_cast<unsigned int>(abs(size.y / (rad * 2)))
		, static_cast<unsigned int>(abs(size.z / (rad * 2))));

	_spacing = rad * 0.1f;
	float spacing = rad * 2.f + _spacing;
	srand(1976);
	float jitter = rad * 0.001f;
	for (unsigned int z = dim3np.z; z > 0; z--){
		float _z = (min_p.z + rad + z * spacing) + frand()*jitter;
		if (_z < max_p.z){
			dim3np.z = z;
			break;
		}
	}
	for (unsigned int y = dim3np.y; y > 0; y--){
		float _y = (min_p.y + rad + y * spacing) + frand()*jitter;
		if (_y < max_p.y){
			dim3np.y = y;
			break;
		}
	}
	for (unsigned int x = dim3np.x; x > 0; x--){
		float _x = (min_p.x + rad + x * spacing) + frand()*jitter;
		if (_x < max_p.x){
			dim3np.x = x;
			break;
		}
	}

	unsigned int np = dim3np.x * dim3np.y * dim3np.z;
	if (!np)
		return 0;
	if (isOnlyCount)
		return np;
	
	
	unsigned int cnt = sid;

	

	for (unsigned int z = 0; z < dim3np.z; z++){
		for (unsigned int y = 0; y < dim3np.y; y++){
			for (unsigned int x = 0; x < dim3np.x; x++){
				float p[3] = { (min_p.x + rad + x * spacing) + frand()*jitter, (min_p.y + rad + y * spacing) + frand()*jitter, (min_p.z + rad + z * spacing) + frand()*jitter };
// 				if ((p[0] + rad) >= min_p.x + size.x || (p[1] + rad) >= min_p.y + size.y || (p[2] + rad) >= min_p.z + size.z)
// 					continue;

				pos[cnt].x = p[0];
				pos[cnt].y = p[1];
				pos[cnt].z = p[2];
				pos[cnt].w = rad;
				cnt++;
			}
		}
	}
	//pos[0].z = 0.f;
	return np;
}

void cube::save_shape_data(QTextStream& ts) const
{
	//QTextStream ts(&(md->modelStream()));
	bool isExistMass = ms ? true : false;
	ts << "OBJECT CUBE " << id << " " << name << " " << roll_type << " " << mat_type << " " << (int)_expression << " " << isExistMass << endl
		<< min_p.x << " " << min_p.y << " " << min_p.z << endl
		<< max_p.x << " " << max_p.y << " " << max_p.z << endl;

	if (isExistMass)
	{
		save_mass_data(ts);
	}
}
// std::fstream& cube::operator<<(std::fstream& oss)
//{
//
//}