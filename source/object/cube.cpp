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

bool cube::define(VEC3D& min, VEC3D& max)
{
	min_p = min;
	max_p = max;

	size.x = (max_p - VEC3D(min_p.x, max_p.y, max_p.z)).length();
	size.y = (max_p - VEC3D(max_p.x, min_p.y, max_p.z)).length();
	size.z = (max_p - VEC3D(max_p.x, max_p.y, min_p.z)).length();

	planes[0].define(min_p, min_p + VEC3D(0, 0, size.z), min_p + VEC3D(size.x, 0, 0));
	planes[1].define(min_p, min_p + VEC3D(0, size.y, 0), min_p + VEC3D(0, 0, size.z));
	planes[2].define(min_p + VEC3D(size.x, 0, 0), min_p + VEC3D(size.x, 0, size.z), min_p + VEC3D(size.x, size.y, 0));
	planes[3].define(min_p, min_p + VEC3D(size.x, 0, 0), min_p + VEC3D(0, size.y, 0));
	planes[4].define(min_p + VEC3D(0, 0, size.z), min_p + VEC3D(0, size.y, size.z), min_p + VEC3D(size.x, 0, size.z));
	planes[5].define(min_p + VEC3D(0, size.y, 0), min_p + VEC3D(size.x, size.y, 0), min_p + VEC3D(0, size.y, size.z));

	//save_shape_data();
	return true;
}

unsigned int cube::makeParticles(double rad, VEC3UI &_size, VEC3D &_spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos, unsigned int sid)
{
	unsigned int np = 0;
	if (isOnlyCount){
		vector3<unsigned int> dim3np(
			static_cast<unsigned int>(abs(size.x / (rad * 2)))
			, static_cast<unsigned int>(abs(size.y / (rad * 2)))
			, static_cast<unsigned int>(abs(size.z / (rad * 2))));

		//VEC3F space;
		double dia = rad * 2.0;
		VEC3D space = rad * 0.1;
		double x_len = dim3np.x * dia + (dim3np.x + 1) * space.x;
		double y_len = dim3np.y * dia + (dim3np.y + 1) * space.y;
		double z_len = dim3np.z * dia + (dim3np.z + 1) * space.z;
		if (x_len > size.x){
			dim3np.x--;
			space.x = (size.x - dim3np.x * dia) / (dim3np.x + 1);
			//x_len = dim3np.x * dia + (dim3np.x + 1) * space.x;
		}
		if (y_len > size.y){
			dim3np.y--;
			space.y = (size.y - dim3np.y * dia) / (dim3np.y + 1);
		}
		if (z_len > size.z){
			dim3np.z--;
			space.z = (size.z - dim3np.z * dia) / (dim3np.z + 1);
		}
		//float spacing = rad * 2.f + _spacing;
		np = dim3np.x * dim3np.y * dim3np.z;
		_spacing = space;
		_size = dim3np;
	}
	else{
		srand(1976);
		double jitter = rad * 0.001;
		unsigned int cnt = 0;
		for (unsigned int i = 0; i <= nstack; i++){
			for (unsigned int z = 0; z < _size.z; z++){
				for (unsigned int y = 0; y < _size.y; y++){
					for (unsigned int x = 0; x < _size.x; x++){
						double p[3] = {
							(min_p.x + rad * (2.0 * x + 1) + (x + 1) * _spacing.x) + frand()*jitter,
							(min_p.y + rad * (2.0 * y + 1) + (y + 1) * _spacing.y) + frand()*jitter,
							(min_p.z + rad * (2.0 * z + 1) + (z + 1) * _spacing.z) + frand()*jitter };

						pos[cnt].x = p[0];
						pos[cnt].y = p[1];
						pos[cnt].z = p[2];
						pos[cnt].w = rad;
						cnt++;
					}
				}
			}
		}
// 		pos[0].x = 0.0; pos[0].y = 0.006; pos[0].z = 0.0;
// 		pos[1].x = 0.005; pos[1].y = 0.016; pos[1].z = 0.0;
	}
	
// 	srand(1976);
// 	float jitter = rad * 0.001f;
// 
// 	
// 
// 	for (unsigned int z = 0; z < dim3np.z; z++){
// 		for (unsigned int y = 0; y < dim3np.y; y++){
// 			for (unsigned int x = 0; x < dim3np.x; x++){
// 				float p[3] = { (min_p.x + rad + x * spacing) + frand()*jitter, (min_p.y + rad + y * spacing) + frand()*jitter, (min_p.z + rad + z * spacing) + frand()*jitter };
// // 				if ((p[0] + rad) >= min_p.x + size.x || (p[1] + rad) >= min_p.y + size.y || (p[2] + rad) >= min_p.z + size.z)
// // 					continue;
// 
// 				pos[cnt].x = p[0];
// 				pos[cnt].y = p[1];
// 				pos[cnt].z = p[2];
// 				pos[cnt].w = rad;
// 				cnt++;
// 			}
// 		}
// 	}
// 	//pos[0].z = 0.f;
	return np;
}

void cube::save_object_data(QTextStream& ts)
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