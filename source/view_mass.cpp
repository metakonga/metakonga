// #include "view_mass.h"
// #include "view_controller.h"
// 
// using namespace parview;
// 
// mass::mass()
// {
// 
// }
// 
// mass::~mass()
// {
// 
// }
// 
// void mass::setMassData(QFile& pf)
// {
// 	vector3<double> v3;
// 	pf.read((char*)&v3, sizeof(vector3<double>));
// 	pos[view_controller::getTotalBuffers()] = vector3<float>(
// 		static_cast<float>(v3.x),
// 		static_cast<float>(v3.y),
// 		static_cast<float>(v3.z)
// 		);
// 	pf.read((char*)&v3, sizeof(vector3<double>));
// 	vel[view_controller::getTotalBuffers()] = vector3<float>(
// 		static_cast<float>(v3.x),
// 		static_cast<float>(v3.y),
// 		static_cast<float>(v3.z)
// 		);
// 	pf.read((char*)&v3, sizeof(vector3<double>));
// 	force[view_controller::getFrame()] = vector3<float>(
// 		static_cast<float>(v3.x),
// 		static_cast<float>(v3.y),
// 		static_cast<float>(v3.z)
// 		);
// }
// 
// void mass::draw(GLenum eMode)
// {
// 
// }
// 
// void mass::define(void* tg)
// {
// 
// }
// 
// void mass::saveCurrentData(QFile& pf)
// {
// 	
// }
// 
// void mass::updateDataFromFile(QFile& pf, unsigned int fdtype)
// {
// 	vector3<double> v3;
// 	pf.read((char*)&v3, sizeof(vector3<double>));
// 	pos[0] = vector3<float>(
// 		static_cast<float>(v3.x),
// 		static_cast<float>(v3.y),
// 		static_cast<float>(v3.z)
// 		);
// 	pf.read((char*)&v3, sizeof(vector3<double>));
// 	vel[0] = vector3<float>(
// 		static_cast<float>(v3.x),
// 		static_cast<float>(v3.y),
// 		static_cast<float>(v3.z)
// 		);
// 	pf.read((char*)&v3, sizeof(vector3<double>));
// 	force[view_controller::getFrame()] = vector3<float>(
// 		static_cast<float>(v3.x),
// 		static_cast<float>(v3.y),
// 		static_cast<float>(v3.z)
// 		);
// }