#ifndef PARTICLES_H
#define PARTICLES_H

#include "algebra.h"
#include "types.h"

namespace parSIM
{
	class Simulation; 

	class particles
	{
	public:
		//particles();
		particles(Simulation *_sim);
		~particles();

		void setColor(color_type clr) { color = clr; }
		void setName(std::string str) { name = str; }
		void setMaterials(material_type mtype);
		void setArrangeShape(std::string& str) { arrange_shape = str; }
		void setArrangeSize(vector3d& v3) { arrange_size = v3; }
		void setArrangePosition(vector3d& v3) { arrange_position = v3; }

		int getMaterial() { return mat_type; }
		std::string Name() { return name; }
		cmaterialType& getMaterialParameters() { return material; }

		void CreateParticles(geometry_type g_type);
		void CreateParticlesByCube();
		void define_device_info();
		void saveResultToBuffers();
		void setSpecificDataFromFile(std::fstream& pf);

		algebra::vector4<double>* Position() { return pos; }
		algebra::vector4<double>* Velocity() { return vel; }
		algebra::vector4<double>* Acceleration() { return acc; }
		algebra::vector4<double>* AngularVelocity() { return omega; }
		algebra::vector4<double>* AngularAcceleration() { return alpha; }

		bool* cu_IsLineContact() { return d_isLineContact; }
		double* cu_Position() 
		{ 
			return d_pos; 
		}
		double* cu_Velocity() { return d_vel; }
		double* cu_Acceleration() { return d_acc; }
		double* cu_AngularVelocity() { return d_omega; }
		double* cu_AngularAcceleration() { return d_alpha; }
		unsigned int Size() { return np; }
		void resize_pos(unsigned int tnp);
		vector4<double>* add_pos_v3data(vector3<double>* v3, double w, unsigned int an);

		bool initialize();
	//	void rearrangement(cell_grid* detector);
		double calMaxRadius();

		static double radius;
		static double mass;
		static double inertia;
		
	private:
		std::string name;
		color_type color;
		unsigned int saveCount;
		unsigned int precision_size;
		
		material_type mat_type;
		cmaterialType material;

		std::string arrange_shape;
		vector3d arrange_size;
		vector3d arrange_position;
		algebra::vector3<unsigned int> dim3np;

		algebra::vector4<double>* pos;
		algebra::vector4<double>* vel;
		algebra::vector4<double>* acc;
		algebra::vector4<double>* omega;
		algebra::vector4<double>* alpha;

		bool* d_isLineContact;
		double* d_pos;
		double* d_vel;
		double* d_acc;
		double* d_omega;
		double* d_alpha;

		unsigned int np;
		unsigned int added_np;

		Simulation *sim;
		geometry *Geo;
		
// 		void AllocMemory();
// 		void Arrangement();
	};
}

#endif