#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "algebra.h"
#include "types.h"
#include <map>
#include "Simulation.h"

struct device_plane_info;
struct cu_polygon;
struct cu_mass_info;
struct double3;
struct uint3;

namespace parSIM
{
	//class Simulation;
	using namespace algebra;

	class geometry
	{
	public:
		geometry() {}
		geometry(Simulation *_sim, std::string& _name);
		virtual ~geometry();
		
		void setName(std::string str) { name = str; }

		Simulation *get_sim() const { return sim; }
		std::string get_name() const { return name; }

		geometry_type& Geometry() { return geo_type; }
		geometry_use& GeometryUse() { return geo_use; }
		material_type& Material() { return mat_type; }
		vector3<double>& Position() { return position; }
		void bindPointMass(pointmass* _pm) { pm = _pm; }
		//vector4<double>& Orientation() { return eparameter; }
		void setMaterial();
// 		void savePosition();
// 		void saveOrientation();
		cmaterialType& getMaterialParameters() { return material; }
		bool IsDefined() { return isDefined; }
		virtual bool save2file(std::fstream& of) = 0;
		virtual bool setSpecificDataFromFile(std::fstream& pf) = 0;
		virtual bool define_geometry() = 0;
		virtual bool define_device_info() = 0;
		virtual void cu_update_geometry(double* A, double3* pos) = 0;
		virtual void cu_hertzian_contact_force(contact_coefficient& coe, bool* isLineContact, double* pos, double* vel, double* omegai, double* force, double* moment, unsigned int np, unsigned int *u1 = 0, unsigned int* u2 = 0, unsigned int* u3 = 0) = 0;

	protected:
		
		std::string name;
		bool isDefined;
		bool isParticleGeometry;
		geometry_type geo_type;
		geometry_use geo_use;
		material_type mat_type;
		vector3<double> position;
		cmaterialType material;
		pointmass* pm;
		Simulation *sim;
	}; 

	namespace geo
	{
		class plane : public geometry
		{
		public:
			plane() {}
			plane(Simulation* _sim, std::string _name);
			plane(const plane& _plane);
			virtual ~plane();

			void insert2simulation();
			virtual bool save2file(std::fstream& of);
			virtual bool setSpecificDataFromFile(std::fstream& pf);
			virtual bool define_geometry();
			virtual bool define_device_info();
			virtual void cu_update_geometry(double* A, double3* pos){};
			virtual void cu_hertzian_contact_force(contact_coefficient& coe, bool* isLineContact, double* pos, double* vel, double* omegai, double* force, double* moment, unsigned int np, unsigned int* u1 = 0, unsigned int* u2 = 0, unsigned int* u3 = 0);

			vector2<double> size;
			vector3<double> u1;
			vector3<double> u2;
			vector3<double> uw;
			vector3<double> xw;
			vector3<double> pa;
			vector3<double> pb;
			double l1, l2;
		};

		class cube : public geometry
		{

		public:
			cube() {}
			cube(Simulation* _sim, std::string _name);
			cube(const cube& _cube);
			virtual ~cube();

			geometry_type& geo_type() { return geometry::geo_type; }
			vector3<double>& cube_size() { return size; }
			void insert2simulation();
			void define(vector3<double> _size, vector3<double> _pos, material_type mtype, geometry_use guse, geometry_type gtype = CUBE);
			virtual bool define_geometry();
			virtual bool setSpecificDataFromFile(std::fstream& pf);
			virtual bool save2file(std::fstream& of);
			virtual bool define_device_info();
			virtual void cu_update_geometry(double* A, double3* pos){};
			void hertzian_contact_force(double r, double dt, contact_coefficient& coe, vector3<double>& pos, vector3<double>& vel, vector3<double>& omega, vector3<double>& nforce, vector3<double>& sforce);
			virtual void cu_hertzian_contact_force(contact_coefficient& coe, bool* isLineContact, double* pos, double* vel, double* omegai, double* force, double* moment, unsigned int np, unsigned int* u1 = 0, unsigned int* u2 = 0, unsigned int* u3 = 0);

		public:
			vector3<double> size;
			std::map<std::string, plane*> planes;
			device_plane_info *d_planes;
		};

		class shape : public geometry
		{
			typedef struct
			{
				vector3<double> P;
				vector3<double> Q;
				vector3<double> R;
				vector3<double> V;
				vector3<double> W;
				vector3<double> N;
			}polygon;			

			enum with_contact  
			{
				LINE,
				VERTEX,
				PLANE
			};

			typedef struct  
			{
				with_contact contact_with;
				double penetration;
				vector3<double> unit;
				vector3<double> contact_point;
			}contact_info;

			typedef struct  
			{
				double radius;
				vector3<double> center;
			}sphere;

			typedef struct  
			{
				vector3<double> point;
				vector3<double> force;
			}shape_force;

		public:
			shape() {}
			shape(Simulation* _sim, std::string _name);
			shape(const shape& _shape);
			virtual ~shape();

			void Rotation();
			geometry_type& geo_type() { return geometry::geo_type; }
			std::string& filePath() { return file_path; }
			unsigned int getNp() { return vertice.sizes(); }
			void update_polygons();
			//void setOrientation(vector3<double>& o) { ep = o; }
			void insert2simulation();
			algebra::vector<vector3<double>>& getVertice() { return vertice; }
			double3* getCuVertice() { return d_vertice; }
			void define(std::string fpath, vector3<double> position, material_type mtype, geometry_use guse, geometry_type gtype = SHAPE);
			virtual bool define_geometry();
			virtual bool setSpecificDataFromFile(std::fstream& pf);
			virtual bool save2file(std::fstream& of);
			virtual bool define_device_info();
			virtual void cu_hertzian_contact_force(
				contact_coefficient& coe, 
				bool* isLineContact,
				double* pos, 
				double* vel, 
				double* omegai, 
				double* force, 
				double* moment, 
				unsigned int np,
				unsigned int *sorted_id,
				unsigned int *cell_start,
				unsigned int *cell_end);

			static int get_id() { return --id; }

			bool hertzian_contact_force(
				unsigned int id,
				double r, 
				contact_coefficient& coe,
				vector3<double>& pos, 
				vector3<double>& vel,
				vector3<double>& omega,
				vector3<double>& force,
				vector3<double>& moment,
				vector3<double>& line_force,
				vector3<double>& line_moment);

			virtual void cu_update_geometry(double* A, double3* pos);

		public:
			static int id;
			bool isLineContact;
			bool isUpdate;
			int contact_count;
			std::string file_path;
			//vector4<double>* spos;

			vector3<double> line_contact_force;
	
			vector3<double> plane_body_force;
			vector3<double> line_body_force;
			vector3<double> body_force;
			vector4<double> body_moment;
			algebra::vector<vector3<double>> vertice;
			algebra::vector<vector3<double>> l_vertice;
			algebra::vector<vector3<unsigned int>> indice;
			algebra::vector<polygon> polygons;

			unsigned int *id_set;
			unsigned int *poly_start;
			unsigned int *poly_end;
			
			cu_polygon* d_polygons;
			double3 *d_vertice;
			double3 *d_local_vertice;
			double3 *d_body_force;
			uint3 *d_indice;
			unsigned int *d_id_set;
			unsigned int *d_poly_start;
			unsigned int *d_poly_end;
			//bool d_contact_part;


			cu_mass_info *d_mass_info;

		private:
			vector3<double> ClosestPtPointTriangle(vector3<double>& p, vector3<double>& a, vector3<double>& b, vector3<double>& c, with_contact *wc);
		};
	}
}



#endif