#ifndef PARSIM_XMLLOADER_H
#define PARSIM_XMLLOADER_H

#include <map>

#include "loader.h"
#include "extern/libxml/parser.h"
#include "extern/libxml/tree.h"

namespace parSIM {

	class XmlLoader : public Loader
	{
	public:
		XmlLoader();

		virtual ~XmlLoader();

		virtual Simulation* Read();

	protected:
		std::map<std::string, xmlNode*> getXmlNodeList(xmlNode*,const xmlChar*);
		std::string getStrValue(std::map<std::string, xmlNode*>& nodes, std::string str);
		int getIntValue(std::map<std::string, xmlNode*>& nodes, std::string str);
		int getIntValue(xmlNode* nodes, std::string str);
		double getDoubleValue(std::map<std::string, xmlNode*>& nodes, std::string str);
		double getDoubleValue(xmlNode* node, std::string str, std::string str2 = "");
		dependency_type<double> getDoubleDependency(std::map<std::string, xmlNode*>& nodes, std::string str);
		dependency_type<double> getDoubleDependency(xmlNode*, std::string str);
		void getParticleElements(std::map<std::string, xmlNode*>& nodes, particles* ps, std::string str);
		void getBoundaryElements(std::map<std::string, xmlNode*>& nodes, Simulation* sim, std::string str);
		void getShpaeElements(std::map<std::string, xmlNode*>& nodes, Simulation* sim, std::string str);
		void getPointMassElements(std::map<std::string, xmlNode*>& nodes, Simulation* sim, std::string str);
		vector3d getVector3dValue(std::map<std::string, xmlNode*>& nodes, std::string str);
		vector3d getVector3dValue(xmlNode* node, std::string str);
		vector4d getVector4dValue(xmlNode* node, std::string str);
	};
}

#endif