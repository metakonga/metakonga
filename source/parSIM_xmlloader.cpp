#include "parSIM.h"
#include <clocale>
#include <cctype>
#include <algorithm>

using namespace parSIM;

XmlLoader::XmlLoader()
{

}

XmlLoader::~XmlLoader()
{

}

Simulation* XmlLoader::Read()
{

	LIBXML_TEST_VERSION

	const char *document = "<doc/>";
	xmlDoc *doc;
	xmlNode *root_element;

	doc = xmlReadFile(path.c_str(), "NULL", 0);//xmlReadMemory(document, 6, path.c_str(), NULL, 0);

	if(doc == NULL){
		Log::Send(Log::Error, "Failed to parse document");
		return NULL;
	}

 	root_element = xmlDocGetRootElement(doc);
 
 	std::map<std::string, xmlNode*> sim_xmlNodes = getXmlNodeList(root_element, xmlCharStrdup("simulation"));
 
 	Simulation* sim = NULL;
	Simulation* subSim = NULL;

 	std::string str = getStrValue(sim_xmlNodes, "solver");
 	if(str=="dem"){
 		Demsimulation* dem = new Demsimulation;
 		sim = dem;
	}


	sim->setCaseName(base_path, case_name);

 	sim->setName(getStrValue(sim_xmlNodes, "name"));
 	str = getStrValue(sim_xmlNodes, "device");
 
 	str == "cpu" ? sim->setDevice(CPU) : sim->setDevice(GPU);
 
 	getIntValue(sim_xmlNodes, "dimension") == 2 ? sim->setDimension(DIM_2) : sim->setDimension(DIM_3);
 
 	getStrValue(sim_xmlNodes, "precision") == "single" ? sim->setPrecision(SINGLE_PRECISION) : sim->setPrecision(DOUBLE_PRECISION);

	str = getStrValue(sim_xmlNodes, "platform");

	sim->setPlatform(str);
 
 	dependency_type<double> time_step = getDoubleDependency(sim_xmlNodes, "time_step");
 
 	double export_time_step = getDoubleValue(sim_xmlNodes, "export_time_step");

	double simulation_time = getDoubleValue(sim_xmlNodes, "simulation_time");
 
 	getStrValue(sim_xmlNodes, "integrator") == "euler" ? sim->setIntegrator(EULER_METHOD) : sim->setIntegrator(VELOCITY_VERLET);

	str = getStrValue(sim_xmlNodes, "solver");
	if(str=="dem") sim->setSolver(DEM);
	else if(str=="sph")	sim->setSolver(SPH);
 
	sim->setDt(time_step.value, time_step.dependency);
	sim->setSimulationTime(simulation_time);
	sim->setResultExportTimeStep(export_time_step);
 	particles *ps = sim->getParticles();
 	//std::list<geometry*> *geometries = sim->getGeometries();
 
 	getParticleElements(sim_xmlNodes, ps, "particle");
  	getBoundaryElements(sim_xmlNodes, sim, "boundary");
 	getShpaeElements(sim_xmlNodes, sim, "shape");

	str = getStrValue(sim_xmlNodes, "subsolver");
	if(str=="dem"){

	}
	else if(str=="mbd"){
		Mbdsimulation* mbd = new Mbdsimulation(sim);
		subSim = mbd;
		subSim->setSolver(MBD);
		str = getStrValue(sim_xmlNodes, "subintegrator");
		if(str=="euler"){
			subSim->setIntegrator(EULER_METHOD);
		}
		else if(str=="HHT"){
			subSim->setIntegrator(IMPLICIT_HHT);
		}
		getPointMassElements(sim_xmlNodes, subSim, "point_mass");
	}
	else if(str=="sph"){

	}

	str = getStrValue(sim_xmlNodes, "specific_data");

	if(str != "") 
		sim->setSpecificDataFileName(str);
		
	return sim;
}

void XmlLoader::getPointMassElements(std::map<std::string, xmlNode*>& nodes, Simulation* sim, std::string str)
{
	std::string name;
	std::string val;
	xmlNode *n;
	for(std::map<std::string, xmlNode*>::iterator node = nodes.begin(); node != nodes.end(); node++){
		n = node->second;
		if(!xmlStrcmp(n->name, (xmlChar*)"object")){
			val = (char*)n->properties->children->content;
			if(val == "rigid_body"){
				mass::rigid_body *rb = new mass::rigid_body(sim, RIGID_BODY);
				for(n = n->children; n; n = n->next){
					if(!xmlStrcmp(n->name, (xmlChar*)"shape_name")){
						val = (char*)n->properties->children->content;
						rb->setGeometry(sim->getGeometry(val));
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"name")){
						val = (char*)n->properties->children->content;
						rb->setName(val);
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"mass")){
						rb->setMass(getDoubleValue(n, (char*)n->name));
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"inertia")){
						rb->setPrincipalInertia(getVector3dValue(n, "inertia"));
						rb->setSymetryInertia(getVector3dValue(n, "sym_inertia"));
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"position")){
						rb->setPosition(getVector3dValue(n, "position"));
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"orientation")){
						rb->setOrientation(getVector4dValue(n, "orientation"));
					}
				}
				rb->define_mass();
			}
		}
	}
}

void XmlLoader::getShpaeElements(std::map<std::string, xmlNode*>& nodes, Simulation* sim, std::string str)
{
	std::string name;
	std::string val;
	xmlNode *n;
	for(std::map<std::string, xmlNode*>::iterator node = nodes.begin(); node != nodes.end(); node++){
		n = node->second;
		if(!xmlStrcmp(n->name, (xmlChar*)"object")){
			val = (char*)n->properties->children->content;
			if(val == "shape"){
				geo::shape *s = new geo::shape(sim, SHAPE);
				for(n = n->children; n; n = n->next){
					if(!xmlStrcmp(n->name, (xmlChar*)"name")){
						s->setName((char*)n->properties->children->content);
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"material")){
						s->Material() = material_str2enum((char*)n->properties->children->content);
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"file")){
						s->filePath() = (char*)n->properties->children->content;
					}
					if(!xmlStrcmp(n->name, (xmlChar*)"position")){
						s->Position() = getVector3dValue(n, (char*)n->name);
					}
				}
				s->insert2simulation();
				break;
			}
		}
	}
}

void XmlLoader::getBoundaryElements(std::map<std::string, xmlNode*>& nodes, Simulation *sim, std::string str)
{
	std::string name;
	std::string val;
	xmlNode* n;
	for(std::map<std::string, xmlNode*>::iterator node = nodes.begin(); node != nodes.end(); node++){
		n = node->second;
		if(!xmlStrcmp(n->name, (xmlChar*)"object")){
			val = (char*)n->properties->children->content;
			if(val == "boundary"){
				for(n = n->children; n; n = n->next){
					name = (char*)n->name;
					if(name == "geometry"){
						val = (char*)n->properties->children->content;
						if(val == "cube"){
							geo::cube *c = new geo::cube(sim, CUBE);
							for(n = node->second->children; n; n = n->next){
								name = (char*)n->name;
								if(name == "name"){
									c->setName(name);
								}
								if(name == "material"){
									c->Material() = material_str2enum((char*)n->properties->children->content);
								}
								if(name == "geometry_size"){
									c->cube_size() = getVector3dValue(n, name);
								}
								if(name == "position"){
									c->Position() = getVector3dValue(n, name);
								}
							}
							c->insert2simulation();
							break;
						}
						else if(val == "rectangle"){
							geo::rectangle *r = new geo::rectangle(sim, RECTANGLE);
							for(n = node->second->children; n; n = n->next){
								name = (char*)n->name;
								if(name == "name"){
									val = (char*)n->properties->children->content;
									r->setName(val);
								}
								if(name == "material"){
									r->Material() = material_str2enum((char*)n->properties->children->content);
								}
								if(name == "geometry_size"){
									r->rectangle_size() = getVector3dValue(n, name);
								}
								if(name == "position"){
									r->Position() = getVector3dValue(n, name);
								}
							}
							r->insert2simulation();
							break;
						}
					}
				}
			}
		}
	}
}

void XmlLoader::getParticleElements(std::map<std::string, xmlNode*>& nodes, particles* ps, std::string str)
{
	std::string name;
	std::string val;
	xmlNode* n;
	//std::map<std::string, xmlNode*>::iterator node = nodes.find(str);
	for(std::map<std::string, xmlNode*>::iterator node = nodes.begin(); node != nodes.end(); node++){
		n = node->second;
		xmlChar *sub_name = xmlStrsub(n->name, 0, 6);
		if(!xmlStrcmp(sub_name, (xmlChar*)"object")){
			val = (char*)n->properties->children->content;
			if(val == "particle")
				break;
		}
	}
	
	for(n = n->children; n; n = n->next){
		name = (char*)n->name;
		if(name == "name"){
			val = (char*)n->properties->children->content;
			ps->setName(val);
		}
		if(name == "material"){
			val = (char*)n->properties->children->content;
			ps->setMaterials(val);
		}
		else if(name == "radius"){
			ps->setRadius(getDoubleValue(nodes, name));
		}
		else if(name == "mass"){
			dependency_type<double> mass = getDoubleDependency(n, name);
			ps->setMass(mass.value, mass.dependency);
		}
		else if(name == "inertia"){
			dependency_type<double> inertia = getDoubleDependency(n, name);
			ps->setInertia(inertia.value, inertia.dependency);
		}
		else if(name == "arrange_shape"){
			val = (char*)n->properties->children->content;
			ps->setArrangeShape(val);
		}
		else if(name == "arrange_size"){
			vector3d dim = getVector3dValue(n, name);
			ps->setArrangeSize(dim);
		}
		else if(name == "arrange_position"){
			vector3d pos = getVector3dValue(n, name);
			ps->setArrangePosition(pos);
		}
// 		if(n->properties){
// 			val = (char*)n->properties->children->content;
// 		}
	}

}

vector3d XmlLoader::getVector3dValue(xmlNode* node, std::string str)
{
	vector3d dim;
	xmlNode *n = NULL;
	for(n = node; n; n = n->next){
		if(!xmlStrcmp(n->name, (xmlChar*)str.c_str())){
			n = n->properties->children;
			break;
		}
	}
// 	if(n == node->second){
// 		Log::Send(Log::Warning, "double XmlLoader::getDoubleValue(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
// 		return vector3d(0.0, 0.0, 0.0);
// 	}

	dim.z = atof((char*)n->content); n = n->parent->next->children;
	dim.y = atof((char*)n->content); n = n->parent->next->children;
	dim.x = atof((char*)n->content);

	return dim;
}

vector4d XmlLoader::getVector4dValue(xmlNode* node, std::string str)
{
	vector4d val;
	xmlNode* n = NULL;
	if(node->properties){
		n = node->properties->children;
	}

	// 	if(n == node->second){
	// 		Log::Send(Log::Warning, "double XmlLoader::getDoubleValue(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
	// 		return vector3d(0.0, 0.0, 0.0);
	// 	}

	val.w = atof((char*)n->content); n = n->parent->next->children;
	val.z = atof((char*)n->content); n = n->parent->next->children;
	val.y = atof((char*)n->content); n = n->parent->next->children;
	val.x = atof((char*)n->content); 

	return val;
}

vector3d XmlLoader::getVector3dValue(std::map<std::string, xmlNode*>& nodes, std::string str)
{
	vector3d dim;
	std::map<std::string, xmlNode*>::iterator node = nodes.find(str);
	xmlNode* n = node->second;
	if(n->properties){
		n = n->properties->children;
	}

	if(n == node->second){
		Log::Send(Log::Warning, "double XmlLoader::getDoubleValue(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
		return vector3d(0.0, 0.0, 0.0);
	}

	dim.z = atof((char*)n->content); n = n->parent->next->children;
	dim.y = atof((char*)n->content); n = n->parent->next->children;
	dim.x = atof((char*)n->content);

	return dim;
}

double XmlLoader::getDoubleValue(std::map<std::string, xmlNode*>& nodes, std::string str)
{
	std::map<std::string, xmlNode*>::iterator node = nodes.find(str);
	xmlNode* n = node->second;
	if(n->properties){
		n = n->properties->children;
	}

	if(n == node->second){
		Log::Send(Log::Warning, "double XmlLoader::getDoubleValue(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
		return 0;
	}

	double return_value = atof((char*)n->content);
	return return_value;
}

double XmlLoader::getDoubleValue(xmlNode* node, std::string str, std::string str2)
{
	double return_value = 0.0;
	if(str2 != ""){
		_xmlAttr *att;
		if(node->properties){
			for(att = node->properties; att; att = att->next){
				if(!xmlStrcmp(att->name, (xmlChar*)str2.c_str()))
				{
					return_value = atof((char*)att->children->content);
					break;
				}
			}		
		}
		return return_value;
	}
	xmlNode* n;
	if(node->properties){
		n = node->properties->children;
		return_value = atof((char*)n->content);
	}

	return return_value;
}

dependency_type<double> XmlLoader::getDoubleDependency(xmlNode* node, std::string str)
{
	dependency_type<double> return_value = {false, 0.0};
	
	if(node->properties){
		node = node->properties->children;
		return_value.dependency = !xmlStrcmp(node->content, (xmlChar*)"yes") ? true : false;
	}
	node = node->parent->next->children;
	if(node){
		return_value.value = atof((char*)node->content);
	}
	return return_value;
}

dependency_type<double> XmlLoader::getDoubleDependency(std::map<std::string, xmlNode*>& nodes, std::string str)
{
	dependency_type<double> return_value = {false, 0.0};
	std::map<std::string, xmlNode*>::iterator node = nodes.find(str);
	xmlNode* n = node->second;
	if(n->properties){
		n = n->properties->children;
	}

	if(n == node->second){
		Log::Send(Log::Warning, "dependency_type<double> XmlLoader::getDoubleDependency(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
		return return_value;
	}

	return_value.dependency = !xmlStrcmp(n->content, (xmlChar*)"yes") ? true : false;
	n = n->parent->next->children;
	return_value.value = atof((char*)n->content);
	
	
	return return_value;
}

int XmlLoader::getIntValue(std::map<std::string, xmlNode*>& nodes, std::string str)
{
	std::map<std::string, xmlNode*>::iterator node = nodes.find(str);
	xmlNode* n = node->second;
	if(n->properties){
		n = n->properties->children;
	}

	if(n == node->second){
		Log::Send(Log::Warning, "int XmlLoader::getIntValue(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
		return 0;
	}

	int return_value = atoi((char*)n->content);
	return return_value;
}

int XmlLoader::getIntValue(xmlNode* node, std::string str)
{
	int return_value;

	if(node->properties){
		node = node->properties->children;
		return_value = atoi((char*)node->content);
	}

	return return_value;
}

std::string XmlLoader::getStrValue(std::map<std::string, xmlNode*>& nodes, std::string str)
{
	std::map<std::string, xmlNode*>::iterator node = nodes.find(str);
	if(node == nodes.end()){
		return "";
	}
	xmlNode* n = node->second;
	if(n->properties){
		n = n->properties->children;
	}

	if(n == node->second){
		Log::Send(Log::Warning, "std::string XmlLoader::getStrValue(std::map<std::string, xmlNode*>&, std::string \"" + str + "\"" + "-->> No exist the children node.");
		return "";
	}
	std::string return_value = (char*)n->content;
	return return_value;
}

std::map<std::string, xmlNode*> XmlLoader::getXmlNodeList(xmlNode* root, const xmlChar* xchar)
{
	int nObject = 0;
	xmlNode* node;
	std::map<std::string, xmlNode*> xmlNodes;
	if(!xmlStrcmp(root->name, xchar)){
		if(root->properties)
		{

		}
		else
		{
			for(node = root->children; node; node = node->next){
				std::string name = (char*)node->name;
				if(name == "object"){
					if(nObject){
						char str_n[10] = {0, };
						sprintf_s(str_n, 10, "%d", nObject);
						name = name + str_n;
					}
					nObject++;
				}
				if(!node->children)
				{
					
					xmlNodes[name] = node;
				}
				else
				{
					xmlNodes[name] = node;
					xmlNode* cnode = node->children;
					for(node = cnode; node; node = node->next){
						std::string name = (char*)node->name;
						xmlNodes[name] = node;
					}
					node = cnode->parent;
				}
			}
		}
	}
	return xmlNodes;
}