#include "xml/xml_editor.h"

xml_editor::xml_editor()
{
	object_step = 0;
	filename = "C:/C++/kangsia/case/dem/temp_vv.xml";
	sim_opts[NAME] = "temp";
	sim_opts[PROCESSING_UNIT] = "cpu";
	sim_opts[DIMENSION] = "3";
	sim_opts[PRECISION] = "single";
	sim_opts[TIME_STEP] = "0 yes";
	sim_opts[EXPORT_TIME_STEP] = "0.01";
	sim_opts[SIMULATION_TIME] = "1.5";

	solver_opts[SOLVER_TYPE] = "dem";
	solver_opts[INTEGRATOR] = "euler";

	particle_opts[PARTICLE_MATERIAL] = "steel";
	particle_opts[PARTICLE_RADIUS] = "0.01";
	particle_opts[PARTICLE_MASS] = "0 yes";
	particle_opts[PARTICLE_INERTIA] = "0 yes";
	particle_opts[PARTICLE_ARRANGE_SHAPE] = "cube";
	particle_opts[PARTICLE_ARRANGE_SIZE] = "0 0 0";
	particle_opts[PARTICLE_ARRANGE_POSITION] = "0 0 0";

	boundary_opts[BOUNDARY_MATERIAL] = "acrylic";
	boundary_opts[BOUNDARY_GEOMETRY] = "cube";
	boundary_opts[BOUNDARY_GEOMETRY_SIZE] = "0 0 0";
	boundary_opts[BOUNDARY_POSITION] = "0 0 0";

	objects.push_back("particle");
	objects.push_back("boundary");
}

xml_editor::~xml_editor()
{

}

void xml_editor::updateOption(int opt, std::list<std::string>& strList)
{
	std::string str = "";
	unsigned int i = 0;
	for(std::list<std::string>::iterator _str = strList.begin(); i < strList.size(); _str++, i++)
	{
		if(i == strList.size() - 1){
			str += *_str;
		}
		else{
			str += *_str + " ";
		}
	}
	std::map<int, std::string>::iterator it;
	if(opt < 10) 
	{
		sim_opts[opt] = str;
	}
	else if(opt < 20)
	{
		solver_opts[opt] = str;
	}
	else if(opt < 100)
	{
		particle_opts[opt] = str;
	}	
	else if(opt < 1000)
	{
		boundary_opts[opt] = str;
	}
}

std::list<std::string> xml_editor::getInputList(int opt)
{
	std::list<std::string> strList;
	switch(opt){
	case 1: strList.push_back("value"); break;
	case 2: strList.push_back("type"); break;
	case 3: strList.push_back("value"); break;
	case 4: strList.push_back("type"); break;
	case 5: strList.push_back("value"); break;
	case 6: strList.push_back("value"); break;
	case 7: strList.push_back("value"); break;
	case 10: strList.push_back("type"); break;
	case 11: strList.push_back("type"); break;
	case 20: strList.push_back("type"); break;
	case 21: strList.push_back("value"); break;
	case 22: 
		strList.push_back("value");
		strList.push_back("dependency");
		break;
	case 23:
		strList.push_back("value");
		strList.push_back("dependency");
		break;
	case 24: strList.push_back("type"); break;
	case 25:
		if(particle_opts.find(24)->second == "cube"){
			strList.push_back("width");
			strList.push_back("height");
			strList.push_back("depth");
		}
		break;
	case 26:
		strList.push_back("x");
		strList.push_back("y");
		strList.push_back("z");
		break;
	case 100: strList.push_back("type"); break;
	case 101: strList.push_back("type"); break;
	case 102: 
		if(boundary_opts.find(101)->second == "cube"){
			strList.push_back("width");
			strList.push_back("height");
			strList.push_back("depth");
		}
		break;
	case 103:
		strList.push_back("x");
		strList.push_back("y");
		strList.push_back("z");
		break;
	}
	return strList;
}

std::string xml_editor::getValue(int opt)
{
	std::map<int, std::string>::iterator it;
	std::string strVal;
	if(opt < 10) 
	{
		it = sim_opts.find(opt);
		strVal = it->second;
	}
	else if(opt < 20)
	{
		it = solver_opts.find(opt);
		strVal = it->second;
	}
	else if(opt < 100)
	{
		it = particle_opts.find(opt);
		strVal = it->second;
	}	
	else if(opt < 1000)
	{
		it = boundary_opts.find(opt);
		strVal = it->second;
	}
	return strVal;
}

std::string xml_editor::get_object_name(int id)
{
	std::list<std::string>::iterator obj = objects.begin();
	for(int i = 0; i < id; i++)
		obj++;
	std::string return_value = obj->c_str();
	return return_value;
}

void xml_editor::makeXmlFile()
{
	
	xmlTextWriterPtr writer;
	std::string str;
	std::map<int, std::string>::iterator it;
	writer = xmlNewTextWriterFilename(filename.c_str(), 0);
	if(writer == NULL) {
		std::cout << "textXmlwriterfilename : Error creating the xml writer" << std::endl;
	}
	xmlCheckErrors( xmlTextWriterStartDocument(writer, NULL, MY_ENCODING, NULL) );
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "simulation") );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "name") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST sim_opts.find(NAME)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "device") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST sim_opts.find(PROCESSING_UNIT)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "dimension") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST sim_opts.find(DIMENSION)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "precision") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST sim_opts.find(PRECISION)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	it = sim_opts.find(TIME_STEP);
	size_t pos = it->second.find(" ");
	str = it->second.substr(pos+1);
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "time_step") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "dependency", BAD_CAST str.c_str()) );
	str = it->second.substr(0, pos);
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST str.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "export_time_step") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST sim_opts.find(EXPORT_TIME_STEP)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "simulation_time") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST sim_opts.find(SIMULATION_TIME)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "solver") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST solver_opts.find(SOLVER_TYPE)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "integrator") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST solver_opts.find(INTEGRATOR)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "object") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type" , BAD_CAST "particle") );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "material") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST particle_opts.find(PARTICLE_MATERIAL)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "radius") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST particle_opts.find(PARTICLE_RADIUS)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	it = particle_opts.find(PARTICLE_MASS);
	pos = it->second.find(" ");
	str = it->second.substr(pos+1);
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "mass") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "dependency", BAD_CAST str.c_str()) );
	str = it->second.substr(0, pos);
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST str.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	it = particle_opts.find(PARTICLE_INERTIA);
	pos = it->second.find(" ");
	str = it->second.substr(pos+1);
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "inertia") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "dependency", BAD_CAST str.c_str()) );
	str = it->second.substr(0, pos);
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST str.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "arrange_shape") );
	str = particle_opts.find(PARTICLE_ARRANGE_SHAPE)->second;
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST str.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	if(str == "cube"){
		xmlWriteElement(writer, "arrange_size", PARTICLE_ARRANGE_SIZE, DIMENSION_TYPE);
	}

	xmlWriteElement(writer, "arrange_position", PARTICLE_ARRANGE_POSITION, VECTOR_TYPE);

	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "object") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type" , BAD_CAST "boundary") );

	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "material") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST boundary_opts.find(BOUNDARY_MATERIAL)->second.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	str = boundary_opts.find(BOUNDARY_GEOMETRY)->second;
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST "geometry") );
	xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST str.c_str()) );
	xmlCheckErrors( xmlTextWriterEndElement(writer) );

	if(str == "cube"){
		xmlWriteElement(writer, "geometry_size", BOUNDARY_GEOMETRY_SIZE, DIMENSION_TYPE);
	}

	xmlWriteElement(writer, "position", BOUNDARY_POSITION, VECTOR_TYPE);

	xmlCheckErrors( xmlTextWriterEndElement(writer) );

 	xmlCheckErrors( xmlTextWriterEndDocument(writer) );

	xmlFreeTextWriter(writer);
}

void xml_editor::xmlWriteElement(xmlTextWriterPtr writer, std::string text, options opt, data_type dt)
{
	std::map<int, std::string>* map_ptr = getMapPointer(opt);
	std::map<int, std::string>::iterator it = map_ptr->find(opt);
	xmlCheckErrors( xmlTextWriterStartElement(writer, BAD_CAST text.c_str()) );
	std::list<std::string> strList = get3parameters(it->second);
	if(dt){
		const std::string *str_ptr = dt==DIMENSION_TYPE ? dimension_str : vector_str;
		for(std::list<std::string>::iterator p = strList.begin(); p != strList.end(); p++){
			xmlCheckErrors( xmlTextWriterWriteAttribute(writer, BAD_CAST (str_ptr++)->c_str(), BAD_CAST p->c_str()) );
		}
		xmlCheckErrors( xmlTextWriterEndElement(writer) );
	}
	
// 	std::string t_str = str.substr(0, pos);
// 	xmlCheck(000000000)
}

std::list<std::string> xml_editor::get3parameters(std::string str)
{
	std::list<std::string> strList;
	size_t pos = str.find(" ");
	std::string p1 = str.substr(0, pos);
	str = str.substr(pos+1, str.size());
	pos = str.find(" ");
	std::string p2 = str.substr(0, pos);
	std::string p3 = str.substr(pos+1, str.size()-1);
	strList.push_back(p3);
	strList.push_back(p2);
	strList.push_back(p1);
	return strList;
}

std::map<int, std::string>* xml_editor::getMapPointer(options opt)
{
	if(opt < 10) 
	{
		return &sim_opts;
	}
	else if(opt < 20)
	{
		return &solver_opts;
	}
	else if(opt < 100)
	{
		return &particle_opts;
	}	
	else if(opt < 1000)
	{
		return &boundary_opts;
	}
	return NULL;
}