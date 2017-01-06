#include "xml/xml_editor.h"

xml_editor *editor;

const std::string strOpt[8] = {"Name", 
	"Processing unit", 
	"Dimension",
	"Precision",
	"Time step",
	"Export time step",
	"Simulation time"
};

const std::string solver_strOpt[2] = {
	"Solver type",
	"Integrator"
};

const std::string particle_strOpt[7] = {
	"P. Material",
	"P. Radius",
	"P. Mass",
	"P. Inertia",
	"P. Shape",
	"P. Size",
	"P. Position"
};

const std::string boundary_strOpt[4] = {
	"B. Material",
	"B. Type",
	"B. Size",
	"B. Position"
};

bool bOpenParticleTree = false;
bool bOpenBoundaryTree = false;
bool bOpenObjectTree = false;


void displayText(std::string com="")
{
	std::cout << "-- simulation option --" << std::endl
		<< "    " << NAME << ". " << strOpt[0] << " : " << editor->getValue(NAME) << std::endl
		<< "    " << PROCESSING_UNIT << ". " << strOpt[1] << " : " << editor->getValue(PROCESSING_UNIT) << std::endl
		<< "    " << DIMENSION << ". " << strOpt[2] << " : " << editor->getValue(DIMENSION) << std::endl
		<< "    " << PRECISION << ". " << strOpt[3] << " : " << editor->getValue(PRECISION) << std::endl
		<< "    " << TIME_STEP << ". " << strOpt[4] << " : " << editor->getValue(TIME_STEP) << std::endl
		<< "    " << EXPORT_TIME_STEP << ". " << strOpt[5] << " : " << editor->getValue(EXPORT_TIME_STEP) << std::endl
		<< "    " << SIMULATION_TIME << ". " << strOpt[6] << " : " << editor->getValue(SIMULATION_TIME) << std::endl
		<< "-- solver option --"  << std::endl
		<< "    " << SOLVER_TYPE << ". " << solver_strOpt[0] << " : " << editor->getValue(SOLVER_TYPE) << std::endl
		<< "    " << INTEGRATOR << ". " << solver_strOpt[1] << " : " << editor->getValue(INTEGRATOR) << std::endl
		<< "-- object option --" << std::endl
// 		<< "    " << PARTICLE_MATERIAL << ". " << strOpt[8] << " : " << editor->getValue(PARTICLE_MATERIAL) << std::endl
// 		<< "    " << PARTICLE_RADIUS << ". " << strOpt[9] << " : " << editor->getValue(PARTICLE_RADIUS) << std::endl
// 		<< "    " << PARTICLE_MASS << ". " << strOpt[10] << " : " << editor->getValue(PARTICLE_MASS) << std::endl
// 		<< "    " << PARTICLE_INERTIA << ". " << strOpt[11] << " : " << editor->getValue(PARTICLE_INERTIA) << std::endl
// 		<< "    " << PARTICLE_ARRANGE_SHAPE << ". " << strOpt[12] << " : " << editor->getValue(PARTICLE_ARRANGE_SHAPE) << std::endl
// 		<< "    " << PARTICLE_ARRANGE_SIZE << ". " << strOpt[13] << " : " << editor->getValue(PARTICLE_ARRANGE_SIZE) << std::endl
// 		<< "-- boundary option --" << std::endl
// 		<< "    " << 
		;
	
	if(com == "object"){
		std::cout << "   < " << "object - " << editor->get_n_objects() << std::endl;
		for(int i = 0; i < editor->get_n_objects(); i++){
			std::cout << "    " << "   > " << editor->get_object_name(i) << std::endl;
		}
		bOpenObjectTree = true;
		//editor->object_step = 1;
	}
	else{
		std::cout << "   > " << "object - " << editor->get_n_objects() << std::endl;
	}
	if(com == "particle" || bOpenParticleTree){
		/*if(editor->object_step == 0) return;*/
		//std::cout << "   < " << "object - " << editor->get_n_objects() << std::endl;
		for(int i = 0; i < editor->get_n_objects(); i++){
			std::string name = editor->get_object_name(i);
			if(name == "particle"){
				std::cout << "    " << "   < " << editor->get_object_name(i) << std::endl;
				std::cout 
					<< "    " << "    " << "    " << PARTICLE_MATERIAL << ". " << particle_strOpt[0] << " : " << editor->getValue(PARTICLE_MATERIAL) << std::endl
					<< "    " << "    " << "    " << PARTICLE_RADIUS << ". " << particle_strOpt[1] << " : " << editor->getValue(PARTICLE_RADIUS) << std::endl
					<< "    " << "    " << "    " << PARTICLE_MASS << ". " << particle_strOpt[2] << " : " << editor->getValue(PARTICLE_MASS) << std::endl
					<< "    " << "    " << "    " << PARTICLE_INERTIA << ". " << particle_strOpt[3] << " : " << editor->getValue(PARTICLE_INERTIA) << std::endl
					<< "    " << "    " << "    " << PARTICLE_ARRANGE_SHAPE << ". " << particle_strOpt[4] << " : " << editor->getValue(PARTICLE_ARRANGE_SHAPE) << std::endl
					<< "    " << "    " << "    " << PARTICLE_ARRANGE_SIZE << ". " << particle_strOpt[5] << " : " << editor->getValue(PARTICLE_ARRANGE_SIZE) << std::endl
					<< "    " << "    " << "    " << PARTICLE_ARRANGE_POSITION << ". " << particle_strOpt[6] << " : " << editor->getValue(PARTICLE_ARRANGE_POSITION) << std::endl
					;
			}
			else if(!bOpenBoundaryTree && com != "boundary"){
				std::cout << "    " << "   > " << editor->get_object_name(i) << std::endl;
			}
// 			else{
// 				std::cout << "    " << "   > " << editor->get_object_name(i) << std::endl;
// 			}
		} 
		bOpenParticleTree = true;
	}
	if(com == "boundary" || bOpenBoundaryTree){
		//if(editor->object_step == 0) return;
		//std::cout << "   < " << "object - " << editor->get_n_objects() << std::endl;
		for(int i = 0; i < editor->get_n_objects(); i++){
			std::string name = editor->get_object_name(i);
			if(name == "boundary"){
				std::cout << "    " << "   < " << editor->get_object_name(i) << std::endl;
				std::cout 
					<< "    " << "    " << "    " << BOUNDARY_MATERIAL << ". " << boundary_strOpt[0] << " : " << editor->getValue(BOUNDARY_MATERIAL) << std::endl
					<< "    " << "    " << "    " << BOUNDARY_GEOMETRY << ". " << boundary_strOpt[1] << " : " << editor->getValue(BOUNDARY_GEOMETRY) << std::endl
					<< "    " << "    " << "    " << BOUNDARY_GEOMETRY_SIZE << ". " << boundary_strOpt[2] << " : " << editor->getValue(BOUNDARY_GEOMETRY_SIZE) << std::endl
					<< "    " << "    " << "    " << BOUNDARY_POSITION << ". " << boundary_strOpt[3] << " : " << editor->getValue(BOUNDARY_POSITION) << std::endl
					;
			}
			else if(!bOpenParticleTree && com != "particle"){
				std::cout << "    " << "   > " << editor->get_object_name(i) << std::endl;
			}
// 			else{
// 				std::cout << "    " << "   > " << editor->get_object_name(i) << std::endl;
// 			}
		}
		bOpenBoundaryTree = true;
	}
// 	if(!bOpenObjectTree)
// 	{
// 		std::cout << "   > " << "object - " << editor->get_n_objects() << std::endl;
// 	}
	if(com == ""){
		bOpenObjectTree = bOpenParticleTree = bOpenBoundaryTree = false;
	}
}

std::string getOptionString(int opt)
{
	if(opt < 10){
		return strOpt[opt];
	}
	else if(opt < 20){
		opt %= 10;
		return solver_strOpt[opt];
	}
	else if(opt < 100){
		opt %= 10;
		return particle_strOpt[opt];
	}
	else if(opt < 1000){
		opt %= 10;
		return boundary_strOpt[opt];
	}
	return "";
}

bool processCommand(std::string& value)
{
	if(value == "done")
	{
		editor->makeXmlFile();
	}
	else
	{
		int opt = atoi(value.c_str());
		if(!opt){
			system("cls");
			displayText(value);
			return true;
		}
		int setwSize = 0;
		std::string optString = getOptionString(opt);
		if(optString == "")
			return true;
		std::string strval;
		std::list<std::string> input;
		std::list<std::string> strlist = editor->getInputList(opt);
		std::cout << " >> " << optString << " >> " << std::endl;
		setwSize += optString.size()+4;
		std::ios::right;
		for(std::list<std::string>::iterator str = strlist.begin(); str != strlist.end(); str++){
			std::cout << std::setw(setwSize) << *str << " >> ";
			std::cin >> strval;
			input.push_back(strval);
		}
		editor->updateOption(opt, input);
		system("cls");	
		displayText();
	}

	return true;
}

int main(int argc, char** argv)
{
	LIBXML_TEST_VERSION

	std::string value;
	editor = new xml_editor;
	displayText();
	do 
	{
		value.clear();
		std::cout << " >> ";
		std::cin >> value;
	} while (processCommand(value));

	delete editor;
	return 0;
}