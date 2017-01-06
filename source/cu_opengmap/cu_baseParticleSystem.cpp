#include "cu_baseParticleSystem.h"
#include <list>

cu_baseParticleSystem::cu_baseParticleSystem()
	: output_step(0)
	, cframe(0)
	, nframe(0)
	, boundaries(0)
	, d_boundaries(0)
	, ps(0)
	, d_sortedPos(0)
	, d_sortedVel(0)
	, d_sortedOmega(0)
	, d_GridParticleIndex(0)
	, d_GridParticleHash(0)
	, d_sortedIndex(0)
	, d_CellEnd(0)
	, d_CellStart(0)
{

}

cu_baseParticleSystem::~cu_baseParticleSystem()
{
	if(boundaries) delete boundaries; boundaries=0;
	if(d_boundaries) checkCudaErrors( cudaFree(d_boundaries) ); d_boundaries=0;
	if(d_GridParticleHash) checkCudaErrors( cudaFree(d_GridParticleHash) ); d_GridParticleHash=0;
	if(d_GridParticleIndex) checkCudaErrors( cudaFree(d_GridParticleIndex) ); d_GridParticleIndex=0;
	if(d_sortedIndex) checkCudaErrors( cudaFree(d_sortedIndex) ); d_sortedIndex=0;
	if(d_CellStart) checkCudaErrors( cudaFree(d_CellStart) ); d_CellStart=0;
	if(d_CellEnd) checkCudaErrors( cudaFree(d_CellEnd) ); d_CellEnd=0;

	if(d_sortedPos) cudaFree(d_sortedPos); d_sortedPos=0;
	if(d_sortedVel) cudaFree(d_sortedVel); d_sortedVel=0;
	if(d_sortedOmega) cudaFree(d_sortedOmega); d_sortedOmega=0;
}

void cu_baseParticleSystem::load_input(const char *path)
{
	std::string ch;
	
	std::fstream inf(path, std::ios::in);
	float3 start_p, end_p;
	cmaterialType<float> material;
	h_paras.mass=0.0f;
	h_paras.inertia=0.0f;
	while(!inf.eof())
	{
 		inf >> ch;
 		if(ch=="nParticle") inf >> h_paras.nParticle;
 		if(ch=="radius") inf >> h_paras.radius;
 		if(ch=="material")	
 		{
 			inf >> ch;
 			if(ch=="iron") 
			{ 
				material.density=(float)IRON_DENSITY; 
				material.youngs=(float)IRON_YOUNGS_MODULUS; 
				material.poisson=(float)IRON_POISSON_RATIO; 
			}
 			else if(ch=="polyethylene") 
			{ 
				material.density=(float)POLYETHYLENE_DENSITY; 
				material.youngs=(float)POLYETHYLENE_YOUNGS_MODULUS; 
				material.poisson=(float)POLYETHYLENE_POISSON_RATIO; 
			}
			else if(ch=="polystyrene") 
			{ 
				material.density=(float)POLYSTYRENE_DENSITY; 
				material.youngs=(float)POLYSTYRENE_YOUNGS_MODULUS; 
				material.poisson=(float)POLYSTYRENE_POISSON_RATIO; 
			}
		}
 		if(ch=="world_origin")
 		{
 			float3 world;
 			inf >> world.x >> world.y >> world.z;
 			h_paras.worldOrigin=world;
 		}
		if(ch=="grid_size")
		{
			//float3 gridsize;
			inf >> h_paras.gridSize.x >> h_paras.gridSize.y >> h_paras.gridSize.z;
			//h_paras.gridSize=;
		}
		if(ch=="gravity")
		{
			float3 gravity;
			inf >> gravity.x >> gravity.y >> gravity.z;
			h_paras.gravity=gravity;
		}
		if(ch=="mass") inf >> h_paras.mass;
		if(ch=="simulation_time") inf >> h_paras.endTime;
		if(ch=="time_step") inf >> h_paras.dt;
		if(ch=="output_step") inf >> output_step;
		if(ch=="arrange") 
		{
			inf >> start_p.x >> start_p.y >> start_p.z;
			inf >> end_p.x >> end_p.y >> end_p.z;
		}
		if(ch=="event")
		{
			event();
			while(ch!="end")
			{
				inf >> ch;
				if(ch=="boundary")
				{
					unsigned neb = 0;
					float exTime=0.0f;
					inf >> neb;
					for(unsigned i(0); i<neb; i++)
					{
						event::eventType ev;
						ev.type = 'b';
						inf >> ev.target;
						inf >> exTime;
						inf >> ev.argType;
						ev.excute_frame = static_cast<unsigned>(exTime/h_paras.dt) + 1;
						switch(ev.argType)
						{
						case 'v':
							inf >> ev.arg_vel.x >> ev.arg_vel.y >> ev.arg_vel.z;
							break;
						case 'p':
							inf >> ev.arg_pos.x >> ev.arg_pos.y >> ev.arg_pos.z;
							break;
						}
						event::addEvent(ev);
					}
				}
			}
		}
	}
	inf.close();
	ps = new cu_setOfParticle;
	ps->arrange_particle(start_p, end_p, h_paras.radius, 2.1f);
	h_paras.nParticle=ps->nP;
	h_paras.cellSize=2*h_paras.radius;
	h_paras.effRadius=h_paras.radius*h_paras.radius / 2*h_paras.radius;
	h_paras.nCell=h_paras.gridSize.x*h_paras.gridSize.y*h_paras.gridSize.z;
	
	nframe=static_cast<unsigned>((h_paras.endTime / h_paras.dt)+1);
	output_frame=(nframe-1) / static_cast<unsigned>((h_paras.endTime / output_step));

	if(!h_paras.mass) 
		h_paras.mass = material.density * 4.0f * (float)PI * h_paras.radius * h_paras.radius * h_paras.radius / 3.0f;
	if(!h_paras.inertia)
		h_paras.inertia=2.0f * h_paras.mass * h_paras.radius * h_paras.radius / 5.0f;

	h_paras.set_kn(material.youngs, material.poisson, h_paras.radius, (float)ACRYLIC_YOUNG_MODULUS, (float)ACRYLIC_POISSON_RATIO);
	h_paras.set_vn(0.9f, 0.98f);
	h_paras.ks_pp=0.8f*h_paras.kn_pp;
	h_paras.ks_pw=0.8f*h_paras.kn_pw;
	h_paras.vs_pp=h_paras.vn_pp;
	h_paras.vs_pw=h_paras.vn_pw;
	h_paras.mus_pp=0.4f;// 상욱이가 수정할 부분
	h_paras.mus_pw=0.3f;// 상욱이가 수정할 부분

	h_paras.half_two_dt = 0.5f*h_paras.dt*h_paras.dt;
	h_paras.invMass = 1 / h_paras.mass;
	h_paras.invInertia = 1 / h_paras.inertia;
	h_paras.half_dt = 0.5f*h_paras.dt;

	for(unsigned i(0); i<h_paras.nParticle; i++)
	{
		ps->h_acc[i*3+0]=h_paras.gravity.x;
		ps->h_acc[i*3+1]=h_paras.gravity.y;
		ps->h_acc[i*3+2]=h_paras.gravity.z;
	}
	/*setSymbolicParameter(&h_paras);*/

	checkCudaErrors( cudaMalloc((void**)&d_GridParticleHash, h_paras.nParticle*sizeof(unsigned)) );
	checkCudaErrors( cudaMalloc((void**)&d_GridParticleIndex, h_paras.nParticle*sizeof(unsigned)) );
	checkCudaErrors( cudaMalloc((void**)&d_sortedIndex, h_paras.nParticle*sizeof(unsigned)) );
	checkCudaErrors( cudaMalloc((void**)&d_CellStart, h_paras.nCell*sizeof(unsigned)) );
	checkCudaErrors( cudaMalloc((void**)&d_CellEnd, h_paras.nCell*sizeof(unsigned)) );

	checkCudaErrors( cudaMalloc((void**)&d_sortedPos, sizeof(float)*h_paras.nParticle*3) );
	checkCudaErrors( cudaMalloc((void**)&d_sortedVel, sizeof(float)*h_paras.nParticle*3) );
	checkCudaErrors( cudaMalloc((void**)&d_sortedOmega, sizeof(float)*h_paras.nParticle*3) );
}

void cu_baseParticleSystem::load_boundary(const char *path)
{
	//unsigned int nQuad;
	unsigned int nQuad_point;
	float* Quad_vertics;
	unsigned int* Quad_index;

	unsigned int j=0;
	std::list<char*> strings;
	FILE *fi;
	fopen_s(&fi,path,"rt");
	unsigned char c=fgetc(fi);
	while(!feof(fi))
	{
		char *chr = new char[20];
		while(c!=' '&&c!='\n') 
		{
			if(c=='#')
			{
				while(c!='\n') 
					c=fgetc(fi);
				c=fgetc(fi);
			}
			while(c==' '||c=='\n') 
				c=fgetc(fi);

			chr[j++]=c;
			c=fgetc(fi);			
		}
		chr[j]='\0';
		strings.push_back(chr);
		/*i++;*/ j=0;
		while(c==' ' || c=='\n') 
			c=fgetc(fi);
		//c=fgetc(fi);
	}

	for(std::list<char*>::iterator ch=strings.begin(); ch!=strings.end(); ch++)
	{
		if(!strcmp(*ch,"quadrangle"))
		{
			h_paras.nBoundary=atoi(*(++ch));
			if(!strcmp(*(++ch),"point"))
			{
				nQuad_point=atoi(*(++ch));
				Quad_vertics = new float[nQuad_point*3];
				Quad_index = new unsigned int[h_paras.nBoundary*4];
				for(j=0; j<nQuad_point*3; j++)
					Quad_vertics[j]=(float)atof(*(++ch));
			}
			else
			{
				//printf("point input data is wrong");
			}
			if(!strcmp(*(++ch),"index"))
			{
				for(j=0; j<h_paras.nBoundary*4; j++)
				{
					Quad_index[j]=atoi(*(++ch));
				}
			}
			else
			{
				//printf("Not match the sequence of quadrangle point index data");
			}
		}
	}

	for(std::list<char*>::iterator ch=strings.begin(); ch!=strings.end(); ch++)
	{
		char* chr = *ch;
		delete [] chr; chr = 0;
	}
	fclose(fi);
	unsigned int* id;
	boundaries=new boundaryType[h_paras.nBoundary];
	for(unsigned int i=0; i<h_paras.nBoundary; i++)
	{
		id=Quad_index+i*4;
		boundaries[i].xw = make_float3(Quad_vertics[id[0]*3+0],Quad_vertics[id[0]*3+1],Quad_vertics[id[0]*3+2]);
		float3 pa= make_float3(Quad_vertics[id[1]*3+0],Quad_vertics[id[1]*3+1],Quad_vertics[id[1]*3+2]) - boundaries[i].xw;
		float3 pb= make_float3(Quad_vertics[id[3]*3+0],Quad_vertics[id[3]*3+1],Quad_vertics[id[3]*3+2]) - boundaries[i].xw;
 		boundaries[i].l1=length(pa);
		boundaries[i].l2=length(pb);
		boundaries[i].u1=pa/boundaries[i].l1;
		boundaries[i].u2=pb/boundaries[i].l2;
		boundaries[i].uw=cross(boundaries[i].u1,boundaries[i].u2);
// 		makeQuadrangle(vector3(Quad_vertics[id[0]*3+0],Quad_vertics[id[0]*3+1],Quad_vertics[id[0]*3+2]),
// 			vector3(Quad_vertics[id[1]*3+0],Quad_vertics[id[1]*3+1],Quad_vertics[id[1]*3+2]),
// 			vector3(Quad_vertics[id[2]*3+0],Quad_vertics[id[2]*3+1],Quad_vertics[id[2]*3+2]),
// 			vector3(Quad_vertics[id[3]*3+0],Quad_vertics[id[3]*3+1],Quad_vertics[id[3]*3+2]));
	}

	if(Quad_vertics) delete [] Quad_vertics; Quad_vertics = 0;
	if(Quad_index) delete [] Quad_index; Quad_index = 0;
}

void cu_baseParticleSystem::exeEvent(unsigned _cframe)
{

	for(eve=event::events.begin(); eve!=event::events.end(); eve++)
	{
		if(eve->excute_frame!=_cframe) break;
		switch(eve->type)
		{
		case 'b':
			switch(eve->argType)
			{
			case 'p':
				if(eve->excute_frame!=_cframe) break;
				changeBoundaryPosition(eve->target, eve->arg_pos);
				checkCudaErrors( cudaMemcpy(d_boundaries, boundaries, sizeof(boundaryType)*h_paras.nBoundary, cudaMemcpyHostToDevice) );
				break;

			case 'v':
				if(eve->excute_frame>_cframe) break;
				changeBoundaryVelocity(eve->target, eve->arg_vel);
				checkCudaErrors( cudaMemcpy(d_boundaries, boundaries, sizeof(boundaryType)*h_paras.nBoundary, cudaMemcpyHostToDevice) );
				break;
			}
			
			
		}
	}
}

void cu_baseParticleSystem::changeBoundaryPosition(unsigned id, tvector3<float>& c_pos)
{
	//std::vector<boundaryType>::iterator boundary;
	boundaries[id].xw+=make_float3(c_pos.x,c_pos.y,c_pos.z);
	//boundary->xw = boundary->xw + c_pos;
}

void cu_baseParticleSystem::changeBoundaryVelocity(unsigned id, tvector3<float>& c_vel)
{
	boundaries[id].xw+=make_float3(c_vel.x,c_vel.y,c_vel.z) * h_paras.dt;
}

//void cu_baseParticleSystem::changeBoundaryVelocity(unsigned id, tvector3<float)