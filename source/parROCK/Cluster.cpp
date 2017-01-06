#include "Cluster.h"

Cluster::Cluster()
	: nballs(0)
{

}

Cluster::Cluster(const Cluster& cl)
	: nballs(0)
{
	nballs = cl.Nballs();
}

Cluster::~Cluster()
{

}

void Cluster::addBall(ball* b)
{
	balls.push_back(b);
	nballs++;
}