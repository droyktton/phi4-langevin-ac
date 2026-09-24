// Langevin dynamics of a circular domain in the disordered phi^4 model
// under an AC field h(t) = h0*cos(2*pi*f*t), at temperature T, with
// periodic boundary conditions (see langevin.h for the model and the
// integration scheme).
//
// The initial condition is a disk of phi=+1 (radius R, centered in the
// sample) in a phi=-1 background. Every tout time units the domain wall
// is detected and written out (walls_*.dat), together with a line of
// summary observables (timeseries_*.dat).
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>
#include "langevin.h"

struct Params
{
	int L=256;
	float h0=0.0, freq=0.0, T=0.0, radius=64.0, delta=0.2, dt=0.05;
	double tmax=100.0, tout=1.0, tconf=0.0;
	unsigned long seed=1, noise_seed=0;
	bool noise_seed_set=false;
};

void usage(const char *prog)
{
	std::cout << "usage: " << prog << " [--option value ...]\n"
		<< "  --L N           lattice size (LxL, periodic)            [256]\n"
		<< "  --h0 X          AC field amplitude                        [0]\n"
		<< "  --freq X        AC field frequency f, h=h0*cos(2 pi f t)  [0]\n"
		<< "  --T X           temperature                               [0]\n"
		<< "  --radius X      radius of the initial (phi=+1) disk       [64]\n"
		<< "  --delta X       disorder amplitude Delta                  [0.2]\n"
		<< "  --dt X          time step                                 [0.05]\n"
		<< "  --tmax X        total simulated time                      [100]\n"
		<< "  --tout X        time between wall outputs                 [1]\n"
		<< "  --tconf X       time between full-field snapshots (0=never) [0]\n"
		<< "  --seed N        disorder seed                             [1]\n"
		<< "  --noise-seed N  thermal-noise seed                        [seed+1]\n"
		<< "example: " << prog << " --L 512 --h0 0.3 --freq 0.001 --T 0.01 --radius 100 --delta 0.2 --dt 0.05 --tmax 2000 --tout 10\n";
}

bool parse(int argc, char **argv, Params &p)
{
	for(int i=1;i<argc;i++){
		std::string opt=argv[i];
		if(opt=="--help" || opt=="-h") return false;
		if(i+1>=argc){ std::cout << "missing value for " << opt << std::endl; return false; }
		const char *v=argv[++i];
		if(opt=="--L") p.L=atoi(v);
		else if(opt=="--h0") p.h0=atof(v);
		else if(opt=="--freq") p.freq=atof(v);
		else if(opt=="--T") p.T=atof(v);
		else if(opt=="--radius") p.radius=atof(v);
		else if(opt=="--delta") p.delta=atof(v);
		else if(opt=="--dt") p.dt=atof(v);
		else if(opt=="--tmax") p.tmax=atof(v);
		else if(opt=="--tout") p.tout=atof(v);
		else if(opt=="--tconf") p.tconf=atof(v);
		else if(opt=="--seed") p.seed=strtoul(v,NULL,10);
		else if(opt=="--noise-seed"){ p.noise_seed=strtoul(v,NULL,10); p.noise_seed_set=true; }
		else{ std::cout << "unknown option " << opt << std::endl; return false; }
	}
	if(!p.noise_seed_set) p.noise_seed=p.seed+1;
	if(p.L<4 || p.dt<=0 || p.tmax<0 || p.tout<=0 || p.T<0){
		std::cout << "invalid parameters (need L>=4, dt>0, tmax>=0, tout>0, T>=0)" << std::endl;
		return false;
	}
	return true;
}

// converts a time interval into a whole number of steps (at least 1)
long steps_for(double interval, float dt)
{
	long n=lround(interval/dt);
	return n<1?1:n;
}

int main(int argc, char **argv)
{
	Params p;
	if(!parse(argc,argv,p)){ usage(argv[0]); return 1; }

	if(p.dt*CEL>0.1)
		std::cout << "WARNING: dt*c=" << p.dt*CEL << " > 0.1; explicit Euler becomes unstable for dt*c >~ 0.25" << std::endl;

	long nsteps=lround(p.tmax/p.dt);
	long nout=steps_for(p.tout,p.dt);
	long nconf=(p.tconf>0)?steps_for(p.tconf,p.dt):0;

	std::ofstream log("logfile.dat");
	log << "L= " << p.L << "\nh0= " << p.h0 << "\nfreq= " << p.freq << "\nT= " << p.T
		<< "\nradius= " << p.radius << "\ndelta= " << p.delta << "\ndt= " << p.dt
		<< "\ntmax= " << p.tmax << " (" << nsteps << " steps)\ntout= " << p.tout << " (" << nout << " steps)"
		<< "\ntconf= " << p.tconf << "\nseed= " << p.seed << "\nnoise_seed= " << p.noise_seed
		<< "\nCEL (c)= " << CEL << "\nEPSILON0= " << EPSILON0 << "\nR0= " << R0 << std::endl;

	System sys(p.L,p.delta,p.h0,p.freq,p.T,p.dt,p.seed,p.noise_seed);
	sys.set_circle(p.radius);

	char filename[200];
	sprintf(filename,"walls_seed%lu_nseed%lu.dat",p.seed,p.noise_seed);
	std::ofstream walls_out(filename);
	sprintf(filename,"timeseries_seed%lu_nseed%lu.dat",p.seed,p.noise_seed);
	std::ofstream ts_out(filename);
	ts_out << "# t h(t) m=<phi> A+ R_eff=sqrt(A+/pi) n_wall xcm ycm <r> var(r)\n";

	auto record=[&](){
		WallStats s=sys.detect_wall();
		sys.print_wall(walls_out,s);
		ts_out << sys.get_time() << " " << sys.get_field() << " " << s.magnetization << " "
			<< s.area_positive << " " << sqrt(s.area_positive/M_PI) << " " << s.num_wall_points << " "
			<< s.xcm << " " << s.ycm << " " << s.r_mean << " " << s.r_var << std::endl;
		walls_out.flush();
	};
	auto snapshot=[&](){
		sprintf(filename,"config_t%g_seed%lu_nseed%lu.dat",sys.get_time(),p.seed,p.noise_seed);
		std::ofstream conf_out(filename);
		sys.print_config(conf_out);
	};

	record();
	if(nconf) snapshot();
	long done=0;
	while(done<nsteps){
		// advance to the next output (wall or snapshot) time
		long next=std::min(nsteps,(done/nout+1)*nout);
		if(nconf) next=std::min(next,(done/nconf+1)*nconf);
		sys.evolve(next-done);
		done=next;
		if(done%nout==0 || done==nsteps) record();
		if(nconf && done%nconf==0) snapshot();
	}

	std::cout << "done: " << nsteps << " steps, t=" << sys.get_time() << std::endl;
	return 0;
}
