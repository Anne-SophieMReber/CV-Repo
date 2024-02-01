#include "Schroedinger_N_C_Class.hpp"
#include <assert.h>
#include <fstream>
#include <string>
#include <armadillo>


using namespace arma;
using namespace std;

int main(int argc, const char* argv[])
{
	//asserting correct amount of arguments
	assert(argc == 12);

	double h = atof(argv[1]);
	double dt = atof(argv[2]);

	double T = atof(argv[3]);

	double x_c = atof(argv[4]);
	double sigma_x = atof(argv[5]);
	double p_x = atof(argv[6]);

	double y_c = atof(argv[7]);
	double sigma_y = atof(argv[8]);
	double p_y = atof(argv[9]);
	double v0 = atof(argv[10]);

	string potential_arg = argv[11];

	//creating instance of SchroedingerCN
	SchroedingerCN test_schroed(h, dt, T);

	//setting potential based on command line argument
	if(potential_arg == "single slit")
	{
		test_schroed.set_potential_single_slit(v0);
		cout << "single slit potential set" << endl;
	}
	else if(potential_arg == "double slit")
	{
		test_schroed.set_potential_double_slit(v0);
		cout << "double slit potential set" << endl;
	}
	else if(potential_arg == "triple slit")
	{
		test_schroed.set_potential_triple_slit(v0);
		cout << "double slit potential set" << endl;
	}
	else 
	{
		cout << "no potential set" << endl;
	}

	//filling A and B matrices
	test_schroed.fill_A_B();
	cout << "A and B filled" << endl;

	//Setting normalised gaussian initial state
	test_schroed.set_initial_state(x_c, y_c, p_x, p_y, sigma_x, sigma_y);
	cout << "initial state set" << endl;

	//Calculate amount of timesteps
	int time_steps = T/dt;
	cout << "Timesteps equals: "<< time_steps << endl;

	//saving probability (arma::vec) and all information as cx_cube(x,y,t)
	cx_cube U(test_schroed.length, test_schroed.length, time_steps);
	vec saved_p(time_steps);

	//looping through Crank-Nicolson and save each x-y for each timestep
	for(int i = 0; i < time_steps; i++)
	{
		test_schroed.single_crank_nicolson();
		//Saving probability
	 	saved_p(i) = norm(test_schroed.get_state_vec());
		assert((abs(saved_p(i)-1)) <= pow(10,-13)); //assert that probability is not deviating too much from 1

		//saving all information
		Row<cx_double> position_row = test_schroed.state_vec.t();
		Mat<cx_double> position_matrix = reshape(position_row, test_schroed.length, test_schroed.length);
		U.slice(i) = position_matrix;

	}

	cout << "finished running" << endl;
	
	//save probability, used to study deviation of probability
	saved_p.save("probability_simulation.bin", arma_binary);

	U.save("U_whole_cube.bin"); //all information in cube form, used plotting time evolution, probability at screen and animation 
	
}
