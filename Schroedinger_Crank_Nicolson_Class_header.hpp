#include <iostream>
#include <armadillo>
#include <string>

class SchroedingerCN
{
private:

public:
    double h; //step_size
    double dt; // time_step, //delta t
    double T; //total time

    arma::cx_vec state_vec; // vector u that gets updates every iteration
    arma::sp_cx_mat A; // matrice A, used for Crank-Nicolson
    arma::sp_cx_mat B; // matrice B, used for Crank-Nicolson
    arma::mat V; // potetial matrix V over x-y plane

    int length;//M-2
    int M;

    SchroedingerCN(double step_size, double time_step, double Tot_time);
    
    void set_potential_single_slit(double v0);

    void set_potential_double_slit(double v0);

    void set_potential_triple_slit(double v0);

    void set_initial_state(double x_c, double y_c, double p_x, double p_y, double sigma_x, double sigma_y);

    void fill_A_B();

    void single_crank_nicolson();

    arma::cx_vec get_state_vec();
};
