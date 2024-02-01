#include "Schroedinger_N_C_Class.hpp"

using namespace std;
using namespace arma;


//Constructor
SchroedingerCN::SchroedingerCN(double step_size, double time_step, double Tot_time)
{
	h = step_size;
	dt = time_step;
	T = Tot_time;

	M = 1/h; //MxM is size of total matrix
	length = M-2; //M-2 is size of matrix without boundary conditions
	V = mat(M,M,fill::zeros); //matrix for potential, set to zero before instance calls for potential
}




void SchroedingerCN::fill_A_B()
{
	//fills up A and B matrices
	//Focuses first on filling up imaginary part of A which is equal to minus the imaginary part of B
	
        int length_sqr = length*length; //the length of the vector u squared (M-2)^2
        double r_const = dt/(2*h*h); //the imaginary part of r (is not yet set to imaginary number)

	//setting up a vector (and b vector = -a.imag) 
	mat reduced_V = trans(V.submat(1,1,M-2,M-2)); //slicing V to fit M-2
	vec k_vec = vectorise(reduced_V, 0); //k = v_ij values in vector form
        vec a_im = 4*r_const+dt/2*k_vec; 

        sp_mat A_im(length_sqr,length_sqr); //sets up the imaginary part of A
					    //
        vec r_vec_sides(length*(length-1), fill::value(r_const));
        A_im.diag(length) = -r_vec_sides; //adds r - values to the (M-2)-th upper triangle
        A_im.diag(-length) = -r_vec_sides; //adds r - values to (M-2)th lower triangle

        vec r_vec_tridiag(length_sqr-1, fill::value(r_const)); //Sets up vector that will be along
        for (int i = 1; i < length; i++)
        {
		//removes each i*(M-2)-1th value from r that will be on tridiagonal
                r_vec_tridiag(i*length-1) = 0;
        }

	//sets all the imaginary parts in the tridiagonal
        A_im.diag() = a_im;
        A_im.diag(-1) = -r_vec_tridiag;
        A_im.diag(+1) = -r_vec_tridiag;

        sp_mat A_B_real = speye(length_sqr,length_sqr); //the real part, is the same for both A and B

	//setting A and B, putting together real and imaginary parts
	A = sp_cx_mat(A_B_real, A_im);
	B = conj(A);

}





void SchroedingerCN::set_potential_single_slit(double v_0)
{
	//sets potential to single slit
	int middle = M/2 -1; //Because of 0 indexing
    	int wallthick = M*0.01;
    	int half_space = M*0.025;
    	int space = M*0.05;


	// wall
    	V.submat(0, middle-wallthick+1, M-1, middle+wallthick) = mat(M, 0.02*M, fill::value(v_0));
	
	// aperture
    	V.submat(middle-half_space+1, middle-wallthick+1, middle+half_space, middle+wallthick) = mat(space, 0.02*M, fill::zeros);
    

}





void SchroedingerCN::set_potential_double_slit(double v_0)
{
	//sets potential to doubule slit
        int middle = M/2 -1; //Because of 0 indexing
        int wallthick = M*0.01;
        int half_space = M*0.025;
    	int space = M*0.05;

	// wall
    	V.submat(0, middle-wallthick+1, M-1, middle+wallthick) = mat(M, 0.02*M, fill::value(v_0));

    	//upper opening
    	V.submat(middle-half_space-space+1, middle-wallthick+1, middle-half_space, middle+wallthick) = mat(space, 0.02*M, fill::zeros);

    	// lower opening
    	V.submat(middle+half_space+1, middle-wallthick+1, middle+half_space+space, middle+wallthick) = mat(space, 0.02*M, fill::zeros);

	
}




void SchroedingerCN::set_potential_triple_slit(double v_0)
{
	//sets potential to triple slit
	int middle = M/2 -1; //Because of 0 indexing
    	int wallthick = M*0.01;
    	int half_space = M*0.025;
	int space = M*0.05;

    	// wall
    	V.submat(0, middle-wallthick+1, M-1, middle+wallthick) = mat(M, 0.02*M, fill::value(v_0));

	// middle opening
    	V.submat(middle-half_space+1, middle-wallthick+1, middle+half_space, middle+wallthick) = mat(space, 0.02*M, fill::zeros);

	// upper opening
    	V.submat(middle-half_space-2*space+1, middle-wallthick+1, middle-half_space-space, middle+wallthick) = mat(space, 0.02*M, fill::zeros);

	// lower opening
    	V.submat(middle+space+half_space+1, middle-wallthick+1, middle+half_space+2*space, middle+wallthick) = mat(space, 0.02*M, fill::zeros);


}


void SchroedingerCN::set_initial_state(double x_c, double y_c, double p_x, double p_y, double sigma_x, double sigma_y)
{
  	//sets initial state as normalised Gaussian wave packet
  	cx_double i(0.0, 1.0);

  	// making a type of meshgrid
  	vec x_real = linspace(0,1,M);
  	vec x_im = vec(M,fill::zeros);

  	vec y_real = linspace(0,1,M);
  	vec y_im = vec(M,fill::zeros);

  	cx_vec x = cx_vec(x_real,x_im);
  	cx_vec y = cx_vec(y_real,y_im);

  	cx_mat X = repmat(x, 1, M);
  	cx_mat Y = repmat(y.t(), M, 1);


  	cx_mat state_gauss = exp(-(pow(X-x_c,2)/(2*pow(sigma_x,2)))-(pow(Y-y_c,2)/(2*pow(sigma_y,2)))+i*p_x*X+i*p_y*Y);

	// cutting out boundaries
	cx_mat state_submat = state_gauss.submat(1,1,M-2,M-2);

  	// normalise
  	cx_mat state_normalised = state_submat/norm(state_submat);

  	// turn initial state matrix into vector
  	state_vec = vectorise(state_normalised);

}


void SchroedingerCN::single_crank_nicolson()
{
	//runs one single Crank-Nicolson, first matrix multiplication and then solve matrix eq. will update state_vec
	cx_mat b = B*state_vec;
	state_vec = spsolve(A,b,"superlu"); 
}

cx_vec SchroedingerCN::get_state_vec()
{
	//used to return state_vec to avoid accidental overriding
	return state_vec;
}
