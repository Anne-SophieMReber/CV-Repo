# Simulating the unitless Schrödinger equation with Crank-Nicolson

This code simulates the unitless Schroedinger equation with the Crank-Nicolson method with the variables:
| step size h | time step dt | total simulation time T | center center x_c | wave center y_c | wavepacket momenta p_x |  wavepacket momenta p_y  | wavepacket width sigma_x | wavepacket width sigma_y  |potential strength v0 | potential type |
| :---      |:----  |:----  |:----  |:----  |:----  |:----  |:----  |:----  |:----  |:----  |

 x_c, y_c, p_x, p_y, sigma_x and sigma_y are all used to initialize the wavepacket that is simulated throughout the code

## Running the code
To compile and link the code, simply write ```make``` into the commandline. After writing make, you can write you own values into the command line as such:


```
./main.exe <h> <dt> <T> <x_c> <sigma_x> <p_x> <y_c> <sigma_y> <p_y> <v0> <string set_potential>
```
(all values are double, except for ```<string set_potential>```)


The different potentials are given as

if ```<string set_potential> = "single slit"``` V is set to single slit potenital with function set_potential_single_slit(v0),

if ```<string set_potential> = "double slit"``` V is set to double slit potenital with function set_potential_double_slit(v0),

if ```<string set_potential> = "triple slit" ``` V is set to single slit potenital with function set_potential_triple_slit(v0),

else: (something must be written to pass assert test), V is not set and equal to zero everywhere.
 
 running the code results in two ```arma_binary files```:
| Name     | Type | Data |
| :---      |:----  |:--- |
|  ```"probability_simulation.bin"``` | arma::vec   | absolute probability for each timestep (p(t))|    
| ```"U_whole_cube.bin"``` | arma::cx_cube  | all complex x-y values for all timesteps in cube form (cx_double(x),cx_double(y),t) |
 

## Build of code
The code consits of 3 files, sorted as headers in the \include foler and as source files in the \src folder. The main is inside ```main.cpp```, which loops creates an instance of the SchroedingerCN class with commandline variables and loops through it for the time and with timesteps given by the commandline arguments.

For the class SchroedingerCN we have two files: a header ```Schroedinger_N_C_Class.hpp``` and a source file ```Schroedinger_N_C_Class.cpp```. 

The method in the class are given bellow. The order from top to bottom is a recommended order that the functions should be called in to avoid problems, such as A and B not being adapted to the correct potential.
|Use |Fucntion |
| :---      |:----  |
|  Constructer:| ```SchroedingerCN(double step_size, double time_step, double Tot_time)```|    
| Potentials:  |```void set_potential_single_slit(double v0)``` |
|  |```void set_potential_double_slit(double v0)``` |
|  | ```void set_potential_triple_slit(double v0)```|
| Filling matrices A and B: | ```void fill_A_B()```  |
| Setting initial state |  ```set_initial_state(double x_c, double y_c, double p_x, double p_y, double sigma_x, double sigma_y)```|
| Running one singular Crank-Nicolson iteration | ```single_crank_nicolson()```|
| Accessing state_vec variable |```arma::cx_vec get_state_vec()```|

 ## Recreating data and plots
To recreate the figures and data given in the report you can run these corresponding lines down below. Beware that the files need to be renamed to not override another. The variables given are to recreate the data, of course these can be changed to explore different topics within solving the Schrödinger equation with the Crank-Nicolson method.

All files can be plotted with the python code ```plotting.py```. To plot the correct file, the file name needs to be manually changed inside the cide.
 
To recreate the data used to look into the conservation of probability, run:
```
 ./main.exe 0.005 0.000025 0.008 0.25 0.05 200 0.5 0.05 0 0 "-"
 ./main.exe 0.005 0.000025 0.008 0.25 0.05 200 0.5 0.10 0 10000000000 "double slit"
```


To get the data to study time evolution, run:
```
./main.exe 0.005 0.000025 0.002 0.25 0.05 200 0.5 0.20 0 10000000000 "double slit"
```

To recreate the figures showing the probability on a detector screen for single, double and triple slit, run:
```
./main.exe 0.005 0.000025 0.002 0.25 0.05 200 0.5 0.20 0 10000000000 "double slit"
./main.exe 0.005 0.000025 0.002 0.25 0.05 200 0.5 0.20 0 10000000000 "single slit"
./main.exe 0.005 0.000025 0.002 0.25 0.05 200 0.5 0.20 0 10000000000 "triple slit"
```
## Misc.
The folders data and plots are the data and figures we simulated ourselves and used in the report.
