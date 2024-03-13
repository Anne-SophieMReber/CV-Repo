# Dipoler i et ytre elektrisk felt

## Introduksjon

Vi ønsker å se nærmere på hva som skjer dersom vi plasserer polare molekyler, dipoler, i et ytre elekrisk felt og tar utgangspunkt i det polare molekylet hydrogenfluorid, HF. I dette eksperimentet vil vi først ta utgangspunkt i en simulering av HCl-molekyler som også er polare molekyler og endre denne til å gjelde for HF. Etter å ha simulert HF-molekylene uten påvirkning simulerer vi dem først i et homogent elektrisk felt og så i et inhomogent elektrisk felt modellert som feltet fra en positiv ladning.

## Teori

Hydrogenfluorid, HF, er som navnet tilsier, en kjemisk forbindelse bestående av ett hydrogenatom og ett fluoratom. Forbindelsen har kokepunkt ved 19.5 $^\circ C$ og smeltepunkt ved -83.6 $^\circ C$. Mellom hydrogenfluoridmolekyler dannes det sterke hydrogenbindinger mellom som vist på figuren under:

![Alt text](https://file%2B.vscode-resource.vscode-cdn.net/c%3A/Users/rache/OneDrive/Notebooks/University/3.%20semester/el-mag/el-mag/computational%20essay/hydrogenbinding.png)

https://chemdictionary.org/hydrogen-bonding-in-hydrogen-flouride-hf/

Polare molekyler er molekyler som har en skjev ladningsfordeling. Atomene i molekylene er bundet sammen med en elektronparbinding, som også kalles en kovalent binding. Dersom to atomer i et molekyl har ulik elektronegativitet, vil de trekke forskjellig på elektronene i kovalent bindingen som gir en polar kovalent binding. For HF-molekylet er det fluor som er mest elektronegativt. Det vil derfor trekke mer på elektronene i bindingen som gir atomet en delvis negativ ladning mens hydrogenet vil få en delvis positiv ladning.
https://en.wikipedia.org/wiki/Hydrogen_fluoride

Fra elektrodynamikken vet vi at like ladninger frastøter hverandre, mens ulike ladninger tiltrekkes.

## Simulasjon av molekylene uten påvirkning

### Oppsetting av simulasjonen

Først importerer vi de nødvendige pakkene:


```python
import numpy as np
import time # to measure time
import matplotlib.pyplot as plt
np.random.seed(5) # sets the seed for the simulation
```

Vi definerer deretter konstanter som vi trenger i simulasjonen. Disse inkluderer atommasseenheten $u$, elementærladningen $e$, permitiviteten i vakuum $\epsilon_0$, Coulombs konstant $k_e$ og Boltzmanns konstant $k$.


```python
u = 1.66053906660e-27 # Atomic mass unit
e_charge = 1.602176634e-19 # Elementary charge (1 eV)
vac_perm = 8.8541878128e-12 # Permitivity in vakuum [s^2*C^2*/m^3/kg]
k_e = 1/(4*np.pi*vac_perm) # Coulomb constant [m^3*kg/s^2/C^2] 
k_boltz = 1.380649e-23 # Boltzmann constant
```

Til simuleringen trenger vi noen simuleringsparametre. Vi trenger å vite dimensjonene til griddet vårt, hvor mange atomer vi vil ha i simulasjonen og lengden/antall tidssteg. Vi definerer derfor disse.


```python
# Simulation parameters
grid_dims = np.array([4,4]) # Determines the number of molecules in the simulation (x * y)
num_molecules = np.prod(grid_dims) # Explicitly determine number of molecules in the simulation, here set to fill the grid
timesteps = 1000 # Number of timesteps to simulate
dt = 5e-17 # Extremely low timestep [s]
```

Vi modellerer de polare molekylene som en elektrisk dipol, altså som en positiv ladning bundet til en negativ ladning. Vi lager en funksjon som genererer en startposisjon for alle HF-molekylene i griddet og lar hvert molekyl være rotert en tilfeldig vinkel. Vi lar initialposisjonen til atomene i hvert molekyl være en "bindingslengde" unna hverandre slik som det er i likevekt.


```python
def make_liquid(num_molecules, dims, bond_length, grid_dims, side_lengths):
    """Function to generate the polar molecules on a square grid"""
    
    # pos lets one atom spawn a bond_length away from the other atom, in a uniformly distributed random direction
    pos = np.random.normal(0,1,size=(num_molecules,dims)) # create an array to hold the x and y values of the hydrogen atom positions, relative to the molecule's location
    r = np.linalg.norm(pos, axis = 1) # determine the length of the random spacing between the atoms
    
    pos = pos/r.reshape(num_molecules,1)*bond_length # scale the positions by the random spacing length to normalize
    """Note: reshape allows the array of shape (num_molecules, dims) to be
    divided by an array (tuple) with shape (num_molecules, )
    by giving it shape (num_molecules, 1), both the x and y component will then be divided by r. """
    
    molecule = np.zeros((num_molecules, dims)) # array to hold the positions of the different atoms
    
    # initialize the molecules in a square grid, with random rotation
    xx = np.linspace(-side_lengths[0]/2, side_lengths[0]/2, grid_dims[0]) # x dimension for meshgrid
    yy = np.linspace(-side_lengths[1]/2, side_lengths[1]/2, grid_dims[1]) # y dimension for meshgrid
    mesh_x, mesh_y = np.meshgrid(xx,yy) # creates two meshgrids 
    
    # ravel flattens the 2d array into a 1d array
    molecule[:,0] = np.ravel(mesh_x) # Sets the x-position for all the hydrogen atoms
    molecule[:,1] = np.ravel(mesh_y) # Sets the y-position for all the hydrogen atoms
    # The centre of the molecules will be in a grid formation
    hydrogen = molecule - pos/2 
    fluoride = molecule + pos/2
    
    return hydrogen, fluoride
```

### Parametre for simulasjonen

For å kunne simulere et spesifikt hydrogenfluoridsystem må vi definere noen parametere. Vi definerer antall atomer i molekylet, hvilken dimensjon vi ønsker å gjøre simulasjonen, og noen flere konstanter som gjelder spesifikt for HF, som ladningsfordelingen og bindingskonstanten. Vi viser hvordan vi finner disse her.


```python
num_atoms = 2 #number of atoms in one molecule
dims = 2 #number of spatial dimensions 
# (the rest of the code is not written to handle anything but two dimensions, though you could change that)
```

#### Finne ladningsfordelingen

Vi finner først ladningen på atomene ved å bruke definisjonen av dipolmoment.
Dipolmoment er definert som 
$$
\mu = Q r
$$
hvor $Q$ er ladningen på atomene og $r$ er avstanden mellom atomene. Ladningen blir da
$$
Q = \frac{\mu}{r}.
$$

Dipolmomentet for HF er 1.91 D, hvor 1 D = $3.3356 \cdot 10^{-30}$ m. Avstanden mellom atomene er $r$ = 91.7 pm ($10^{-12}$ m).

Vi finner ladningsfordelingen ved å dele ladningen, $Q$, på elektronladningen (siden det er elektroner de deler i bindingen).

https://techiescientist.com/hf-polar-or-nonpolar/
https://chem.libretexts.org/Courses/Mount_Royal_University/Chem_1201/Unit_3%3A_Chemical_Bonding_I_-_Lewis_Theory/3.4%3A_Bond_Polarity


```python
# Regner ut ladningsfordelingen
D_mom = 1.91 # Dipolmomentet
D = 3.3356e-30 # Verdien av 1 D [m]
r = 91.7e-12 # Avstanden mellom molekylene [m]

Q = (D * D_mom) / r # Ladningen på atomene
distr = Q / e_charge # Ladningsfordelingen mellom atomene

print(f"Ladningsfordelingen mellom atomene i molekylet = {distr:.2e}")
```

    Ladningsfordelingen mellom atomene i molekylet = 4.34e-01
    

#### Bindingsstyrkekonstanten til HF

Bindingsstyrkekonstanten regnet vi ut ved hjelp av overgangsfrekvensen når HF går fra grunntilstanden til første eksitasjonstilstanden.
Formelen vi bruker er 
$$ k = \omega^2\mu $$
hvor $\mu$ er gjennomsnittlig molekylærvekt som regnes ut slik: 
$$\frac{m_H+m_F}{m_Hm_F} = \frac{1.0078u\cdot18.994u}{1.0078u+18.994u} = 0.95702u$$
Dette må etterpå ganges med $1.6605\cdot10^{-27}\frac{kg}{u}$ for å konverte det til kg. $\omega$ er frekvensen $2\pi12.4\cdot10^{13}$

Ganget sammen får vi 
$$k = 0.95702u\cdot\left(2\pi12.4\cdot10^{13}Hz\right)^{2}\cdot1.6605\cdot10^{-27}\frac{kg}{u} \approx 970 \frac{N}{m}$$

http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibrot.html#c1

http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibspe.html

Her er resten av parametrene vi definerer:


```python
partial_charge = distr*e_charge # The partial charges for the atoms in HF
# https://chem.libretexts.org/Courses/Mount_Royal_University/Chem_1201/Unit_3%3A_Chemical_Bonding_I_-_Lewis_Theory/3.4%3A_Bond_Polarity
charge_H = partial_charge # Effective charge of the hydrogen
charge_F = -partial_charge # Effective charge of the fluorine
# F is more electronegative than H, so the electron is, on average, closer to the F

mass_H = 1.00794*u # Mass of hydrogen
mass_F= 18.9984*u # Mass of fluorine
#http://www.chemicalelements.com/elements

bond_length = 91.7e-12 # [m] the distance between H and F in HF
# http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibrot.html#c3
side_lengths = [2*grid_dims[0]*bond_length, 2*grid_dims[1]*bond_length]
# Determines the spacing of the molecules in x and y direction

k_spring = 970 #bond strength of HF
#http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibrot.html#c2
```

Vi kaller så på funksjonen vår som fant startpunktet til molekylene slik at vi kan bruke disse posisjonene videre i simulasjonen.


```python
# Set the initial position for all the atoms
pos_H, pos_F = make_liquid(num_molecules, dims, bond_length, grid_dims, side_lengths)
```

Deretter definerer vi en funksjon for det elekriske feltet:


```python
# electric field function
def Efield(r,qi,ri):
    epsilon_0 =  8.8541878128*10**(-12) #vakuum permitivitet
    const = 1/(4*np.pi*epsilon_0)
    Ri = r - ri
    Rinorm = np.linalg.norm(Ri)
    return const*qi*Ri/Rinorm**3

```

#### Krefter som virker på molekylene

I simulasjonen vår vil vi modellere molekylene med både kreftene som virker mellom atomene i molekylene og kreftene som virker mellom hvert molekyl.

Vi bruker harmonisk oscillatorpotenisal for å modellere bindingene mellom atomene i molekylene. Kraften mellom atomene kan dermed regnes ut ved å bruke Hooke's lov som gjort i [hyperphysics](http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibrot.html). Denne kraften er gitt som

$$
F(x) = k(x - x_0)
$$

hvor $k=k\_spring$ er fjærkonstanten, $x=spring\_distance$ er avstanden mellom de to atomene i molekylet og $x_0=bond\_length$ er likevektsavstanden til hydrogenfluorid. Fjærkonstanten er den samme konstanten som vi i stad fant da vi regnet ut bindingsstyrkekonstanten.

Den andre kraften vi modellerer er kraften mellom molekylene. Denne kraften er Coulombskraften. Kraften oppstår siden hydrogenatomet og fluoratomet i molekylet har motsatt delvis ladning og er gitt som

$$
F(x) = k_e \frac{q_H q_{HF}}{x^2}
$$

hvor $k_e$ er Coloumbs konstant, $q_H=+ \delta$ og $q_F = - \delta$ er ladningene til de to atomene, mens $\delta$ er den delvise ladningen.

Vi lager en funksjon som beregner bevegelsen til molekylene. Først i funksjonen definerer vi en rekke tomme arrays som vi fyller senere. Vi definerer så de to kreftene som virker på molekylene og integrerer disse numerisk ved å bruke Euler Cromer.


```python
def movement(pos_H,pos_F, t_steps, Qi, Ri, Ec):
    # Initialize numpy arrays
    position_H = np.zeros((t_steps, num_molecules, dims))
    position_F = np.zeros((t_steps, num_molecules, dims))
    matrix_shape = position_H.shape 
    # The other arrays (vel, acc, etc.) should be the same shape as the position arrays
    velocity_H = np.zeros(matrix_shape) # Velocity
    velocity_F = np.zeros(matrix_shape) # Velocity
    acceleration_H = np.zeros(matrix_shape) # Acceleration
    acceleration_F = np.zeros(matrix_shape) # Acceleration
    coulomb_acc_H = np.zeros(matrix_shape) # Intermolecular coulomb interaction
    coulomb_acc_F = np.zeros(matrix_shape) # Intermolecular coulomb interaction
    spring_acc_H = np.zeros(matrix_shape) # Intramolecular interaction
    spring_acc_F = np.zeros(matrix_shape) # Intramolecular interaction

    position_H[0], position_F[0] = pos_H, pos_F
    
    E_acc_H = np.zeros(matrix_shape) # Intramolecular interaction
    E_acc_F = np.zeros(matrix_shape) # Intramolecular interaction
    
    Ec_acc_H = np.zeros(matrix_shape) # Intramolecular interaction
    Ec_acc_F = np.zeros(matrix_shape) # Intramolecular interaction
    
    for t in range(t_steps-1):
        #-----------------------------------------------
        # Spring potential to simulate atom interactions in a molecule, the bond force, using for loops
        # This potential is a good approximations only when the atoms are Fose to the equilibrium, 
        # it does not allow the bondings to "break"
        for i in range(num_molecules):
            spring_difference = position_F[t,i] - position_H[t,i]
            # position of one atom minus the position of the other,
            # gives the vector from H to F
            spring_distance = np.linalg.norm(spring_difference)
            # Distance between the bound atoms
            spring_unitvector = spring_difference/spring_distance
            # Unit vector pointing from H to F (from F to H would then be its negative)
            spring_force = k_spring*(spring_distance - bond_length) 
            # Hooke's law, spring constant*distance from equilibrium
            spring_acc_H[t,i] = spring_unitvector*spring_force/mass_H #direction*force/mass
            spring_acc_F[t,i] = -spring_unitvector*spring_force/mass_F #direction*force/mass

        #-----------------------------------------------
        # Coulomb force between molecules
        for i in range(num_molecules):#num_molecules):
            coulomb_mask = np.arange(0,num_molecules)!=i #a mask to ignore atoms in the same molecule (same index)
            # Helps us in somewhat vectorizing the calculation

            coulomb_difference_H_H = position_H[t,i] - position_H[t,coulomb_mask] 
            #x and y difference between the hydrogen and other all other hydrogen atoms
            coulomb_difference_H_F = position_H[t,i] - position_F[t,coulomb_mask] 
            #x and y difference between the hydrogen and all chlorine atoms in other molecules
            coulomb_difference_F_H = position_F[t,i] - position_H[t,coulomb_mask] 
            #x and y difference between the chlorine and all hydrogen atoms in other molecules
            coulomb_difference_F_F = position_F[t,i] - position_F[t,coulomb_mask]
            #x and y difference between the hydrogen and other all other hydrogen atoms

            # Calculates the distance from one atom to num_molecules-1 other atoms
            coulomb_distance_H_H = np.linalg.norm(coulomb_difference_H_H, axis = 1) # The distances between the atoms
            coulomb_distance_H_F = np.linalg.norm(coulomb_difference_H_F, axis = 1)
            coulomb_distance_F_H = np.linalg.norm(coulomb_difference_F_H, axis = 1)
            coulomb_distance_F_F = np.linalg.norm(coulomb_difference_F_F, axis = 1)

            # Reshape distance from (num_molecules-1,) to (num_molecules-1, 1)
            # This gives us num_molecules-1 unit vectors from one atom to num_molecules-1 others
            coulomb_unitvector_H_H = coulomb_difference_H_H/coulomb_distance_H_H.reshape(-1,1)
            coulomb_unitvector_H_F = coulomb_difference_H_F/coulomb_distance_H_F.reshape(-1,1)
            coulomb_unitvector_F_H = coulomb_difference_F_H/coulomb_distance_F_H.reshape(-1,1)
            coulomb_unitvector_F_F = coulomb_difference_F_F/coulomb_distance_F_F.reshape(-1,1)

            # Calculate the force according to Coulomb's law
            coulomb_force_H_H = k_e*charge_H*charge_H*coulomb_unitvector_H_H/coulomb_distance_H_H.reshape(-1,1)**2
            coulomb_force_H_F = k_e*charge_H*charge_F*coulomb_unitvector_H_F/coulomb_distance_H_F.reshape(-1,1)**2
            coulomb_force_F_H = k_e*charge_F*charge_H*coulomb_unitvector_F_H/coulomb_distance_F_H.reshape(-1,1)**2
            coulomb_force_F_F = k_e*charge_F*charge_F*coulomb_unitvector_F_F/coulomb_distance_F_F.reshape(-1,1)**2

            # Sum together the forces from the different molecules acting upon atom i, and divide by mass to get acc
            coulomb_acc_H[t,i] = (np.sum(coulomb_force_H_H, axis = 0)\
                               + np.sum(coulomb_force_H_F, axis = 0))/mass_H
            coulomb_acc_F[t,i] = (np.sum(coulomb_force_F_H, axis = 0)\
                                + np.sum(coulomb_force_F_F, axis = 0))/mass_F
        if np.any(Qi) != 0:
            for i in range(num_molecules):
                E_acc_H[t,i]  = charge_H*Efield(position_H[t,i] ,Qi,Ri)/mass_H
                E_acc_F[t,i]  = charge_F*Efield(position_F[t,i] ,Qi,Ri)/mass_F
        
        if np.any(Ec) != 0:
            for i in range(num_molecules):
                Ec_acc_H[t,i]  = charge_H*Ec/mass_H
                Ec_acc_F[t,i]  = charge_F*Ec/mass_F
                
        #----------------------------------------------
        # Sum the accelerations
        acceleration_H[t] = coulomb_acc_H[t] + spring_acc_H[t] + E_acc_H[t] + Ec_acc_H[t]
        acceleration_F[t] = coulomb_acc_F[t] + spring_acc_F[t] + E_acc_F[t] + Ec_acc_F[t]
        
        # Euler Chromer
        velocity_H[t+1] = velocity_H[t] + acceleration_H[t]*dt
        velocity_F[t+1] = velocity_F[t] + acceleration_F[t]*dt
        position_H[t+1] = position_H[t] + velocity_H[t+1]*dt
        position_F[t+1] = position_F[t] + velocity_F[t+1]*dt

        if t!=0 and t%int(t_steps/20) == 0: # Every long for-loop needs some kind of "loading bar"
            print(t, 'of', t_steps)
    return position_H, position_F
```


```python
timeit_start = time.time()
position_H, position_F = movement(pos_H,pos_F, 900,0,0,0) #uten noe E-felt

timeit_stop = time.time()
print('It took %.1f seconds to simulate %i molecules for %i timesteps' \
      %(timeit_stop-timeit_start, num_molecules, 900))
```

    45 of 900
    90 of 900
    135 of 900
    180 of 900
    225 of 900
    270 of 900
    315 of 900
    360 of 900
    405 of 900
    450 of 900
    495 of 900
    540 of 900
    585 of 900
    630 of 900
    675 of 900
    720 of 900
    765 of 900
    810 of 900
    855 of 900
    It took 2.0 seconds to simulate 16 molecules for 900 timesteps
    

Vi lager så en figur av simulasjonen som viser initialposisjonene til molekylene og posisjonene til molekylene etter det er gått en viss tid.


```python
plt.subplot(121) #Plots the initial positions of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(position_H[0,:,0], position_H[0,:,1], '+r', lw=2)
plt.plot(position_F[0,:,0], position_F[0,:,1], 'b', marker = 1, linestyle = 'None')

plt.subplot(122) # Plots the final position of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(position_H[-1,:,0], position_H[-1,:,1], '+r', lw=2)
plt.plot(position_F[-1,:,0], position_F[-1,:,1], 'b', marker = 1, linestyle = 'None')

plt.show()
```


    
![png](output_34_0.png)
    


#### Forklaring av plott

Vi simulerer molekylene for 900 tidssteg og ser i figuren for sluttposisjonene at molekylene retter seg slik at den delvis positive ladningen til et molekyl og den delvis negative ladningen til et annet molekyl tiltrekker hverandre. Dette er på grunn av de Coloumbske kreftene som virker. Positiv ladning og negativ ladning tiltrekker hverandre som forventet. Denne motsatte tiltrekkingen med hydrogen involvert er det vi kaller hydrogenbindinger som beskrevet i teoridelen.

## I homogent E-felt

Et elektrisk felt beskriver de elektriske kreftene som virker i området og oppstår på grunn av elektriske ladninger. Et homogent elektrisk felt er et felt som har samme styrke og retning alle steder. Et eksempel er feltet som settes opp mellom to ladede plater med ulik ladning. Elektrisk felt er definert som

$$
\vec{E} = \frac{\vec{F}}{q} =\frac{Q}{4 \pi \epsilon_0 r^2}
$$

hvor $F$ er kraften som virker på en ladning $q$ i et punkt. Kraften $F$ kan skrives som $\vec{F} = \frac{qQ}{4 \pi \epsilon_0} \frac{\vec{R}}{r^3}$. Her er $q$ en ladning med posisjon $\vec{r}$ og $Q$ er en annen ladning med posisjon $\vec{r'}$. Avstanden $r$ er dermed $r = |\vec{R}| = |\vec{r}-\vec{r'}|$. $\epsilon_0$ er en konstant som vi kaller permitiviteten i vakuum.

https://snl.no/elektrisk_felt

Vi simulerer nå hvordan dipolene vil oppføre seg når vi setter på et ytre homogent elektrisk felt. Vi definerer først noen verdier som vi trenger. Vi må definere permitiviteten, $\epsilon_0$ i vakuum, lager en array med verdier for et elektrisk felt som virker i $x$-retning og kaller på funksjonen som gir oss bevegelsen til molekylene med det elektriske feltet som vi definerte.


```python
epsilon_0 =  8.8541878128*10**(-12)
Ec = np.array([35.5*1/(4*np.pi*epsilon_0),0]) #Efelt i x retning
position_H_n2, position_F_n2 = movement(pos_H,pos_F, 900,0,0, Ec)
```

    45 of 900
    90 of 900
    135 of 900
    180 of 900
    225 of 900
    270 of 900
    315 of 900
    360 of 900
    405 of 900
    450 of 900
    495 of 900
    540 of 900
    585 of 900
    630 of 900
    675 of 900
    720 of 900
    765 of 900
    810 of 900
    855 of 900
    

Vi lager så en figur av molekylene i det homogene elektriske feltet vi definerte. Vi lager totalt tre figurer der den første figuren, figuren til venstre, viser initialposisjonene til molekylene med en gang de plasseres i feltet, den andre figuren, figuren til høyre, viser posisjonene til molekylene etter de har vært i feltet en stund og plottet underst viser posisjonene til molekylene uten påvirkning av noe felt som vi fant tidligere etter det er gått samme tid.


```python
N = 10
xx = np.linspace(-side_lengths[0]/2, side_lengths[0]/2, N) # x dimension for meshgrid
yy = np.linspace(-side_lengths[1]/2, side_lengths[1]/2, N) # y dimension for meshgrid
rx, ry = np.meshgrid(xx,yy) # creates two meshgrids 

Ecx = np.full((N,N), Ec[0]) #plottet feltet (visualisere)
Ecy = np.full((N,N), Ec[1]) #plottet feltet (visualisere)

plt.subplot(121) #Plots the initial positions of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.quiver(rx,ry,Ecx,Ecy,color = 'grey')
plt.plot(position_H_n2[0,:,0], position_H_n2[0,:,1], '+r', lw=2)
plt.plot(position_F_n2[0,:,0], position_F_n2[0,:,1], 'b', marker = 1, linestyle = 'None')

plt.subplot(122) # Plots the final position of the molecules
plt.quiver(rx,ry,Ecx,Ecy,color = 'grey')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(position_H_n2[-1,:,0], position_H_n2[-1,:,1], '+r', lw=2)
plt.plot(position_F_n2[-1,:,0], position_F_n2[-1,:,1], 'b', marker = 1, linestyle = 'None')
plt.show()

plt.subplot(122) # Plots the final position of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(position_H[-1,:,0], position_H[-1,:,1], '+r', lw=2)
plt.plot(position_F[-1,:,0], position_F[-1,:,1], 'b', marker = 1, linestyle = 'None')

plt.show()
```


    
![png](output_42_0.png)
    



    
![png](output_42_1.png)
    


#### Forklaring av plott

Det elektriske feltet er visualisert med grå piler. Vi ser at det er homogent siden pilene er like store som viser oss at feltet har samme styrke over alt og de peker alle i samme retning. 

Umiddelbart når vi plasserer molekylene i feltet vil det ikke skje noe, men lar vi det gå litt tid ser vi at molekylene retter seg opp etter feltet. Siden feltet er homogent kan vi anta at det kommer fra at vi har en positiv plate på venstre side og en negativ plate på høyre side. Feltet peker nemlig den veien en positiv ladning ville bevegd seg som vil være fra den postitve platen til den negative platen (fra venstre til høyre). Molekylene retter seg opp slik at den positive delen av molekylene rettes mot den negative platen mens den negative delen av molekylene rettes mot den positive platen. Sammenligner vi posisjonene til molekylene i feltet etter de har rettet seg opp med posisjonene til molekylene uten påvirkning fra feltet men kun av kreftene mellom dem (figuren underst) ser vi at molekylene holder seg på omtrent samme posisjon, og roteres rundt denne posisjonen.

# I inhomogent felt - med ladning i midten

Vi simulerer hvordan molekylene vil bevege seg i et inhomogent felt ved å legge en ladning i midten. Denne ladningen vil da gi opphav til et elektrisk felt som peker radielt utover. Vi lar ladningen i midten ha samme ladning som den positive delen av HF-molekylet. Vi forventer at den negative delen av molekylene som ligger nærmest vil rette seg mot midten.


```python
#  visualisere feltet for ladning charge_H i orgio
NQ = 20
Q = charge_H
Ex = np.zeros((N,N),float)
Ey = np.zeros((N,N),float)

for i in range(len(rx.flat)):
    r = np.array([rx.flat[i],ry.flat[i],0])
    for j in range(NQ):
        theta = 2*np.pi/NQ*j
        rj = np.array([0,0,0])
        R = r - rj
        dq = Q/NQ
        dE = Efield(r,dq,rj)
        Ex.flat[i] = Ex.flat[i] + dE[0]
        Ey.flat[i] = Ey.flat[i] + dE[1]

```


```python
position_H_n, position_F_n = movement(pos_H,pos_F, 900, charge_H,0,0) #med felt fra ladning i midten
```

    45 of 900
    90 of 900
    135 of 900
    180 of 900
    225 of 900
    270 of 900
    315 of 900
    360 of 900
    405 of 900
    450 of 900
    495 of 900
    540 of 900
    585 of 900
    630 of 900
    675 of 900
    720 of 900
    765 of 900
    810 of 900
    855 of 900
    


```python
plt.subplot(121) #Plots the initial positions of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.quiver(rx,ry,Ex,Ey,color = 'grey')
plt.plot(position_H_n[0,:,0], position_H_n[0,:,1], '+r', lw=2)
plt.plot(position_F_n[0,:,0], position_F_n[0,:,1], 'b', marker = 1, linestyle = 'None')

plt.subplot(122) #Plots the final position of the molecules
plt.quiver(rx,ry,Ex,Ey,color = 'grey')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(position_H_n[-1,:,0], position_H_n[-1,:,1], '+r', lw=2)
plt.plot(position_F_n[-1,:,0], position_F_n[-1,:,1], 'b', marker = 1, linestyle = 'None')


plt.show()


plt.subplot(122) # Plots the final position of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(position_H[-1,:,0], position_H[-1,:,1], '+r', lw=2)
plt.plot(position_F[-1,:,0], position_F[-1,:,1], 'b', marker = 1, linestyle = 'None')

plt.show()
```


    
![png](output_48_0.png)
    



    
![png](output_48_1.png)
    


#### Forklaring av plott

Vi ser at etter molekylene blir påvirket av feltet (figuren øverst til høyre) vil de aller nærmeste molekylene snu seg slik at den negative delen peker mot den positive ladningen i midten. Vi ser at molekylene lengre vekk fra ladningen ikke vil være særlig påvirket, og vi ser også at feltstyrken fort blir veldig liten. Sammenligner vi med figuren under som viser hvordan molekylene beveger seg kun ved påvirkning av kreftene mellom dem ser vi at molekylene ytterst har omtrent samme posisjon i de to plottene. Kommer vi langt nok unna den positive ladningen vil altså feltet bli så lite at det kun er kreftene mellom atomene som virker.

## Bare elektrisk felt, ikke noe binding

Selv om resultatene virker lovende og vi ser at molekylene i figurene beveger seg slik vi forventer at de vil, ønsker vi å sjekke helt sikkert at det ikke er noe galt med koden vår som eventuelt gir oss falske riktige resultater. Vi ønsker derfor å simulere hvordan frie positive og negative ladninger ville oppført seg i et inhomoent felt fra en positiv ladning og bruker samme metode som vi brukte tidligere for molekylene. Vi lar de frie ladningene ha samme ladningsfordeling som de har i dipolene. Vi bruker i denne simulasjonen derimot en dobblet så stor ladning her for å se en forsterket effekt.


```python
timeit_start = time.time()
def E_acc(pos_H, pos_F, Qi, Ri,t_steps):
    
    # Initialize numpy arrays
    position_H = np.zeros((t_steps, num_molecules, dims))
    position_F = np.zeros((t_steps, num_molecules, dims))
    matrix_shape = position_H.shape 
    # The other arrays (vel, acc, etc.) should be the same shape as the position arrays
    velocity_H = np.zeros(matrix_shape) # Velocity
    velocity_F = np.zeros(matrix_shape) # Velocity
    acceleration_H = np.zeros(matrix_shape) # Acceleration
    acceleration_F = np.zeros(matrix_shape) # Acceleration

    position_H[0], position_F[0] = pos_H, pos_F

    E_acc_H = np.zeros(matrix_shape) # Intramolecular interaction
    E_acc_F = np.zeros(matrix_shape) # Intramolecular interaction
    
    for t in range(t_steps-1):                            
        # Electric force between molecules                          
        for i in range(num_molecules):#num_molecules):
            for k in range(len(qi)):
                E_acc_H[t,i]  += charge_H*Efield(position_H_n[t,i] ,qi[k],ri[k])/mass_H
                E_acc_F[t,i]  += charge_F*Efield(position_F_n[t,i] ,qi[k],ri[k])/mass_F

#         E_acc_H = 
#         E_acc_H[t]  = charge_H*Efield(position_H_n[t] ,charge_F,0)/mass_H
#         E_acc_F[t]  = charge_F*Efield(position_F_n[t] ,charge_F,0)/mass_F

        # Euler Chromer
        velocity_H[t+1] = velocity_H[t] +  E_acc_H[t]*dt
        velocity_F[t+1] = velocity_F[t] +  E_acc_F[t]*dt
        position_H[t+1] = position_H[t] + velocity_H[t+1]*dt
        position_F[t+1] = position_F[t] + velocity_F[t+1]*dt

        if t!=0 and t%int(t_steps/20) == 0: # Every long for-loop needs some kind of "loading bar"
            print(t, 'of', t_steps)
#             print(E_acc_H[t])
    return position_H, position_F

qi = np.array([2*charge_H])
ri = np.array([0,0])

pos_H_n, pos_F_n = E_acc(pos_H, pos_F, qi, ri, 500)

timeit_stop = time.time()
print('It took %.1f seconds to simulate %i molecules for %i timesteps' \
      %(timeit_stop-timeit_start, num_molecules, timesteps))

```

    25 of 500
    50 of 500
    75 of 500
    100 of 500
    125 of 500
    150 of 500
    175 of 500
    200 of 500
    225 of 500
    250 of 500
    275 of 500
    300 of 500
    325 of 500
    350 of 500
    375 of 500
    400 of 500
    425 of 500
    450 of 500
    475 of 500
    It took 0.3 seconds to simulate 16 molecules for 1000 timesteps
    


```python
plt.subplot(121) #Plots the initial positions of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.quiver(rx,ry,Ex,Ey)
plt.plot(pos_H_n[0,:,0], pos_H_n[0,:,1], '+r', lw=2)
plt.plot(pos_F_n[0,:,0], pos_F_n[0,:,1], 'b', marker = 1, linestyle = 'None')

plt.subplot(122) # Plots the final position of the molecules
plt.gca().set_aspect('equal', adjustable='box')
plt.quiver(rx,ry,Ex,Ey)
plt.plot(pos_H_n[-1,:,0], pos_H_n[-1,:,1], '+r', lw=2)
plt.plot(pos_F_n[-1,:,0], pos_F_n[-1,:,1], 'b', marker = 1, linestyle = 'None')

plt.show()
```


    
![png](output_54_0.png)
    


#### Forklaring av plott

Vi ser at vi kan stole på koden vår siden atomene beveger seg akkurat som vi ville forventet. De negative fluoratomene tltrekkes av den positive ladningen som er plassert i midten og de fri positive hydrogenatomene frastøtes.

### Diskusjon

Vi ser i den første simulasjonen at på grunn av kreftene som virker mellom dipolene så vil dipolene rette seg slik at den positive delen på hvert molekyl, hydrogenatomet, er rettet mot den negative delen på et annet molekyl, fluoratomet. Dette kommer som sagt av den Coulombske kraften som virker mellom dem. Vi vet at bindingen mellom dipolene er en sterk binding på grunn av hydrogenet som deltar i bindingen og gir en hydrogenbinding.

Da vi lot dipolene bli påvirket av et ytre homogent elektrisk felt så vi at dipolene rettet seg opp etter feltet. Den positive delen på alle dipolene pekte i motsatt retning av den negative delen på dipolene. Den poitive delen pekte i samme retning som feltet, og siden feltet peker i den retningen en positiv ladning ville bevegd seg gir det mening at den positive delen av dipolen peker denne veien. Den negative delen pekte i motsatt vei, altså mot feltretningen og siden det er den veien en negativ ladning ville bevegd seg i et elektrisk felt gir også dette mening.

Vi lot så dipolene bli påvirket av et ytre inhomoget elektrisk felt som vi modellerte ved å plassere en positiv ladning i midten. Denne ladningen lot vi være tilsvarende ladningen til den positive delen av dipolene. Vi så at for dipolene som lå nærmest ladningen i midten rettet den positive delen av dipolen seg vekk fra ladningen, så i samme retning som feltet, mens den negative delen rettet seg mot ladningen, altså i motsatt retning av feltet. Siden det var dette som skjedde også i det homogene elektriske feltet var dette som forventet. Det vi også så var at siden ladningen i midten er såpass liten vil feltet avta raskt radielt utover og det er bare de aller nærmeste molekylene som blir påvirket. De andre molekylene lengre unna vil være nærmest upåvirket av feltet og rette seg på samme måte som de gjorde upåvirket av feltet hvor den positive delen av dipolene retter seg mot den negative delen av en annen dipol.

For å teste om simulasjonen vår faktisk funker slik som den skal og ikke gir oss falske resultater brukte vi samme simulasjonsmetode men for frie ladninger som vi allerede vet hvordan oppfører seg. Vi så da tydelig at de negative ladningene samlet seg rundt den positive ladningen i midten, mens de positive ladningene ble frastøtet. Dette forsterker at simulasjonen vi gjorde for dipolene gir oss riktige resultater.

For styrke metoden vår videre kunne vi simulert for flere ulike felter, brukt andre tidssteg og simulert flere molekyler og sett at vi hadde fått samme effekten. Likvel vet vi at like ladninger frastøtes og ulike ladninger tiltrekkes så resultatene vi fikk virker nokså rimlige.

Koden i seg selv er fleksibel nok til å løpes med mange forskjellige homogene og inhomogene felt, f.eks. dipoler eller flere ladninger på rad.

### Konklusjon

Vi har funnet ut at dipoler som blir påvirket av et ytre homogent elektrisk felt vil rette seg opp etter feltet. Den positive delen av dipolen vil peke i samme retning som feltet og den negative delen vil peke i motsatt retning av feltet. Vi fant også at dipoler som er påvirket av et inhomogent felt vil reagere på omtrent samme vis. Den positive delen av dipolen vil peke i samme retning som feltet og den negative delen vil peke i motsatt retning. Denne effekten vil derimot avta med feltet.

### Referanser

Koden brukt til å simulere posisjonene til molekylene uten påvirkning av et ytre elektrisk felt er i stor grad basert på koden fra et skisseprosjekt med tittelen Polar Diatomic Molecules. Andre referanser finnes underveis i rapporten.
