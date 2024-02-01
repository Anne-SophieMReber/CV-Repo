import numpy as np
import matplotlib.pyplot as plt
import pyarma as pa
from matplotlib.animation import FuncAnimation
import matplotlib

#imports cube and converts to numpy matrix 
cube_arma = pa.cx_cube()
cube_arma.load("U_whole_cube.bin") #filename of automatic output
cube_numpy = np.array(cube_arma)
complex_data = np.transpose(cube_numpy)
data = (np.conjugate(complex_data)*complex_data).real
print(data.shape) #(x,y,t)


def animate():
    #animates x-y-plane over all timesteps, the code originally from Computational Physics FYS-3150 and has been altered
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    # Create figure
    fig = plt.figure()
    ax = plt.gca()

    # Create a colour scale normalization according to the max z value in the first frame
    norm = matplotlib.cm.colors.Normalize(vmin=0.0, vmax=np.max(data[:,:,0]))

    # # Plot the first frame
    img = ax.imshow(data[:,:,0], extent=[x_min,x_max,y_min,y_max], cmap=plt.get_cmap("viridis"), norm=norm)
    # img = ax.imshow(data[:,:,0], cmap=plt.get_cmap("viridis"), norm=norm)

    # Axis labels
    fontsize = 12
    plt.xlabel("x", fontsize=fontsize)
    plt.ylabel("y", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Add a colourbar
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("probability density", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Add a text element showing the time
    time_txt = plt.text(0.95, 0.95, "t = {:.3e}".format(0.00), color="white", 
                        horizontalalignment="right", verticalalignment="top", fontsize=fontsize)

    # Function that takes care of updating the z data and other things for each frame
    dt = 1/len(data[0,0,:])

    def animation(i):
        # Normalize the colour scale to the current frame?
        # norm = matplotlib.cm.colors.Normalize(vmin=0.0, vmax=np.max(data[:,:,i]))
        img.set_norm(norm)

        # Update z data
        img.set_data(data[:,:,i])

        # Update the time label
        current_time = i * dt
        time_txt.set_text("t = {:.3e}".format(current_time))

        return img

    # # Use matplotlib.animation.FuncAnimation to put it all together
    anim = FuncAnimation(fig, animation, interval=50, frames=np.arange(0, len(data[0,0,:])), repeat=True, blit=0)
    #anim.save("animation.mp4") #saves animation as .mp4 file

    # # Run the animation!
    plt.show()


# animate()

def screen():
    #plots probability along x at y = 159 at timestep 79
    screen = data[:,159,79]
    tot_sum = np.sum(screen)
    probability = screen/tot_sum
    

    x = np.linspace(0,1,len(probability))

    plt.plot(x, probability)
    plt.grid()

    fontsize = 18
    
    plt.ylabel("Probability density", fontsize = fontsize)
    plt.xlabel("Position on y-axis", fontsize = fontsize)
    #plt.savefig("prob_dist_screen.pdf") #saves figure as pdf
    plt.show()


screen()


def colormap():
    #plots probability over x-y-plane for timesteps 0, 39 and 79

    U_1 = complex_data[:,:,0]
    U_2 = complex_data[:,:,39]
    U_3 = complex_data[:,:,79]

    U = np.array((U_1,U_2,U_3))

    prob = np.abs(U)       # square root of probability = norm of u

    U_real = U.real
    U_imag = U.imag


    M = int(1/0.005)
    X, Y = np.meshgrid(np.linspace(0,1,M), np.linspace(0,1,M))

    ticks_location = np.arange(0, 198, 24.75)
    ticks_location = np.append(ticks_location, [197])
    ticks_label = np.arange(0, 1, 0.125)
    ticks_label = np.append(ticks_label, [1])
    ticks_label = ticks_label.astype(str)

    for i in range(3):


        plt.imshow(prob[i])
        plt.xlabel(r"$x_i$", fontsize=18)
        plt.ylabel(r"$y_j$", fontsize=18)
        plt.xticks(ticks_location, ticks_label)
        plt.yticks(ticks_location, ticks_label[::-1])
        plt.colorbar().ax.set_title(r"$\sqrt{p_{ij}}$", size=18)

        plt.savefig(f"Prob_colmap_{str(i)}.pdf")
        plt.show()
        plt.close()


        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        img1 = axes[0].imshow(U_real[i])
        axes[0].set_xlabel(r"$x_i$", fontsize=18)
        axes[0].set_ylabel(r"$y_j$", fontsize=18)
        axes[0].set_xticks(ticks_location, ticks_label)
        axes[0].set_yticks(ticks_location, ticks_label[::-1])
        plt.colorbar(img1, ax=axes[0]).ax.set_title(r"Re($u_{ij}$)",size=18)

        img2 = axes[1].imshow(U_imag[i])
        axes[1].set_xlabel(r"$x_i$", fontsize=18)
        axes[1].set_ylabel(r"$y_j$", fontsize=18)
        axes[1].set_xticks(ticks_location, ticks_label)
        axes[1].set_yticks(ticks_location, ticks_label[::-1])
        plt.colorbar(img2, ax=axes[1]).ax.set_title(r"Im($u_{ij}$)",size=18)

        plt.savefig(f"Re_Im_colmap_{str(i)}.pdf")
        plt.show()


colormap()



def normalization_test(filename, label):
    #plots float(probability of total x-y-plane) for each timestep t
    A = pa.mat()
    A.load(filename)
    A_np = np.array(A)

    t = np.linspace(0,len(A_np),len(A_np))

    plt.plot(t,(A_np-1)/1e-14,"--.", label = label)
    plt.legend(fontsize = 16)
    plt.xlabel("time steps", fontsize = 16)
    plt.ylabel("deviation of probability [1e-14]", fontsize = 16)
    print((np.max(abs(A_np-1)))/1e-14)
    #plt.savefig("probability_consistancy.pdf") #saves as 
    plt.grid()
    plt.show()

normalization_test("probability_simulation.bin", "label") #filename of automatic output

