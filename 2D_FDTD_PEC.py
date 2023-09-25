import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# universal constants
c_0 = 299792458

# simulation parameters
class SimulParam:
    def __init__(self, maxx, maxy, dx, dy, dt, freq):
        self.maxx = maxx    # number of grid points in x-direction
        self.maxy = maxy    # number of grid points in y-direction
        self.dx = dx        # grid spacing in x-direction
        self.dy = dy        # grid spacing in y-direction
        self.dt = dt        # time step
        self.freq = freq    # source freqeuncy
        self.lamb = c_0/freq    # wavelength (m)
        self.eps_r = np.ones((maxx,maxy))   # relative permittivity
        self.mu_r = np.ones((maxx,maxy))    # relative permeability

class mainFDTD:
    def __init__(self):
        print("running FDTD")
    def fdtd_2D(param, anim_save):
        Nx = param.maxx
        Ny = param.maxy
        dx = param.dx
        dy = param.dy
        dt = param.dt
        freq = param.freq
        lamb = param.lamb
        # initialize material
        eps_r = param.eps_r
        mu_r = param.mu_r
        # update coefficient
        mDz = c_0*dt
        #mEz = c_0*dt/eps_r
        mHx = c_0*dt/mu_r
        mHy = c_0*dt/mu_r

        # create the electric and magnetic fields
        dz = np.zeros((Nx, Ny))
        ez = np.zeros((Nx, Ny))
        hy = np.zeros((Nx, Ny))
        hx = np.zeros((Nx, Ny))
        cEx = np.float64(np.zeros((Nx, Ny)))
        cEy = np.float64(np.zeros((Nx, Ny)))
        cHz = np.float64(np.zeros((Nx, Ny)))
        # set up the source
        source_pos = Nx // 3  # source position
        source_amp = 1e-2  # source amplitude
        tau = 0.8/freq

        #prepare frame
        fig = plt.figure()
        frames = []

        # main FDTD simulation loop
        #ez only
        for t in np.arange(0, 1000*dt, dt):
            # update the magnetic field in spatial domain
            for nx in np.arange(0,Nx):
                for ny in np.arange(0,Ny-1):
                    cEx[nx,ny] = (ez[nx,ny+1] - ez[nx,ny])/dy
                cEx[nx,Ny-1] = (0.0 - ez[nx, Ny-1])/dy
            for ny in np.arange(0,Ny):
                for nx in np.arange(0,Nx-1):
                    cEy[nx,ny] = -(ez[nx+1,ny] - ez[nx,ny])/dx
                cEy[Nx-1 ,ny] = -(0.0 - ez[Nx-1, ny])/dx
            #update the hy and hx given cEx cEy
            hx[:, :] += -mHx *cEx[:, :] 
            hy[:, :] += -mHy *cEy[:, :]
        
            # apply Dirichlet boundary conditions
            #hy[:, -1] = hy[:, 0]

            # update the electric field
            #Curl of Hz
            cHz[0,0] = (hy[0,0]-0.0)/dx - (hx[0,0]-0.0)/dy
            for nx in np.arange(1,Nx):
                cHz[nx,0] = (hy[nx,0] - hy[nx-1,0])/dx - (hx[nx,0] - 0.0)/dy
            for ny in np.arange(1,Ny):
                cHz[0,ny] = (hy[0,ny]-0.0)/dx - (hx[0,ny]-hx[0,ny-1])/dy
                for nx in np.arange(1,Nx):    
                    cHz[nx,ny] = (hy[nx,ny]-hy[nx-1,ny])/dx - (hx[nx,ny] - hx[nx,ny-1])/dy
                    
            dz[:, :] += mDz * cHz[:, :] 

            # later H -> E to H -> D -> E
            ez[:, :] = dz[:, :]/eps_r

            # apply the source
            ez[ source_pos, source_pos] += source_amp * np.exp(-(t-2*tau/4)**2/(tau/4)**2) #* np.sin(2*np.pi*freq*t)
            
            #frame save
            
            frames.append([plt.imshow(ez.transpose(),cmap='jet', origin='lower', extent=[0,Nx*dx,0,Ny*dy], interpolation='bilinear', animated=True)])
# cmap, 'seismic', 'jet'
        # 
        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
        plt.colorbar()
        # save the animation as an mp4 file
        
        if anim_save == 1:
            ani.save('FDTD_motion.mp4', writer='ffmpeg', fps=30)

        plt.show()


if __name__ == "__main__":
    param = SimulParam(75,75,0.01,0.01,0.2e-10,1e9)
    mainFDTD.fdtd_2D(param,0) # 1 for animation save, 0 for no save

