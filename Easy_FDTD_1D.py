import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from scipy.fft import fft, ifft, fftfreq, fftshift

# universal constants
c_0 = 299792458

# simulation parameters
class SimulParam1d:
    def __init__(self, maxN, dz, dt, freq):
        self.maxN = maxN    # number of grid points in x-direction
        self.dz = dz        # grid spacing in z-direction
        self.dt = dt        # time step
        self.freq = freq    # source freqeuncy
        self.lamb = c_0/freq    # wavelength (m)
        self.eps_r = np.ones(maxN)   # relative permittivity
        self.mu_r = np.ones(maxN)    # relative permeability = free space
        for xn in np.arange(maxN//2,maxN*3//4):
            self.eps_r[xn] += 1.5
 
def source_f(source_amp, tau, t):
    return -source_amp * np.exp(-(t-6*tau)**2/tau**2) #* np.sin(2*np.pi*3e9*t)

class mainFDTD:
    def __init__(self):
        print("running FDTD")
    def fdtd_1D(i, param, line, line2, line3, line4):
        Nz = param.maxN
        dz = param.dz
        dt = param.dt
        freq = param.freq
        #lamb = c_0/freq
        # initialize material
        eps_r = param.eps_r
        mu_r = param.mu_r
        # update coefficient
        #mDz = c_0*dt
        #mEx = c_0*dt/eps_r
        mEy = c_0*dt/eps_r
        mHx = c_0*dt/mu_r
        #mHy = c_0*dt/mu_r

        # create the electric and magnetic fields
        #ex = np.zeros(Nz)
        ey = np.zeros(Nz)
        #hy = np.zeros(Nz)
        hx = np.zeros(Nz)
        #cEx = np.float64(np.zeros(Nz))
        cEy = np.zeros(Nz)
        cHx = np.zeros(Nz)
        #cHy = np.zeros(Nz)
        #cHz = np.float64(np.zeros((Nx, Ny)))
        # set up the source
        source_pos = Nz // 6  # source position
        source_amp = 1e-2  # source amplitude
        tau = 0.5/freq

        # 1D Perfect Matched Layer prep
        h3 = 0.0
        h2 = 0.0
        h1 = 0.0
        e3 = 0.0
        e2 = 0.0
        e1 = 0.0
        #i = 0
        # main FDTD simulation loop
        # ey hx only
        
        # frequency range 0 - 4 GHz, fN freqeuncys
        fN = 256
        f_range = np.linspace(0, 4e9,fN)
        kerK = np.exp(-1j* 2*np.pi * dt * f_range)
        ref_R = np.ones(fN) *(0.0+0.0j)
        tr_T = np.ones(fN) * (0.0+0.0j)
        src_I = np.ones(fN) * (0.0+0.0j)

        for t in np.arange(0, i*dt, dt):
            # 1. Dirichlet boundary conditions
            # 2. Perfect Matched Layer. 

            # update the magnetic field in spatial domain
            
            for nz in np.arange(0,Nz-1):
                cEy[nz] = (ey[nz+1] - ey[nz])/dz    
            cEy[Nz-1] = (e3 - ey[Nz-1])/dz

            #for nz in np.arange(0,Nz-1):
            #    cEx[nz] = (ex[nz+1]-ex[nz])/dz    
            #cEx[Nz-1] = (0.0 - ex[Nz-1])/dz
             
            #update the hy and hx given cEx cEy
            hx[:] += mHx * cEy[:]
            
            # apply the source TF/SF
            hx[source_pos-1] = hx[source_pos-1] - mHx[source_pos-1]/dz * source_f(source_amp,tau,t)
            h3 = h2
            h2 = h1
            h1 = hx[0]
            

            #hy[:] += -mHy * cEx[:]

            # update the electric field
            #Curl of Hz
            cHx[0] = (hx[0] - h3)/dz 
            for nz in np.arange(1,Nz):
                cHx[nz] = (hx[nz] - hx[nz-1])/dz 
            ey[:] += mEy[:] * cHx[:] 

            # Ex Hy
            #cHy[0] = (hy[0] - 0.0)/dz 
            #for nz in np.arange(1,Nz):
            #    cHy[nz] = (hy[nz] - hy[nz-1])/dz 

            #ex[:] += -mEx * cHy[:]
            # apply the source TF/SF
            ey[source_pos] -= mEy[source_pos]/dz * -1 * source_f(source_amp, tau, t+dt/2+dz/(2*c_0)) 
            e3 = e2
            e2 = e1
            e1 = ey[Nz-1]
            
            # Fourier Transform
            for nf in np.arange(0,fN):
                ref_R[nf] += (kerK[nf]**(t//dt+0)) * ey[1]
                tr_T[nf] += (kerK[nf]**(t//dt+0)) * ey[Nz-2]
                src_I[nf] += (kerK[nf]**(t//dt+0)) * source_amp * np.exp(-(t-6*tau)**2/tau**2)
                if src_I[nf] == 0.0 + 0.0j:
                    src_I[nf] = 1e-100
            #frame save
            #ax.set_xlim((0.0,1.0))
            x = np.arange(0,dz*Nz,dz)
            line.set_xdata(x)
            line.set_ydata(ey)
            line2.set_xdata(x)
            line2.set_ydata(hx)
            #frames.append([lines[i]])
            
        #ref_R = ref_R *dt  (more accurate FT)
        #tr_T = tr_T *dt    (but will be normlized with src_I anyway, *dt not needed for this demonstration)
        #src_I = src_I *dt
        for nf in np.arange(0,fN):
            if src_I[nf] == 0.0 + 0.0j:
                src_I[nf] = 1e-100 
        line3.set_xdata(f_range)
        line3.set_ydata((np.abs(tr_T)/np.abs(src_I))**2)
        line4.set_xdata(f_range)
        line4.set_ydata((np.abs(ref_R)/np.abs(src_I))**2)
        #print(eps_r)
        return line, line2, line3, line4


if __name__ == "__main__":
    param = SimulParam1d(200,0.005,0.005/(2*c_0), 5e9)
    fig, (ax1,ax2) = plt.subplots(2,1)
    line, = ax1.plot([], 'b-')
    line2, = ax1.plot([], 'r-')
    line3, = ax2.plot([], 'b-')
    line4, = ax2.plot([], 'r-')
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(-0.022,0.022)
    ax1.axvline(x=0.5, linestyle = 'dashed')
    ax1.axvline(x=0.75, linestyle='dashed')
    ax1.set_xlabel('Dist (m)')
    ax2.set_xlim(0.0, 4e9)
    ax2.set_ylim(-0.1, 1.4)
    ax2.set_xlabel('Freq (GHz)')
    ani = animation.FuncAnimation(fig, mainFDTD.fdtd_1D, frames = 1000, fargs = (param, line, line2, line3, line4) ,interval = 25)
    
    ani.save('FDTD_1Dmotion5G.mp4', writer = 'ffmpeg', fps=30)
    plt.show()