
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def run_plot():
    fig, axes = plt.subplots(2, 2)
    M, Sus, Bind = [], [], []
    Tc = 4.5082

    try:
        Energy1, Cv1, Binder1, Suscept1, Mag1, t_span = torch.load('monte_data_8cubed200k(22,)temps1'+'.pth')
        Energy2, Cv2, Binder2, Suscept2, Mag2, t_span = torch.load('monte_data_8cubed200k(22,)temps2'+'.pth')
        Energy3, Cv3, Binder3, Suscept3, Mag3, t_span = torch.load('monte_data_8cubed100k(22,)temps3'+'.pth')
        Energy4, Cv4, Binder4, Suscept4, Mag4, t_span = torch.load('monte_data_8cubed200k(22,)temps4'+'.pth')
        Energy5, Cv5, Binder5, Suscept5, Mag5, t_span = torch.load('monte_data_8cubed300k(22,)temps5'+'.pth')
        t_span=np.asarray(t_span)
        
        Energy = (np.asarray(Energy1)*2+np.asarray(Energy2)*2+np.asarray(Energy3)+np.asarray(Energy4)*2+np.asarray(Energy5)*3)/10
        Cv =  (np.asarray(Cv1)*2+np.asarray(Cv2)*2+np.asarray(Cv3)+np.asarray(Cv4)*2+np.asarray(Cv5)*3)/10
        Binder = (np.asarray(Binder1)*2+np.asarray(Binder2)*2+np.asarray(Binder3)+np.asarray(Binder4)*2+np.asarray(Binder5)*3)/10
        Suscept = (np.asarray(Suscept1)*2+np.asarray(Suscept2)*2+np.asarray(Suscept3)+np.asarray(Suscept4)*2+np.asarray(Suscept5)*3)/10
        Mag = (np.asarray(Mag1)*2+np.asarray(Mag2)*2+np.asarray(Mag3)+np.asarray(Mag4)*2+np.asarray(Mag5)*3)/10

        Bind.append(Binder)
        Sus.append(Suscept)
        M.append(Mag)
        axes[0, 0].plot(t_span, Energy, 'o--') # E
        axes[0, 1].plot(t_span, Cv, 'o--') # Cv
        axes[1, 0].plot(t_span, Binder, 'o--') # Binder
        axes[1, 1].plot(t_span, Suscept, 'o--') # Susceptibility

    except FileNotFoundError:
        print()

    try:
        Energy1, Cv1, Binder1, Suscept1, Mag1, t_span = torch.load('monte_data_16cubed200k(22,)temps1'+'.pth')
        Energy2, Cv2, Binder2, Suscept2, Mag2, t_span = torch.load('monte_data_16cubed200k(22,)temps2'+'.pth')
        Energy3, Cv3, Binder3, Suscept3, Mag3, t_span = torch.load('monte_data_16cubed100k(22,)temps3'+'.pth')
        Energy4, Cv4, Binder4, Suscept4, Mag4, t_span = torch.load('monte_data_16cubed200k(22,)temps4'+'.pth')
        Energy5, Cv5, Binder5, Suscept5, Mag5, t_span = torch.load('monte_data_16cubed300k(22,)temps5'+'.pth')
        t_span=np.asarray(t_span)
        
        Energy = (np.asarray(Energy1)*2+np.asarray(Energy2)*2+np.asarray(Energy3)+np.asarray(Energy4)*2+np.asarray(Energy5)*3)/10
        Cv =  (np.asarray(Cv1)*2+np.asarray(Cv2)*2+np.asarray(Cv3)+np.asarray(Cv4)*2+np.asarray(Cv5)*3)/10
        Binder = (np.asarray(Binder1)*2+np.asarray(Binder2)*2+np.asarray(Binder3)+np.asarray(Binder4)*2+np.asarray(Binder5)*3)/10
        Suscept = (np.asarray(Suscept1)*2+np.asarray(Suscept2)*2+np.asarray(Suscept3)+np.asarray(Suscept4)*2+np.asarray(Suscept5)*3)/10
        Mag = (np.asarray(Mag1)*2+np.asarray(Mag2)*2+np.asarray(Mag3)+np.asarray(Mag4)*2+np.asarray(Mag5)*3)/10
        
        Bind.append(Binder)
        Sus.append(Suscept)
        M.append(Mag)
        axes[0, 0].plot(t_span, Energy, '^--') # E
        axes[0, 1].plot(t_span, Cv, '^--') # Cv
        axes[1, 0].plot(t_span, Binder, '^--') # Binder
        axes[1, 1].plot(t_span, Suscept, '^--') # Susceptibility
    except FileNotFoundError:
        print()

    try:
        Energy1, Cv1, Binder1, Suscept1, Mag1, t_span = torch.load('monte_data_32cubed200k(22,)temps1'+'.pth')
        Energy2, Cv2, Binder2, Suscept2, Mag2, t_span = torch.load('monte_data_32cubed200k(22,)temps2'+'.pth')
        Energy3, Cv3, Binder3, Suscept3, Mag3, t_span = torch.load('monte_data_32cubed100k(22,)temps3'+'.pth')
        Energy4, Cv4, Binder4, Suscept4, Mag4, t_span = torch.load('monte_data_32cubed200k(22,)temps4'+'.pth')
        Energy5, Cv5, Binder5, Suscept5, Mag5, t_span = torch.load('monte_data_32cubed300k(22,)temps5'+'.pth')
        t_span=np.asarray(t_span)
        
        Energy = (np.asarray(Energy1)*2+np.asarray(Energy2)*2+np.asarray(Energy3)+np.asarray(Energy4)*2+np.asarray(Energy5)*3)/10
        Cv =  (np.asarray(Cv1)*2+np.asarray(Cv2)*2+np.asarray(Cv3)+np.asarray(Cv4)*2+np.asarray(Cv5)*3)/10
        Binder = (np.asarray(Binder1)*2+np.asarray(Binder2)*2+np.asarray(Binder3)+np.asarray(Binder4)*2+np.asarray(Binder5)*3)/10
        Suscept = (np.asarray(Suscept1)*2+np.asarray(Suscept2)*2+np.asarray(Suscept3)+np.asarray(Suscept4)*2+np.asarray(Suscept5)*3)/10
        Mag = (np.asarray(Mag1)*2+np.asarray(Mag2)*2+np.asarray(Mag3)+np.asarray(Mag4)*2+np.asarray(Mag5)*3)/10

        Bind.append(Binder)
        Sus.append(Suscept)
        M.append(Mag)
        axes[0, 0].plot(t_span, Energy, '4--') # E
        axes[0, 1].plot(t_span, Cv, '4--') # Cv
        axes[1, 0].plot(t_span, Binder, '4--') # Binder
        axes[1, 1].plot(t_span, Suscept, '4--') # Susceptibility

    except FileNotFoundError:
        print()

    try:
        Energy1, Cv1, Binder1, Suscept1, Mag1, t_span = torch.load('monte_data_48cubed200k(22,)temps1'+'.pth')
        Energy2, Cv2, Binder2, Suscept2, Mag2, t_span = torch.load('monte_data_48cubed200k(22,)temps2'+'.pth')
        Energy3, Cv3, Binder3, Suscept3, Mag3, t_span = torch.load('monte_data_48cubed100k(22,)temps3'+'.pth')
        Energy4, Cv4, Binder4, Suscept4, Mag4, t_span = torch.load('monte_data_48cubed200k(22,)temps4'+'.pth')
        Energy5, Cv5, Binder5, Suscept5, Mag5, t_span = torch.load('monte_data_48cubed300k(22,)temps5'+'.pth')
        t_span=np.asarray(t_span)
        
        Energy = (np.asarray(Energy1)*2+np.asarray(Energy2)*2+np.asarray(Energy3)+np.asarray(Energy4)*2+np.asarray(Energy5)*3)/10
        Cv =  (np.asarray(Cv1)*2+np.asarray(Cv2)*2+np.asarray(Cv3)+np.asarray(Cv4)*2+np.asarray(Cv5)*3)/10
        Binder = (np.asarray(Binder1)*2+np.asarray(Binder2)*2+np.asarray(Binder3)+np.asarray(Binder4)*2+np.asarray(Binder5)*3)/10
        Suscept = (np.asarray(Suscept1)*2+np.asarray(Suscept2)*2+np.asarray(Suscept3)+np.asarray(Suscept4)*2+np.asarray(Suscept5)*3)/10
        Mag = (np.asarray(Mag1)*2+np.asarray(Mag2)*2+np.asarray(Mag3)+np.asarray(Mag4)*2+np.asarray(Mag5)*3)/10

        Bind.append(Binder)
        Sus.append(Suscept)
        M.append(Mag)
        axes[0, 0].plot(t_span, Energy, 's--') # E
        axes[0, 1].plot(t_span, Cv, 's--') # Cv
        axes[1, 0].plot(t_span, Binder, 's--') # Binder
        axes[1, 1].plot(t_span, Suscept, 's--') # Susceptibility

    except FileNotFoundError:
        print()

    try:
        Energy1, Cv1, Binder1, Suscept1, Mag1, t_span = torch.load('monte_data_64cubed200k(22,)temps1'+'.pth')
        Energy2, Cv2, Binder2, Suscept2, Mag2, t_span = torch.load('monte_data_64cubed200k(22,)temps2'+'.pth')
        Energy3, Cv3, Binder3, Suscept3, Mag3, t_span = torch.load('monte_data_64cubed100k(22,)temps3'+'.pth')
        Energy4, Cv4, Binder4, Suscept4, Mag4, t_span = torch.load('monte_data_64cubed200k(22,)temps4'+'.pth')
        Energy5, Cv5, Binder5, Suscept5, Mag5, t_span = torch.load('monte_data_64cubed300k(22,)temps5'+'.pth')
        t_span=np.asarray(t_span)
        
        Energy = (np.asarray(Energy1)*2+np.asarray(Energy2)*2+np.asarray(Energy3)+np.asarray(Energy4)*2+np.asarray(Energy5)*3)/10
        Cv =  (np.asarray(Cv1)*2+np.asarray(Cv2)*2+np.asarray(Cv3)+np.asarray(Cv4)*2+np.asarray(Cv5)*3)/10
        Binder = (np.asarray(Binder1)*2+np.asarray(Binder2)*2+np.asarray(Binder3)+np.asarray(Binder4)*2+np.asarray(Binder5)*3)/10
        Suscept = (np.asarray(Suscept1)*2+np.asarray(Suscept2)*2+np.asarray(Suscept3)+np.asarray(Suscept4)*2+np.asarray(Suscept5)*3)/10
        Mag = (np.asarray(Mag1)*2+np.asarray(Mag2)*2+np.asarray(Mag3)+np.asarray(Mag4)*2+np.asarray(Mag5)*3)/10

        Bind.append(Binder)
        Sus.append(Suscept)
        M.append(Mag)
        axes[0, 0].plot(t_span, Energy, 'x--') # E
        axes[0, 1].plot(t_span, Cv, 'x--') # Cv
        axes[1, 0].plot(t_span, Binder, 'x--') # Binder
        axes[1, 1].plot(t_span, Suscept, 'x--') # Susceptibility

        #axes[0, 0].set_title('Expected energy')
        axes[0, 0].set_xlabel('$\\tau$', fontsize=13)
        axes[0, 0].set_ylabel('<E>', fontsize=13)
        axes[0, 0].legend(['N=8','N=16','N=32','N=48','N=64'])

        #axes[0, 1].set_title('Specific heat')
        axes[0, 1].set_xlabel('$\\tau$', fontsize=13)
        axes[0, 1].set_ylabel('$C_v$', fontsize=13)
        axes[0, 1].legend(['N=8','N=16','N=32','N=48','N=64'])

        #axes[1, 0].set_title('Binder cumilant')
        axes[1, 0].set_xlabel('$\\tau$', fontsize=13)
        axes[1, 0].set_ylabel('g', fontsize=13)
        axes[1, 0].legend(['N=8','N=16','N=32','N=48','N=64'])

        #axes[1, 1].set_title('Susceptibility')
        axes[1, 1].set_xlabel('$\\tau$', fontsize=13)
        axes[1, 1].set_ylabel('$\\chi$', fontsize=13)
        axes[1, 1].legend(['N=8','N=16','N=32','N=48','N=64'])

    except FileNotFoundError:
        print()
    
    plt.figure(2)

    try:
        plt.plot(t_span,M[0],'o--')
    except IndexError:
        print()
    try:
        plt.plot(t_span,M[1],'^--')
    except IndexError:
        print()
    try:
        plt.plot(t_span,M[2],'4--')
    except IndexError:
        print()
    try:
        plt.plot(t_span,M[3],'s--')
    except IndexError:
        print()
    try:
        plt.plot(t_span,M[4],'x--')
        plt.xlabel('$\\tau$', fontsize=13)
        plt.ylabel('<M>', fontsize=13)
        plt.legend(['N=8','N=16','N=32','N=48','N=64'])
    except IndexError:
        print()

    v = 0.63
    fig1, axes1 = plt.subplots(1, 2)

    try:
        axes1[0].plot((8**(1/v))*(t_span-Tc)/Tc,Bind[0],'o')
    except IndexError:
        print()
    try:
        axes1[0].plot((16**(1/v))*(t_span-Tc)/Tc,Bind[1],'^')
    except IndexError:
        print()
    try:
        axes1[0].plot((32**(1/v))*(t_span-Tc)/Tc,Bind[2],'4')
    except IndexError:
        print()
    try:
        axes1[0].plot((48**(1/v))*(t_span-Tc)/Tc,Bind[3],'s')
    except IndexError:
        print()
    try:
        axes1[0].plot((64**(1/v))*(t_span-Tc)/Tc,Bind[4],'x')
        axes1[0].legend(['N=8','N=16','N=32','N=48','N=64'])
        axes1[0].set_xlabel(r'$L^{\frac{1}{\nu}}[\frac{\tau-T_c}{T_c}]$', fontsize=15)
        axes1[0].set_ylabel('g', fontsize=13)
    except IndexError:
        print()
    
    gamma = -1.285

    try:
        axes1[1].plot((8**(1/v))*(t_span-Tc)/Tc,(8**(gamma/v))*Sus[0],'o')
    except IndexError:
        print()
    try:
        axes1[1].plot((16**(1/v))*(t_span-Tc)/Tc,(16**(gamma/v))*Sus[1],'^')
    except IndexError:
        print()
    try:
        axes1[1].plot((32**(1/v))*(t_span-Tc)/Tc,(32**(gamma/v))*Sus[2],'4')
    except IndexError:
        print()
    try:
        axes1[1].plot((48**(1/v))*(t_span-Tc)/Tc,(48**(gamma/v))*Sus[3],'s')
    except IndexError:
        print()
    try:
        axes1[1].plot((64**(1/v))*(t_span-Tc)/Tc,(64**(gamma/v))*Sus[4],'x')
        axes1[1].legend(['N=8','N=16','N=32','N=48','N=64'])
        axes1[1].set_xlabel(r'$L^{\frac{1}{\nu}}[\frac{\tau-T_c}{T_c}]$', fontsize=15)
        axes1[1].set_ylabel(r'$L^{\frac{\gamma}{\nu}} \chi$', fontsize=14)
    except IndexError:
        print()
    
    plt.show()

def bench_plot():
    numb, numb_para, t = torch.load('numba_bench.pth')
    #numb_lap, numb_para_lap, t2 = torch.load('numba_bench_laptop.pth')
    single = torch.load('python_single_bench.pth')
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].semilogy(t, numb)
    ax[0, 0].semilogy(t, numb_para)
    try:
        ax[0, 0].semilogy(t2, numb_lap)
        ax[0, 0].semilogy(t2, numb_para_lap)
    except NameError:
        print()
    ax[0, 0].semilogy(t[:len(single)], single)
    ax[0, 0].set_ylabel('Seconds per sweep')
    ax[0, 0].set_xlabel('N')
    ax[0, 0].legend(['Numba','Numba Parallel','Python'])
    ax[0, 0].grid()

    ax[0, 1].semilogy(t, t**3/numb)
    ax[0, 1].semilogy(t, t**3/numb_para)
    try:
        ax[0, 1].semilogy(t2, t2**3/numb_lap)
        ax[0, 1].semilogy(t2, t2**3/numb_para_lap)
    except NameError:
        print()
    ax[0, 1].semilogy(t[:len(single)], t[:len(single)]**3/single)
    ax[0, 1].set_ylabel('Latice cells calculated per second')
    ax[0, 1].set_xlabel('N')
    ax[0, 1].legend(['Numba','Numba Parallel','Python'])
    ax[0, 1].grid()

    ax[1, 0].semilogy(t[:len(single)], (t[:len(single)]**3/numb[:len(single)])/(t[:len(single)]**3/single))
    ax[1, 0].semilogy(t[:len(single)], (t[:len(single)]**3/numb_para[:len(single)])/(t[:len(single)]**3/single))
    try:
        ax[1, 0].semilogy(t[:len(single)], (t[:len(single)]**3/numb_lap[:len(single)])/(t[:len(single)]**3/single))
        ax[1, 0].semilogy(t[:len(single)], (t[:len(single)]**3/numb_para_lap[:len(single)])/(t[:len(single)]**3/single))
    except NameError:
        print()
    ax[1, 0].set_ylabel('x times faster than python')
    ax[1, 0].set_xlabel('N')
    ax[1, 0].legend(['Numba vs. Python','Numba Parallel vs. Python'])
    ax[1, 0].grid()

    ax[1, 1].plot(t, (t**3/numb_para)/(t**3/numb))
    try:
        ax[1, 1].plot(t2, (t2**3/numb_para_lap)/(t2**3/numb_lap))
    except NameError:
        print()
    ax[1, 1].set_ylabel('x times faster')
    ax[1, 1].set_xlabel('N')
    ax[1, 1].legend(['Parallel vs. Singel'])
    ax[1, 1].grid()

    plt.show()


#run_plot()
bench_plot()