# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 2025

@author Gyutae Park

@Original author: Seokyeong Lee

Created on Mon May 13 17:39:51 2019

"""
import numpy as np
import matplotlib.pyplot as plt
import kwant
           
class gate: # any external effects on DUT (gates, bias, magnetic field, etc.)
    gates = [] # list of gates
    others = [] # bias, magnetic field, whatever
    all_gates = [] # potential energy of DUT in eV for all gates at 1 V, made by gate.pot_all()
#    SIMdim = None # grid in nm
#    gate_shape_basis = None
    gate_shape_fig,gate_shape_ax = plt.subplots()
    gate_shape_basis2 =None
    gate_shape_ax.set_xlabel("coordinates in nm")
    gate_shape_ax.set_title("Gate shape")


    def __init__(self, DUT, name, shape, typ = 'gate'):
        self.shape = shape # [vertices positions]
        self.name = name
        self.SIMdim = [np.linspace(0, DUT.DUTdim[0], DUT.SIMdim[0]),
                       np.linspace(0, DUT.DUTdim[1], DUT.SIMdim[1]),
                       DUT.DUTdim[2]]
        # kwant.plot(DUT.DUT,ax =gate.gate_shape_ax)

        if typ == 'gate':
            self.pot_basis() # creates potential energy basis in DUT for 1 V on gate
                             # self.basis = potential_energy[y][x] ~ np.array (note q = -e < 0)
            gate.gates.append(self)
#            if type(self.gate_shape_basis) == type(None):
#            gate_shape_basis
            self.gate_shape(DUT)

            
        elif typ == 'bias':
            pass

        else:
            raise("your 'gate' #{} is named wrongly.".format(len(gate.gates) 
                                                             + len(gate.others) + 1))
            
    def pot_basis(self):
        def cot(x):
            tan_x = np.tan(x)
            tan_x[tan_x == 0] += 10**-15
        
            return 1 / tan_x
        
        z = self.SIMdim[2]
        X,Y = np.meshgrid(self.SIMdim[0], self.SIMdim[1])
        r = X+Y*1j
                
        pn = len(self.shape)
        p = np.array(self.shape)
        p = p[:,0] + 1j*p[:,1]
        
        basis = np.zeros(np.shape(r))
        for m in range(pn):
            p0, p1, p2 = p[(m - 1) % pn], p[m], p[(m + 1) % pn]
    
            a = -np.angle((r - p1)/(p0 - p1))
            b = np.angle((r - p1)/(p2 - p1))
    
            sin_g = z / np.sqrt(abs(r-p1)**2 + z**2)
            
            basis += ( (np.arctan(cot(a)) - np.arctan(sin_g * cot(a))) 
                            + (np.arctan(cot(b)) - np.arctan(sin_g * cot(b))) )
        
        self.basis = - basis / (2 * np.pi)
    
    def pot_all():
        gate.all_gates = np.transpose(np.zeros((len(gate.SIMdim[0]), len(gate.SIMdim[1]))))
        
        for i in range(len(gate.gates)):
            gate.all_gates += -gate.gates[i].basis
        
    #    potfig=plt.figure()
    #    plt.imshow(gate.all_gates, 
    #               origin = 'lower', 
    #               extent = (min(gate.SIMdim[0]), max(gate.SIMdim[0]),
    #                         min(gate.SIMdim[1]), max(gate.SIMdim[1])),
    #               cmap = 'hot_r',
    #               clim = (0, np.amax(gate.all_gates)))
    #    plt.title("Potential [V] by all gates at 1 V (z = {} nm)".format(gate.SIMdim[2]))
    #    plt.xlabel("Coordinates in nm")
    #    plt.colorbar()  
        
    def gate_shape(self,DUT):
        def cot(x):
            tan_x = np.tan(x)
            tan_x[tan_x == 0] += 10**-15
        
            return 1 / tan_x
        
        z = 10**-3
        X,Y = np.meshgrid(self.SIMdim[0], self.SIMdim[1])
        r = X+Y*1j
        gate_shape_basis = np.zeros((DUT.SIMdim[0],DUT.SIMdim[1]))
        pn = len(self.shape)
        p = np.array(self.shape)
        p = p[:,0] + 1j*p[:,1]
        
        basis = np.zeros(np.shape(r))
        print(self.shape)
        for m in range(pn):
            p0, p1, p2 = p[(m - 1) % pn], p[m], p[(m + 1) % pn]
    
            a = -np.angle((r - p1)/(p0 - p1))
            b = np.angle((r - p1)/(p2 - p1))
    
            sin_g = z / np.sqrt(abs(r-p1)**2 + z**2)
            
            basis += ( (np.arctan(cot(a)) - np.arctan(sin_g * cot(a))) 
                            + (np.arctan(cot(b)) - np.arctan(sin_g * cot(b))) )
        
        
       
        gate_shape_basis2 = np.round(gate_shape_basis +np.transpose(basis) / (2 * np.pi)+0.3)
        gate_shape_basis2[np.where(gate_shape_basis2 <1)] = np.nan     
        gate.gate_shape_ax.pcolor(X,Y,np.transpose(gate_shape_basis2 ),cmap = 'Reds',zorder=1)

        DUT.add_gates(basis)