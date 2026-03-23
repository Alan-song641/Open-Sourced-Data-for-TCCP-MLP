
from ase.io import read, Trajectory, write
import os
import sys
import numpy as np

from pyamff.aseCalc_ele import aseCalc_ele


step = 1e-5  # Angstrom

pp = './'
# pp = '/mnt/d/Alan/PhD/pyamff_training/QEQ/Cucc/qeq-v2/nonGC/same_stru/fptimer-linear/train01'
pp = '/mnt/d/Alan/PhD/pyamff_training/QEQ/Cucc/qeq-v2/nonGC/same_stru/fptimer-analytical/train00'
# pp = '/mnt/d/Alan/PhD/pyamff_training/QEQ/NaCl/na8cl8/test_fp'
# pp = '/mnt/d/Alan/PhD/pyamff_training/QEQ/AuMgO/undoped/test_fp'

center = read('/mnt/d/Alan/PhD/pyamff_training/QEQ/Cucc/same_stru.traj', index='0')
# center = read(pp+'/test.traj', index='2')

atom_inds = np.arange(0, len(center))  # atom indices to test

model_path = pp+'/pyamff.pt'
fp_path = pp+'/fpParas.dat'
calc_name = 'aseCalc'

calc  = aseCalc_ele(dir_path=pp, model=pp+'/pyamff.pt')

# center = read(pp+'/train.traj', index='0')

center.calc = calc
orig_energy = center.get_potential_energy()
forces = center.get_forces()
print("atomID  Ele  numerical   analytical      diff      frac")

for direc in range(3):  # 0 means x direction, 1 means y direction, 2 means z direction
    disp = [0, 0, 0]
    disp[direc] += step
    print("Direction: {}".format(direc))
    for atom_ind in atom_inds:
        left = center.copy()
        left.positions[atom_ind] -= disp
        
        # plot_dir_path = './data/left{}'.format(atom_ind)
        # traj_path = './data/left{}.traj'.format(atom_ind)
        # left = read(traj_path)
        left.calc = calc
        left_eng = left.get_potential_energy()
        left_force = left.get_forces()

        # traj_path = './data/right{}.traj'.format(atom_ind)
        right = center.copy()
        right.positions[atom_ind] += disp
        # right = read(traj_path)
        right.calc = calc
        right_eng = right.get_potential_energy()
        right_force = right.get_forces()

        num_forces = (left_eng - right_eng) / (2*step)
        analyt_forces = forces[atom_ind][direc] # 0 means x direction
        
        # Handle division by zero or very small values
        if abs(analyt_forces) < 1e-10:
            if abs(num_forces) < 1e-10:
                # Both forces are essentially zero
                frac = 100.0  # Consider it a 100% match
            else:
                # Analytical force is zero but numerical isn't
                frac = float('inf')
        else:
            # Normal case - no division issues
            frac = num_forces / analyt_forces * 100
            
        print(' {}  {}  {:.9f}      {:.9f}      {:.9f}       {:.3f}'.format(
            atom_ind, center.get_chemical_symbols()[atom_ind], 
            num_forces, analyt_forces, num_forces-analyt_forces, frac))
