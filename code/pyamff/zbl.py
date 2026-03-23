from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
import numpy as np


alpha = 1 / 137.035999046  # no unit
hbar = 6.582119569e-16  # eV * s
c = 299792458 * 1e10  # m/s * A/m
prefactor = alpha * hbar * c  # eV * A

tunable_a = 0.1
Z_C = 6
Z_CU = 29


def is_cu_c_pair(Z_i, Z_j):
    return ((Z_i == Z_CU) & (Z_j == Z_C)) | ((Z_i == Z_C) & (Z_j == Z_CU))

def a(Z_i, Z_j):
    return 0.46850 / (Z_i**0.23 + Z_j**0.23)


def phi(x):
    return (0.18175 * np.exp(-3.19980 * x)
            + 0.50986 * np.exp(-0.94229 * x)
            + 0.28022 * np.exp(-0.40290 * x)
            + 0.02817 * np.exp(-0.20162 * x))


def phi_derivative(x):
    return (- 3.19980 * 0.18175 * np.exp(-3.19980 * x)
            - 0.94229 * 0.50986 * np.exp(-0.94229 * x)
            - 0.40290 * 0.28022 * np.exp(-0.40290 * x)
            - 0.20162 * 0.02817 * np.exp(-0.20162 * x))


def zbl_energy(Z_i, Z_j, r_ij):
    return prefactor * Z_i * Z_j / r_ij * phi(r_ij / a(Z_i, Z_j))


def zbl_force(Z_i, Z_j, r_ij):
    return (-prefactor
            * (Z_i * Z_j)
            * (phi_derivative(r_ij / a(Z_i, Z_j)) / (r_ij * a(Z_i, Z_j))
               - phi(r_ij / a(Z_i, Z_j)) / r_ij**2))


def inner_loop(ai, positions, cell, offsets, neighbors, forces, numbers, Ra, Rb):
    if len(neighbors) == 0:
        return 0.0, 0.0

    pair_mask = is_cu_c_pair(numbers[ai], numbers[neighbors])
    if not np.any(pair_mask):
        return 0.0, 0.0

    neighbors = neighbors[pair_mask]
    offsets = offsets[pair_mask]

    cells = np.dot(offsets, cell)
    v_ij = positions[neighbors] + cells - positions[ai]
    r_ij = np.linalg.norm(v_ij, axis=1)

    # smooth-minimal distance of atom i’s near neighbors
    de_i = np.exp(-r_ij / tunable_a)
    nu_i = r_ij * de_i
    sigma_i = np.sum(nu_i) / np.sum(de_i)

    u_i = (sigma_i - Ra) / (Rb - Ra)

    if sigma_i < Ra:
        weight_i = 1.0
    elif sigma_i >= Rb:
        weight_i = 0.0
    else:
        weight_i = -6 * u_i**5 + 15 * u_i**4 - 10 * u_i**3 + 1

    pair_energies = zbl_energy(numbers[ai], numbers[neighbors], r_ij)
    pair_force_magnitudes = zbl_force(numbers[ai], numbers[neighbors], r_ij)
    pair_forces = v_ij / r_ij[:, None] * pair_force_magnitudes[:, None]

    forces[neighbors] += pair_forces * weight_i
    forces[ai] -= pair_forces.sum(axis=0) * weight_i

    # NOTE: energy and forces are already weighted by weight_i
    # print(weight_i)
    return np.sum(pair_energies)*weight_i, weight_i


def outer_loop(positions, cell, offsets_list, neighbors_list, forces, numbers, Ra, Rb):
    energy = np.zeros(len(positions))
    weights = np.zeros(len(positions))
    for ai in range(len(positions)):
        neighbors = neighbors_list[ai]
        offsets = offsets_list[ai]
        energy[ai], weights[ai] = inner_loop(ai, positions, cell, offsets, neighbors, forces, numbers, Ra, Rb)
    
    energy = np.sum(energy)

    return energy, weights


class ZBLCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, cutoff, skin=1.0, **kwargs):
        self._cutoff = cutoff
        self._skin = skin
        super().__init__(**kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if not ('forces' in properties or 'energy' in properties):
            return

        n_atoms = len(self.atoms)

        if 'numbers' in system_changes:
            self.nl = NeighborList([self._cutoff / 2] * n_atoms,
                                   self_interaction=False,
                                   skin=self._skin)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell.array

        neighbors_list, offsets_list = [], []
        for ai in range(n_atoms):
            neighbors, offsets = self.nl.get_neighbors(ai)
            neighbors_list.append(neighbors)
            offsets_list.append(offsets.astype(np.float64))

        forces = np.zeros((n_atoms, 3))

        energy, weights = outer_loop(positions, cell, offsets_list, neighbors_list, 
                            forces, numbers, Ra=self._cutoff-self._skin, Rb=self._cutoff)

        # self.results['energy'] = energy
        # self.results['forces'] = forces
        return energy, forces, weights
