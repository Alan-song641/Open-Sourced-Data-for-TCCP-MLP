import numpy as np
import itertools
from numpy.linalg import inv
import time
from numpy.typing import ArrayLike

def timing_decorator(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper


# from pymatgen
class Lattice():
    """
    A lattice object. Essentially a matrix with conversion matrices. In
    general, it is assumed that length units are in Angstroms and angles are in
    degrees unless otherwise stated.
    """

    # Properties lazily generated for efficiency.
    def __init__(self, matrix, pbc=(True, True, True)):
        """
        Create a lattice from any sequence of 9 numbers. Note that the sequence
        is assumed to be read one row at a time. Each row represents one
        lattice vector.

        Args:
            matrix: Sequence of numbers in any form. Examples of acceptable
                input.
                
            pbc: a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.
        """
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        m.setflags(write=False)
        self._matrix = m
        self._inv_matrix = None
        self._pbc = tuple(pbc)

    @property
    def reciprocal_lattice(self):
        """
        Return the reciprocal lattice. Note that this is the standard
        reciprocal lattice used for solid state physics with a factor of 2 *
        pi. If you are looking for the crystallographic reciprocal lattice,
        use the reciprocal_lattice_crystallographic property.
        The property is lazily generated for efficiency.
        """
        v = np.linalg.inv(self._matrix).T
        return Lattice(v * 2 * np.pi)

    @property
    def abc(self):
        """
        :return: The lengths (a, b, c) of the lattice.
        """
        return tuple(np.sqrt(np.sum(self._matrix**2, axis=1)).tolist())  # type: ignore
    
    # @timing_decorator
    def get_points_in_sphere(
        self,
        frac_points: ArrayLike,
        center: ArrayLike,
        r: float,
        zip_results: bool = True,
    ):
        """Find all points within a sphere from the point taking into account
        periodic boundary conditions. This includes sites in other periodic images.

        Algorithm:

        1. place sphere of radius r in crystal and determine minimum supercell
           (parallelepiped) which would contain a sphere of radius r. for this
           we need the projection of a_1 on a unit vector perpendicular
           to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
           determine how many a_1's it will take to contain the sphere.

           Nxmax = r * length_of_b_1 / (2 Pi)

        2. keep points falling within r.

        Args:
            frac_points: All points in the lattice in fractional coordinates.
            center: Cartesian coordinates of center of sphere.
            r: radius of sphere.
            zip_results (bool): Whether to zip the results together to group by
                point, or return the raw frac_coord, dist, index arrays

        Returns:
            if zip_results:
                [(frac_coord, dist, index, supercell_image) ...] since most of the time, subsequent
                processing requires the distance, index number of the atom, or index of the image
            else:
                frac_coords, dists, inds, image
        """
        try:
            from pyamff.fingerprints.ewald_neighbors import find_points_in_spheres
        except ImportError as e:
            print(e)

        else:
            frac_points = np.ascontiguousarray(frac_points, dtype=float)
            cart_coords = np.ascontiguousarray(self.get_cartesian_coords(frac_points), dtype=float)
            center_coords = np.ascontiguousarray([center], dtype=float)
            pbc = np.ascontiguousarray(self._pbc, dtype=np.int64)
            latt_matrix = np.ascontiguousarray(self._matrix, dtype=float)

            _, indices, images, distances = find_points_in_spheres(
                all_coords=cart_coords,
                center_coords=center_coords,
                r=float(r),
                pbc=pbc,
                lattice=latt_matrix,
                tol=1e-8,
            )

            if len(indices) < 1:
                # Return empty np.array (not list or tuple) to ensure consistent return type
                # whether sphere contains points or not
                return np.array([]) if zip_results else tuple(np.array([]) for _ in range(4))
            frac_coords = frac_points[indices] + images
            if zip_results:
                return tuple(zip(frac_coords, distances, indices, images))
            return frac_coords, distances, indices, images
                                    

    def _one_to_three(self, label1d, ny, nz):
        """
        Convert a 1D index array to 3D index array

        Args:
            label1d: (array) 1D index array
            ny: (int) number of cells in y direction
            nz: (int) number of cells in z direction

        Returns: (nx3) int array of index

        """
        last = np.mod(label1d, nz)
        second = np.mod((label1d - last) / nz, ny)
        first = (label1d - last - second * nz) / (ny * nz)
        return np.concatenate([first, second, last], axis=1)

    def _three_to_one(self, label3d, ny, nz):
        """
        The reverse of _one_to_three
        """
        return np.array(label3d[:, 0] * ny * nz + label3d[:, 1] * nz +
                        label3d[:, 2]).reshape((-1, 1))

    def _compute_cube_index(self, coords, global_min, radius):
        """
        Compute the cube index from coordinates
        Args:
            coords: (nx3 array) atom coordinates
            global_min: (float) lower boundary of coordinates
            radius: (float) cutoff radius

        Returns: (nx3 array) int indices

        """
        return np.array(np.floor((coords - global_min) / radius), dtype=int)

    def find_neighbors(self, label, nx, ny, nz):
        """
        Given a cube index, find the neighbor cube indices

        Args:
            label: (array) (n,) or (n x 3) indice array
            nx: (int) number of cells in y direction
            ny: (int) number of cells in y direction
            nz: (int) number of cells in z direction

        Returns: neighbor cell indices

        """

        array = [[-1, 0, 1]] * 3
        neighbor_vectors = np.array(list(itertools.product(*array)), dtype=int)
        if np.shape(label)[1] == 1:
            label3d = self._one_to_three(label, ny, nz)
        else:
            label3d = label
        all_labels = label3d[:, None, :] - neighbor_vectors[None, :, :]
        filtered_labels = []
        # filter out out-of-bound labels i.e., label < 0
        for labels in all_labels:
            ind = (labels[:, 0] < nx) * (labels[:, 1] < ny) * (
                labels[:, 2] < nz) * np.all(labels > -1e-5, axis=1)
            filtered_labels.append(labels[ind])
        return filtered_labels

    def get_cartesian_coords(self, fractional_coords):
        """
        Returns the Cartesian coordinates given fractional coordinates.

        Args:
            fractional_coords (3x1 array): Fractional coords.

        Returns:
            Cartesian coordinates
        """
        return np.dot(fractional_coords, self._matrix)

    def get_fractional_coords(self, cart_coords):
        """
        Returns the fractional coordinates given Cartesian coordinates.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
        return np.dot(cart_coords, self.inv_matrix)

    @property
    def inv_matrix(self) -> np.ndarray:
        """
        Inverse of lattice matrix.
        """
        if self._inv_matrix is None:
            self._inv_matrix = inv(self._matrix)
            self._inv_matrix.setflags(write=False)
        return self._inv_matrix