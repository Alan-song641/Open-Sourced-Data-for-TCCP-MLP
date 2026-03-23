MODULE sphCoords
    USE nlist
    IMPLICIT NONE
    PUBLIC

    CONTAINS

    SUBROUTINE calcSphCoords(natoms, npairs, pos_car, neighs, pairs,pair_indices, unitvects_pair, min_dists, nearest_neigh, max_neighs, coords)
        IMPLICIT NONE
        !Define input values
        DOUBLE PRECISION,PARAMETER :: PI = 4.*ATAN(1.0_8)
        DOUBLE PRECISION,PARAMETER :: RAD2DEG = 180.0_8/PI
        INTEGER :: nAtoms, npairs, MAX_NEIGHS
        INTEGER, DIMENSION(nAtoms) :: nearest_neigh
        DOUBLE PRECISION, DIMENSION(nAtoms): min_dists
        INTEGER,INTENT(INOUT),DIMENSION(nAtoms,MAX_NEIGHS) :: neighs
        DOUBLE PRECISION, DIMENSION(nFPs) :: etas, r_ss, r_cuts
        DOUBLE PRECISION :: Rijs
        INTEGER :: i, j
        DOUBLE PRECISION, DIMENSION(nAtoms, 3, 3) :: local_frame

        DO i = 1, nAtoms
          pair_index = nearest_neigh(i)
          neigh_index = pair_indices(2, pair_index)
          uvect = unitvects_pair(1, pair_index, :)  !i-->j
          IF (neigh_index == i) THEN
             neigh_index = pair_indices(1, pair_index)
             uvect = unitvects_pair(2, pair_index, :)  !j-->i
          ENDIF
          call to_local_frame(pos_car(i), uvect, local_frame(i, :, :))

          r_sq = pos_car(i)**2
          r = sqrt(r_sq)
          theta = atan(pos_car(i, 2)/pos_car(i, 1))
          ph = acos(pos_car(i,3)/r)

    END SUBROUTINE

    SUBROUTINE to_local_frame(origin_p, x_vect, local_frame)
        DOUBLE PRECISION, DIMENSION(3,3) :: local_frame
        DOUBLE PRECISION, DIMENSION(3) :: origin_p
        DOUBLE PRECISION, DIMENSION(3) :: x_vect
        local_frame(1,:) = x_vect
        call cross_product(origin_p, x_vect, local_frame(2,:))
        !print *, local_frame(2,:)
        call cross_product(x_vect, local_frame(2,:),local_frame(3,:))
    END SUBROUTINE
   
    SUBROUTINE cross_product(a, b, cross)
        DOUBLE PRECISION, DIMENSION(3) :: cross
        DOUBLE PRECISION, DIMENSION(3) :: a, b
        cross(1) = a(2) * b(3) - a(3) * b(2)
        cross(2) = a(3) * b(1) - a(1) * b(3)
        cross(3) = a(1) * b(2) - a(2) * b(1)
    END SUBROUTINE

END MODULE
