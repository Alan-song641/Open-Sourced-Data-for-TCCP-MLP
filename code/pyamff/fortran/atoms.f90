MODULE atomsProp

    IMPLICIT NONE
    PUBLIC
    
    INTEGER :: tnAtoms
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: pool_pos_car
    INTEGER, DIMENSION(:), ALLOCATABLE :: pool_ids
    INTEGER, DIMENSION(:), ALLOCATABLE :: supersymbols

    DOUBLE PRECISION,DIMENSION(:), ALLOCATABLE :: pairs 
    !INTEGER, DIMENSION(:), ALLOCATABLE :: pair_start, pair_end
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: pair_info
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: pair_indices 
    !INTEGER, DIMENSION(:,:), ALLOCATABLE :: pair_ghosts
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: pair_global_indices
    DOUBLE PRECISION, DIMENSION(:, :, :), ALLOCATABLE :: unitvects_pair

    INTEGER, DIMENSION(:), ALLOCATABLE :: gvects
    INTEGER, DIMENSION(:, :), ALLOCATABLE :: tneighs
    INTEGER, DIMENSION(:, :), ALLOCATABLE :: tneighs_incell
    INTEGER, DIMENSION(:), ALLOCATABLE :: tnum_neigh
    INTEGER, DIMENSION(:), ALLOCATABLE :: num_eachpair
    DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: angles
    INTEGER, DIMENSION(:, :), ALLOCATABLE :: angles_indices
    !Fingerprints of atoms
    DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: fps
    DOUBLE PRECISION, DIMENSION(:, :, :, :), ALLOCATABLE :: dfps
    INTEGER :: global_max_neighs

    CONTAINS 

    SUBROUTINE dummynn()
    END SUBROUTINE

    SUBROUTINE allocate_outputs(nAtoms, maxfps, maxnneighs)
      INTEGER :: nAtoms, maxfps, maxnneighs
      CALL deallocate_outputs()
      ALLOCATE(fps(nAtoms, maxfps))
      ALLOCATE(dfps(nAtoms, maxnneighs+1, 3, maxfps))
      global_max_neighs = maxnneighs
      fps = 0.0
      dfps = 0.0
    END SUBROUTINE

    SUBROUTINE deallocate_outputs()
      IF (ALLOCATED(fps)) THEN
        DEALLOCATE(fps)
      END IF
      IF (ALLOCATED(dfps)) THEN
        DEALLOCATE(dfps)
      END IF
    END SUBROUTINE
END MODULE
