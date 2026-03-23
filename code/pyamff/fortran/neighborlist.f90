MODULE nlist
    USE atomsProp
    
    IMPLICIT NONE
    PUBLIC

    CONTAINS

!-----------------------------------------------------------------------------------!
! calc:  calculate the neighborlist based upon positions, pos, and cell, cell
!-----------------------------------------------------------------------------------!

    SUBROUTINE calcCellNum(nAtoms, pos_car, cell, rcut, ncells)
        INTEGER :: nAtoms
        DOUBLE PRECISION :: rcut
        DOUBLE PRECISION,DIMENSION(nAtoms,3) :: pos_car
        DOUBLE PRECISION,DIMENSION(3,3) :: cell

!f2py   INTENT(IN) :: nAtoms
!f2py   INTENT(IN) :: pos_car
!f2py   INTENT(IN) :: cell
!f2py   INTENT(IN) :: rcut
        !Variables
        DOUBLE PRECISION, DIMENSION(3) :: orth_latt, cross_ab,cross_bc, cross_ac

        !Output
        INTEGER, DIMENSION(3) :: ncells
!f2py   INTENT(OUT) :: ncells
        cross_bc = cross_product(cell(2, :), cell(3,:))
        cross_ac = cross_product(cell(1, :), cell(3,:))
        cross_ab = cross_product(cell(1, :), cell(2,:))
        orth_latt(1) = ABS(SUM(cell(1,:) * cross_bc) / norm(cross_bc))
        orth_latt(2) = ABS(SUM(cell(2,:) * cross_ac) / norm(cross_ac))
        orth_latt(3) = ABS(SUM(cell(3,:) * cross_ab) / norm(cross_ab))
        ncells = INT(2*rcut /orth_latt) + 1
        !print*, 'ncells', ncells
        !CALL EXIT(1)
    END SUBROUTINE

    SUBROUTINE genSupercell(nAtoms, pos_car, symbols, cell, ncells, tncells, rcut, supercell)

        INTEGER :: nAtoms, tncells
        DOUBLE PRECISION :: rcut
        DOUBLE PRECISION,DIMENSION(nAtoms,3) :: pos_car
        INTEGER, DIMENSION(nAtoms) :: symbols
        DOUBLE PRECISION,DIMENSION(3,3) :: cell
        
        !Variables
        INTEGER :: i, j, k, i_offset, id
        INTEGER, DIMENSION(3) :: ncells
        INTEGER,DIMENSION(tncells,3) :: offset_list  !
        DOUBLE PRECISION,DIMENSION(tncells,3) :: offset_vector
        
        !Output
        DOUBLE PRECISION,DIMENSION(3,3) :: supercell
        i_offset = 1
        DO i = 1, ncells(1)
            DO j = 1, ncells(2)
                DO k = 1, ncells(3)
                    offset_list(i_offset, :) = (/i-1, j-1, k-1/)
                    i_offset = i_offset + 1
                END DO
            END DO
        END DO
        offset_vector = MATMUL(offset_list, cell)
        supercell(1,:) = ncells(1) * cell(1,:)
        supercell(2,:) = ncells(2) * cell(2,:)
        supercell(3,:) = ncells(3) * cell(3,:)
        supersymbols = 0
        DO i = 1, tncells
           DO j = 1, nAtoms
                id = (i-1)*nAtoms+j
                pool_pos_car(id,:) = offset_vector(i,:) + pos_car(j,:)
                pool_ids(id) = j
                supersymbols(id) = symbols(j)
           END DO
        END DO
    END SUBROUTINE



    SUBROUTINE calcNlist(nAtoms, MAX_NEIGHS, pos_car, cell, symbols, rcut, nelement, &
                         forceEngine, rmins, tncells,nneigh_incell, &
                         npairs_incell, num_pairs, num_neigh, neighs, neighs_incell, skipNeighList)
        IMPLICIT NONE
        DOUBLE PRECISION,PARAMETER :: PI = 4.*ATAN(1.0_8)
        DOUBLE PRECISION,PARAMETER :: RAD2DEG = 180.0_8/PI
        LOGICAL, OPTIONAL :: skipNeighList ! change for_lammps as well

        ! input values
        INTEGER :: nAtoms, MAX_NEIGHS, nelement, forceEngine, tncells
        DOUBLE PRECISION :: rcut
        DOUBLE PRECISION,DIMENSION(nAtoms,3) :: pos_car
        DOUBLE PRECISION,DIMENSION(3,3) :: cell
        INTEGER, DIMENSION(nAtoms) :: symbols

        ! variables
        DOUBLE PRECISION,DIMENSION(nAtoms,3) :: pos_dir
        DOUBLE PRECISION,DIMENSION(3,3) :: car2dir, dir2car
        DOUBLE PRECISION,DIMENSION(3) :: v_dir, v_car, v_unit
        DOUBLE PRECISION,DIMENSION(3) :: v_dir_o
        DOUBLE PRECISION :: dist, dist2, rcut2
        INTEGER :: i, j, natoms_percell

        !output valeus
        INTEGER :: num_pairs, npairs_incell
        INTEGER, DIMENSION(nAtoms) :: num_neigh
        INTEGER,DIMENSION(nAtoms,MAX_NEIGHS) :: neighs
        INTEGER,DIMENSION(nAtoms,MAX_NEIGHS) :: neighs_incell
        INTEGER,DIMENSION(nAtoms/tncells) :: nneigh_incell
        DOUBLE PRECISION, DIMENSION(nelement, nelement) :: rmins

        rcut2 = rcut**2  ! should check the rcut is defined
        natoms_percell = nAtoms/tncells
        dir2car = TRANSPOSE(cell)
        car2dir = inverse(dir2car)
        ! to direct coordinates
        DO i = 1, nAtoms
            pos_dir(i,:) = MATMUL(car2dir, pos_car(i,:))
        END DO
        ! pair terms
        num_pairs = 0
        npairs_incell = 0
        nneigh_incell = 0
        num_neigh = 0
        IF(.NOT. PRESENT(skipNeighList) .OR. (PRESENT(skipNeighList) .AND. .NOT. skipNeighList)) THEN
            neighs = 0
        END IF
        pair_indices = 0
        pair_global_indices=0
        unitvects_pair=0

        !print *, 'init', pair_indices
        !n_ghost = nAtoms
        pair_info = 0
        gvects = 0
        !pair_end = 0
        num_eachpair = 0
        DO i = 1, nAtoms
            DO j = i+1, nAtoms
                v_dir = pos_dir(j,:) - pos_dir(i,:)
                v_dir_o = v_dir
                v_dir = MOD(v_dir + 100.5, 1.0) - 0.5
                v_car = MATMUL(dir2car, v_dir)
                dist2 = SUM(v_car*v_car)
                IF (dist2<rcut2) THEN
                    num_neigh(i) = num_neigh(i) + 1
                    num_neigh(j) = num_neigh(j) + 1
                    dist = SQRT(dist2)
                    !print*, i, j, dist
                    v_unit = v_car/dist
                    !Record minimum distance of two atoms
                    IF (dist<rmins(symbols(pool_ids(i)), symbols(pool_ids(j)))) THEN
                        IF (forceEngine == 0) THEN
                            rmins(symbols(pool_ids(i)), symbols(pool_ids(j))) = dist
                            rmins(symbols(pool_ids(j)), symbols(pool_ids(i))) = dist
                        ELSE IF (forceEngine == 1) THEN
                            dist = rmins(symbols(pool_ids(i)), symbols(pool_ids(j)))
                        END IF
                    END IF

                    ! make neighs to record the index in pairs not origin structure
                    IF (i <= natoms_percell) THEN
                       npairs_incell = npairs_incell+1
                       IF ((ANY(neighs_incell(i,:)==pool_ids(j))) .OR. (pool_ids(j)==i)) THEN
                         CONTINUE
                       ELSE
                          nneigh_incell(i) = nneigh_incell(i) + 1
                          neighs_incell(i, nneigh_incell(i)) = pool_ids(j)
                       END IF
                    END IF
                    IF (j <= natoms_percell) THEN
                       IF ((ANY(neighs_incell(j,:)==pool_ids(i))) .OR. (pool_ids(i)==j)) THEN
                         CONTINUE
                       ELSE
                          nneigh_incell(j) = nneigh_incell(j) + 1
                          neighs_incell(j, nneigh_incell(j)) = pool_ids(i)
                       END IF
                    END IF
                    num_pairs = num_pairs + 1
                    num_eachpair(i) = num_eachpair(i) + 1
                    IF(.NOT. PRESENT(skipNeighList) .OR. (PRESENT(skipNeighList) .AND. .NOT. skipNeighList)) THEN
                        neighs(i, num_neigh(i)) = j
                        neighs(j, num_neigh(j)) = i
                    END IF
                    pairs(num_pairs) = dist 

                    pair_indices(1, num_pairs) = pool_ids(i)
                    pair_indices(2, num_pairs) = pool_ids(j)

                    unitvects_pair(1, num_pairs,:) = v_unit
                    unitvects_pair(2, num_pairs,:) = -v_unit
                    pair_info(i, num_eachpair(i)) = num_pairs
                    pair_global_indices(1, num_pairs) = i
                    pair_global_indices(2, num_pairs) = j
                    IF (ANY((v_dir*v_dir_o<0.0))) THEN
                       gvects(num_pairs) = -1
                    ELSE
                       gvects(num_pairs) = 1
                    END IF
                END IF
            END DO
        END DO
    END SUBROUTINE

!-----------------------------------------------------------------------------------!
! norm:  return the norm of A(3) vector
!-----------------------------------------------------------------------------------!

    FUNCTION norm(A)

        DOUBLE PRECISION,DIMENSION(3) :: A
        DOUBLE PRECISION :: norm
        norm = SQRT(SUM(A * A))
        RETURN

    END FUNCTION

!-----------------------------------------------------------------------------------!
! cross_product: return the cross product of a(3) and b(3)
!-----------------------------------------------------------------------------------!

    FUNCTION cross_product(a,b)

        DOUBLE PRECISION,DIMENSION(3) :: a,b
        DOUBLE PRECISION,DIMENSION(3) :: cross_product

        cross_product(1) = a(2)*b(3) - a(3)*b(2)
        cross_product(2) = a(3)*b(1) - a(1)*b(3)
        cross_product(3) = a(1)*b(2) - a(2)*b(1)

        RETURN
    END FUNCTION cross_product

!-----------------------------------------------------------------------------------!
! inverse:  return the inverse of A(3,3)
!-----------------------------------------------------------------------------------!

    FUNCTION inverse(A)

        DOUBLE PRECISION,DIMENSION(3,3) :: A
        DOUBLE PRECISION,DIMENSION(3,3) :: inverse
        DOUBLE PRECISION :: det

        det = determinant(A)
        IF (det == 0) STOP 'Divide by zero in matrix inverse'
        inverse = adjoint(A)/det

        RETURN
    END FUNCTION inverse

!-----------------------------------------------------------------------------------!
! adjoint: return the adjoint of A(3,3)
!-----------------------------------------------------------------------------------!

    FUNCTION adjoint(A)

        DOUBLE PRECISION,DIMENSION(3,3) :: A
        DOUBLE PRECISION,DIMENSION(3,3) :: adjoint

        adjoint(1,1) = A(2,2)*A(3,3) - A(3,2)*A(2,3)
        adjoint(1,2) = A(1,3)*A(3,2) - A(1,2)*A(3,3)
        adjoint(1,3) = A(1,2)*A(2,3) - A(2,2)*A(1,3)

        adjoint(2,1) = A(2,3)*A(3,1) - A(2,1)*A(3,3)
        adjoint(2,2) = A(1,1)*A(3,3) - A(1,3)*A(3,1)
        adjoint(2,3) = A(1,3)*A(2,1) - A(1,1)*A(2,3)

        adjoint(3,1) = A(2,1)*A(3,2) - A(2,2)*A(3,1)
        adjoint(3,2) = A(1,2)*A(3,1) - A(1,1)*A(3,2)
        adjoint(3,3) = A(1,1)*A(2,2) - A(1,2)*A(2,1)

        RETURN
    END FUNCTION adjoint

!-----------------------------------------------------------------------------------!
! determinant: of a 3x3 matrix 
!-----------------------------------------------------------------------------------!

    FUNCTION determinant(A)

        DOUBLE PRECISION,DIMENSION(3,3) :: A
        DOUBLE PRECISION :: determinant

        determinant = A(1,1)*A(2,2)*A(3,3) &
                    - A(1,1)*A(2,3)*A(3,2) &
                    - A(1,2)*A(2,1)*A(3,3) &
                    + A(1,2)*A(2,3)*A(3,1) &
                    + A(1,3)*A(2,1)*A(3,2) &
                    - A(1,3)*A(2,2)*A(3,1)

        RETURN
    END FUNCTION

END MODULE
