MODULE nlist

    IMPLICIT NONE
    PUBLIC

    CONTAINS

!-----------------------------------------------------------------------------------!
! calc:  calculate the neighborlist based upon positions, pos, and cell, cell
!-----------------------------------------------------------------------------------!

    SUBROUTINE calcNlist(nAtoms, MAX_NEIGHS, pos_car, cell, rcut, nelement, symbols, forceEngine, rmins, &
                        num_pairs, num_neigh, neighs, pairs, pair_indices, pair_start, pair_end, unitvects_pair, &
                        min_dists, nearest_neigh)

        DOUBLE PRECISION,PARAMETER :: PI = 4.*ATAN(1.0_8)
        DOUBLE PRECISION,PARAMETER :: RAD2DEG = 180.0_8/PI
        !INTEGER,PARAMETER :: MAX_NEIGHS = 100  ! should make this dynamic
        !INTEGER,PARAMETER :: MAX_ANGLES = 1024  ! should make this dynamic

        ! input values
        INTEGER :: nAtoms, MAX_NEIGHS, nelement, forceEngine
        DOUBLE PRECISION :: rcut
        DOUBLE PRECISION,INTENT(IN),DIMENSION(nAtoms,3) :: pos_car
        DOUBLE PRECISION,INTENT(IN),DIMENSION(3,3) :: cell
        INTEGER,INTENT(INOUT),DIMENSION(nAtoms,MAX_NEIGHS) :: neighs
        INTEGER, DIMENSION(nAtoms) :: symbols

        ! variables
        INTEGER, DIMENSION(nAtoms) :: num_eachpair
        DOUBLE PRECISION,DIMENSION(nAtoms,3) :: pos_dir
        DOUBLE PRECISION,DIMENSION(3,3) :: car2dir, dir2car
        DOUBLE PRECISION,DIMENSION(3) :: v_dir, v_car, v_unit
        DOUBLE PRECISION :: dist, dist2, rcut2
        INTEGER :: i, j, check

        !output valeus
        INTEGER :: num_pairs
        INTEGER, DIMENSION(nAtoms) :: pair_start, pair_end
        DOUBLE PRECISION,INTENT(INOUT),DIMENSION(nAtoms*MAX_NEIGHS) :: pairs
        INTEGER, DIMENSION(2, nAtoms*MAX_NEIGHS) :: pair_indices
        INTEGER, DIMENSION(nAtoms) :: num_neigh, nearest_neigh
        DOUBLE PRECISION, DIMENSION(nAtoms) :: min_dists
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(2,nAtoms*MAX_NEIGHS,3) :: unitvects_pair
        DOUBLE PRECISION, DIMENSION(nelement, nelement) :: rmins

        rcut2 = rcut**2  ! should check the rcut is defined
        !print *, 'ruct', rcut
        dir2car = TRANSPOSE(cell)
        car2dir = inverse(dir2car)
        !print *, car2dir
        ! to direct coordinates
        DO i = 1, nAtoms
            !print*, i, pos_car(i,:)
            pos_dir(i,:) = MATMUL(car2dir, pos_car(i,:))
            !print*, i, pos_car(i,:)
        END DO
        ! pair terms
        num_pairs = 0
        num_neigh = 0
        neighs = 0
        pair_indices = 0
        min_dists = 1000.0
        !print *, 'init', pair_indices
        check = 0
        pair_start = 0
        pair_end = 0
        num_eachpair = 0
        !print*, 'forceEngine', forceEngine
        DO i = 1, nAtoms
            DO j = i+1, nAtoms
                !check = check +1
                !print *, check
                v_dir = pos_dir(j,:) - pos_dir(i,:)
                !v_dir = pos_dir(:,j) - pos_dir(:,i)
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
                    IF (dist<rmins(symbols(i), symbols(j))) THEN
                        IF (forceEngine == 0) THEN
                            rmins(symbols(i), symbols(j)) = dist
                            rmins(symbols(j), symbols(i)) = dist
                        ELSE IF (forceEngine == 1) THEN
                            dist = rmins(symbols(i), symbols(j))
                            !print*,'replaced', dist
                        END IF
                    END IF

                    !Lei added
                    !make neighs to record the index in pairs not origin structure
                    !num_pairs = (i-1)*(nAtoms-1)+j
                    num_pairs = num_pairs + 1
                    IF (pair_start(i) == 0) THEN
                        pair_start(i) = num_pairs
                    ENDIF
                    num_eachpair(i) = num_eachpair(i) + 1
                    neighs(i, num_neigh(i)) = num_pairs
                    neighs(j, num_neigh(j)) = num_pairs
                    pairs(num_pairs) = dist  
                    pair_indices(1, num_pairs) = i
                    !print *, num_pairs, i
                    pair_indices(2, num_pairs) = j
                    IF (dist<min_dists(i)) THEN
                       min_dists(i) = dist
                       nearest_neigh = num_pairs
                    ENDIF
                    IF (dist<min_dists(j)) THEN
                       min_dists(j) = dist
                       nearest_neigh = num_pairs
                    ENDIF

                    !dists(i,num_neigh(i)) = dist
                    !dists(j,num_neigh(j)) = dist
                    !vects(i,num_neigh(i),:) = v_car
                    !vects(j,num_neigh(j),:) = -v_car
                    !neigh_indices(i, num_neigh(i)) = j
                    !neigh_indices(j, num_neigh(j)) = i
                    !unitvects(i,num_neigh(i),:) = v_unit
                    !unitvects(j,num_neigh(j),:) = -v_unit
                    unitvects_pair(1, num_pairs,:) = v_unit
                    unitvects_pair(2, num_pairs,:) = -v_unit
                END IF
            END DO
        END DO
        !print*, 'calcNeigh Done'
        pair_end = pair_start + num_eachpair - 1
        !print *,'50 start', pair_start(3)
        !print *,'50 end', pair_end(3)

    END SUBROUTINE

!-----------------------------------------------------------------------------------!
! inverse:  return the inverse of A(3,3)
!-----------------------------------------------------------------------------------!

    FUNCTION inverse(A)

        DOUBLE PRECISION,INTENT(IN),DIMENSION(3,3) :: A
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

        DOUBLE PRECISION,INTENT(IN),DIMENSION(3,3) :: A
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

        DOUBLE PRECISION,INTENT(IN),DIMENSION(3,3) :: A
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
