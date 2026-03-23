MODULE pyamff
    USE fpCalc
    USE atomsProp
    IMPLICIT NONE
    LOGICAL :: read_file
    !PRIVATE
    !PUBLIC::calc_eon, calc_ase, tdfps, tfps
    !DOUBLE PRECISION, DIMENSION(:, :, :, :), ALLOCATABLE :: tdfps
    !DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: tfps
    CONTAINS

    !SUBROUTINE calc_eon(pos_car, nAtoms, ncoords, box, atomicNumbers, F, U, nelement, uniqueNrs) BIND(C,name='calc_eon')
    SUBROUTINE calc_eon(nAtoms, R, box, atomicNumbers, F, U, nelement, uniqueNrs) BIND(C,name='calc_eon')
        USE, INTRINSIC :: iso_c_binding
        !USE fpCalc
        USE nlist
        USE normalize
        USE fnnmodule
        USE nnType
        IMPLICIT NONE

        INTEGER(c_long), INTENT(IN) :: nAtoms
        INTEGER :: NAT, i, j
        REAL(c_double), DIMENSION(nAtoms*3), INTENT(IN) :: R
        REAL(c_double), DIMENSION(9), INTENT(IN) :: box
        REAL(c_double), INTENT(OUT) :: U
        REAL(c_double), DIMENSION(nAtoms*3):: F
        INTEGER(c_int), DIMENSION(nAtoms) :: atomicNumbers

        !INTEGER, PARAMETER :: MAX_FPs = 100
        INTEGER, PARAMETER :: MAX_NEIGHS = 100
        INTEGER, PARAMETER ::forceEngine = 1
        DOUBLE PRECISION, DIMENSION(nAtoms,3) :: pos_car
        INTEGER, DIMENSION(nAtoms) :: symbols
        CHARACTER*3, DIMENSION(nAtoms) :: atomicSymbols

        DOUBLE PRECISION, DIMENSION(3,3) :: cell

        INTEGER(c_int) :: nelement
        INTEGER(c_int), DIMENSION(nelement) :: uniqueNrs
        CHARACTER*20 :: filename
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements
        DOUBLE PRECISION, DIMENSION(nAtoms,3) :: pos_dir
        DOUBLE PRECISION, DIMENSION(3,3) :: dir2car

        !DOUBLE PRECISION, DIMENSION(nAtoms, MAX_FPS) :: fps
        !DOUBLE PRECISION, DIMENSION(nAtoms, MAX_NEIGHS, 3, MAX_FPS) :: dfps
        CHARACTER*3, DIMENSION(92) :: elementArray
        INTEGER, DIMENSION(nAtoms, MAX_NEIGHS) :: neighs
        INTEGER, DIMENSION(nAtoms, MAX_NEIGHS) :: sub_neighs
        INTEGER, DIMENSION(nAtoms) :: num_neigh, sub_num_neigh

        DATA elementArray / "H","He","Li","Be","B","C","N","O", &
                 "F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc", &
                 "Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se", &
                 "Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag", &
                 "Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd", &
                 "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta", &
                 "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn", &
                 "Fr","Ra","Ac","Th","Pa","U" /

        DO i = 1, nelement
            ! Read from pos.con
            uniq_elements(i) = elementArray(uniqueNrs(i))
        END DO

        DO i = 1, nAtoms
            DO j = 1, nelement
                IF (atomicNumbers(i) == uniqueNrs(j)) THEN 
                    symbols(i) = j
                END IF
            END DO
        END DO

        cell(1,1) = box(1)
        cell(1,2) = box(2)
        cell(1,3) = box(3)

        cell(2,1) = box(4)
        cell(2,2) = box(5)
        cell(2,3) = box(6)

        cell(3,1) = box(7)
        cell(3,2) = box(8)
        cell(3,3) = box(9)

        DO i = 1, nAtoms
            pos_car(i,1) = R(3*(i-1) + 1)
            pos_car(i,2) = R(3*(i-1) + 2)
            pos_car(i,3) = R(3*(i-1) + 3)
            !print *, "pos_car", pos_car(i,1), pos_car(i,2), pos_car(i,3)
        END DO

        nat = nAtoms
        !dfps = 0.0
        !fps = 0.0
        !print*, fps
        !CALL calcfps(nAt, pos_car, cell, symbols, MAX_FPs, nelement, forceEngine, &
        CALL calcfps(nAt, pos_car, cell, symbols,  nelement, forceEngine, &
                     sub_num_neigh, sub_neighs, num_neigh)
                     !fps, dfps, sub_num_neigh, sub_neighs, num_neigh)
        CALL normalizeFPs(nelement, nAt, uniq_elements, MAX_FPS, MAX_NEIGHS, &
                                  sub_num_neigh, sub_neighs)
                                  !sub_num_neigh, sub_neighs, fps, dfps)
        CALL forwardCalc(sub_num_neigh, MAX_NEIGHS, sub_neighs,&
                         !fps, dfps, &
                         MAX_FPs, MAXVAL(natoms_arr), MAXVAL(nhidneurons))

        U = Etotal 

        !open(unit=2, file='eonfc.dat', ACTION="write", STATUS="replace")

        DO i=1,nAtoms
            F(3*(i-1)+1) = forces(1,i) 
            F(3*(i-1)+2) = forces(2,i)
            F(3*(i-1)+3) = forces(3,i)
        !   write(2,*) F(3*(i-1)+1), F(3*(i-1)+2), F(3*(i-1)+3)
        END DO
        !close(2)
        CALL atomscleanup
        !CALL nncleanup()
    END SUBROUTINE

    SUBROUTINE calc_ase(nAtoms, pos_car, box, atomicNumbers, nelement, uniqueNrs, mlff_file, F, U)

        !USE fpCalc
        USE nlist
        USE normalize
        USE fnnmodule
        USE nnType
        IMPLICIT NONE

        INTEGER :: nAtoms
        INTEGER :: i, j, k
        !REAL, DIMENSION(nAtoms*3), INTENT(IN) :: R
        DOUBLE PRECISION, DIMENSION(9), INTENT(IN) :: box
        DOUBLE PRECISION, INTENT(OUT) :: U
        DOUBLE PRECISION, DIMENSION(nAtoms,3), INTENT(OUT):: F
        INTEGER, DIMENSION(nAtoms) :: atomicNumbers

        !INTEGER, PARAMETER :: MAX_FPs = 500
        INTEGER, PARAMETER :: MAX_NEIGHS = 100
        INTEGER, PARAMETER ::forceEngine = 1
        DOUBLE PRECISION, DIMENSION(nAtoms,3), INTENT(IN) :: pos_car
        INTEGER, DIMENSION(nAtoms) :: symbols
        CHARACTER*3, DIMENSION(nAtoms) :: atomicSymbols

        DOUBLE PRECISION, DIMENSION(3,3) :: cell

        INTEGER :: nelement
        INTEGER, DIMENSION(nelement), INTENT(IN) :: uniqueNrs
        CHARACTER*20 :: filename
        CHARACTER*100 :: mlff_file
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements
        DOUBLE PRECISION, DIMENSION(nAtoms,3) :: pos_dir
        DOUBLE PRECISION, DIMENSION(3,3) :: dir2car

        !DOUBLE PRECISION, DIMENSION(nAtoms, MAX_FPS) :: fps
        !DOUBLE PRECISION, DIMENSION(nAtoms, MAX_NEIGHS, 3, MAX_FPS) :: dfps
        CHARACTER*3, DIMENSION(92) :: elementArray
        INTEGER, DIMENSION(nAtoms, MAX_NEIGHS) :: neighs
        INTEGER, DIMENSION(nAtoms, MAX_NEIGHS) :: sub_neighs
        INTEGER, DIMENSION(nAtoms) :: num_neigh, sub_num_neigh

        DATA elementArray / "H","He","Li","Be","B","C","N","O", &
                 "F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc", &
                 "Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se", &
                 "Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag", &
                 "Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd", &
                 "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta", &
                 "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn", &
                 "Fr","Ra","Ac","Th","Pa","U" /

        DO i = 1, nelement
            ! Read from pos.con
            uniq_elements(i) = elementArray(uniqueNrs(i))
        END DO

        DO i = 1, nAtoms
            DO j = 1, nelement
                IF (atomicNumbers(i) == uniqueNrs(j)) THEN
                    symbols(i) = j
                END IF
            END DO
        END DO
        cell(1,1) = box(1)
        cell(1,2) = box(2)
        cell(1,3) = box(3)

        cell(2,1) = box(4)
        cell(2,2) = box(5)
        cell(2,3) = box(6)

        cell(3,1) = box(7)
        cell(3,2) = box(8)
        cell(3,3) = box(9)

        !dfps = 0.0
        !fps = 0.0
        ! Read mlff.pyamff only once for the very first image
        !read_file = .True.
        IF (read_file) THEN
          !CALL read_mlff(nAtoms, nelement, MAX_FPS, atomicNumbers, uniqueNrs)
          CALL read_mlff(nelement, mlff_file)
        END IF

        ! Update atomic information
        CALL update_atomInfo(nAtoms, nelement, MAX_FPS, symbols)

        CALL calcfps(nAtoms, pos_car, cell, symbols, nelement, forceEngine, &
                     sub_num_neigh, sub_neighs, num_neigh)
                     !fps, dfps, sub_num_neigh, sub_neighs, num_neigh)

        CALL normalizeFPs(nelement, nAtoms, uniq_elements, MAX_FPS, MAX_NEIGHS, &
                          sub_num_neigh, sub_neighs)
                          !sub_num_neigh, sub_neighs, fps, dfps)
        CALL forwardCalc(sub_num_neigh, MAX_NEIGHS, sub_neighs,&
                         !fps, dfps, &
                         MAX_FPS, MAXVAL(natoms_arr), MAXVAL(nhidneurons))
        U = Etotal

        DO i = 1, nAtoms
            F(i,1) = forces(1,i) 
            F(i,2) = forces(2,i)
            F(i,3) = forces(3,i)
        END DO
        !Deallocate arrays
        CALL atomscleanup
        CALL nncleanup_ase
    END SUBROUTINE

    SUBROUTINE calc_lammps(nAtoms,R,box,atomicNumbers,F,U,nelement,uniqueNrs, &
        MAX_NEIGHS,neighlist,tAtoms,num_cells_dir,P,tags) BIND(C,name='calc_lammps')
        !MAX_FPS,MAX_NEIGHS,neighlist,tAtoms,num_cells_dir,P,tags) BIND(C,name='calc_lammps')
        USE, INTRINSIC :: iso_c_binding
        USE nlist
        USE normalize
        USE fnnmodule
        USE nnType
        USE atomsProp, only: pool_pos_car, pool_ids, supersymbols 
        IMPLICIT NONE

        INTEGER :: NAT, TAT, i, j, id, tcells
        INTEGER(c_int) :: nelement
        INTEGER(c_long), INTENT(IN) :: nAtoms, tAtoms
        !INTEGER(c_int), INTENT(IN) :: MAX_FPS,MAX_NEIGHS
        INTEGER(c_int), INTENT(IN) :: MAX_NEIGHS
        INTEGER(c_int), DIMENSION(nelement) :: uniqueNrs
        INTEGER(c_int), DIMENSION(nAtoms), INTENT(IN) :: atomicNumbers
        INTEGER(c_int), DIMENSION(tAtoms), INTENT(IN) :: tags
        INTEGER(c_int), DIMENSION(3), INTENT(IN) :: num_cells_dir
        INTEGER(c_int), DIMENSION(tAtoms*MAX_NEIGHS), INTENT(IN) :: neighlist
        REAL(c_double), DIMENSION(9), INTENT(IN) :: box
        REAL(c_double), DIMENSION(nAtoms*3), INTENT(IN) :: R
        REAL(c_double), DIMENSION(tAtoms*3), INTENT(IN) :: P
        REAL(c_double), DIMENSION(nAtoms*3), INTENT(OUT) :: F
        REAL(c_double), INTENT(OUT) :: U

        LOGICAL, PARAMETER :: usingLammps = .TRUE.
        INTEGER, PARAMETER ::forceEngine = 1
        INTEGER, DIMENSION(nAtoms) :: symbols
        INTEGER, DIMENSION(nAtoms) :: num_neigh, sub_num_neigh
        INTEGER, DIMENSION(nAtoms, nAtoms) :: sub_neighs
        INTEGER, DIMENSION(nAtoms, MAX_NEIGHS) :: neighs
        ! INTEGER, DIMENSION(tAtoms, MAX_NEIGHS) :: total_neighs

        CHARACTER*20 :: filename
        CHARACTER*3, DIMENSION(92) :: elementArray
        CHARACTER*3, DIMENSION(nAtoms) :: atomicSymbols
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements
        DOUBLE PRECISION, DIMENSION(3,3) :: cell, dir2car
        DOUBLE PRECISION, DIMENSION(nAtoms,3) :: pos_car, pos_dir
        DOUBLE PRECISION, DIMENSION(tAtoms,3) :: full_pos_car
        !DOUBLE PRECISION, DIMENSION(nAtoms, MAX_FPS) :: fps
        !DOUBLE PRECISION, DIMENSION(nAtoms, nAtoms, 3, MAX_FPS) :: dfps
        DOUBLE PRECISION, DIMENSION(nAtoms, MAX_NEIGHS, 3, MAX_FPS) :: temp_dfps

        DATA elementArray / "H","He","Li","Be","B","C","N","O", &
                "F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc", &
                "Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se", &
                "Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag", &
                "Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd", &
                "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta", &
                "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn", &
                "Fr","Ra","Ac","Th","Pa","U" /

        IF ( .NOT. (MAX_NEIGHS .EQ. 100)) THEN
            print *, "PyAMFF requires MAX_NEIGHS  be set to 100"
            print *, "Use 'neigh_modify one 100' in your input file "
            CALL EXIT(1)
        END IF
        DO i = 1, nelement
            ! Read from pos.con
            uniq_elements(i) = elementArray(uniqueNrs(i))
        END DO

        DO i = 1, nAtoms
            DO j = 1, nelement
                IF (atomicNumbers(i) == uniqueNrs(j)) THEN 
                    symbols(i) = j
                END IF
            END DO
        END DO

        ! make it so cell is passed as input and don't need to rewrite
        cell(1,1) = box(1)
        cell(1,2) = box(2)
        cell(1,3) = box(3)

        cell(2,1) = box(4)
        cell(2,2) = box(5)

        cell(2,3) = box(6)

        cell(3,1) = box(7)
        cell(3,2) = box(8)
        cell(3,3) = box(9)

        DO i = 1, nAtoms
            pos_car(i,1) = R(3*(i-1) + 1)
            pos_car(i,2) = R(3*(i-1) + 2)
            pos_car(i,3) = R(3*(i-1) + 3)
        END DO

        tnAtoms = tAtoms
        ALLOCATE(pool_pos_car(tAtoms,3))
        ALLOCATE(pool_ids(tAtoms))
        ALLOCATE(supersymbols(tAtoms))
        ALLOCATE(tneighs(tAtoms, MAX_NEIGHS))
        tcells = num_cells_dir(1) * num_cells_dir(2) * num_cells_dir(3)

        DO j=1, tAtoms
            pool_pos_car(j,:) = P(3*j-2:3*j)
            pool_ids(j) = tags(j)
            supersymbols(j) = symbols(tags(j))
        END DO
        
        DO i=1, tAtoms
            tneighs(i,:) = neighlist(MAX_NEIGHS*(i-1)+1:MAX_NEIGHS*i)
        END DO

        temp_dfps = 0.0
        nat = nAtoms
        tat = tAtoms
        !dfps = 0.0
        !fps = 0.0
        ! calcLammps: neighsDefined global with default of false
        !CALL calcfps(nAt, pos_car, cell, symbols, MAX_FPs, nelement, forceEngine, &
        !            fps, temp_dfps, neighs, num_neigh,num_cells_dir,usingLammps)
        !CALL ghost_dfps_correct(nelement, nAt, MAX_FPS, MAX_NEIGHS, num_neigh, neighs, &
        !    sub_num_neigh, sub_neighs, temp_dfps, dfps)

        CALL calcfps(nAt, pos_car, cell, symbols, nelement, forceEngine, &
                    sub_num_neigh, sub_neighs, num_neigh,num_cells_dir,usingLammps)
                    !fps, dfps, sub_num_neigh, sub_neighs, num_neigh,num_cells_dir,usingLammps)
        !CALL ghost_dfps_correct(nelement, nAt, MAX_FPS, MAX_NEIGHS, num_neigh, neighs, &
        !    sub_num_neigh, sub_neighs)
        CALL normalizeFPs(nelement, nAt, uniq_elements, MAX_FPs, MAX_NEIGHS, &
                                sub_num_neigh, sub_neighs)
                                !sub_num_neigh, sub_neighs, fps, dfps)

        CALL forwardCalc(sub_num_neigh, MAX_NEIGHS, sub_neighs,&
                        !fps, dfps, &
                        MAX_fps, MAXVAL(natoms_arr), MAXVAL(nhidneurons))

        U = Etotal

        DO i=1,nAtoms
            F(3*(i-1)+1) = forces(1,i) 
            F(3*(i-1)+2) = forces(2,i)
            F(3*(i-1)+3) = forces(3,i)
        END DO
        print *, "Energy = ", U
        CALL atomscleanup
    END SUBROUTINE

END MODULE

