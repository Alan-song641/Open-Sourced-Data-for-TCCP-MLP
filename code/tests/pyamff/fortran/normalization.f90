MODULE normalize
    USE atomsProp
    USE fbp, only: find_loc_char, find_loc_int
    USE nnType, only: natoms_arr, nGs, atom_idx, fpminvs, fpmaxvs, diffs, magnitude, interceptScale

    IMPLICIT NONE
    PUBLIC

    CONTAINS

    !TODO
    SUBROUTINE normalizeFPs(nelement, nAtoms, uniq_elements, MAX_FPS, MAX_NNEIGHS, &
              nneighbors,neighborlists)
        IMPLICIT NONE
        ! Inputs
        INTEGER :: MAX_FPS, max_nneighs, nelement, nAtoms
        INTEGER, DIMENSION(nAtoms) :: nneighbors
        INTEGER, DIMENSION(nelement) :: idx_arr, idx_arr2
        INTEGER, DIMENSION(nAtoms, MAX_NNEIGHS) :: neighborlists
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements

        ! Variables
        INTEGER :: ptr, ptr2, i, j, j2, m, max_natoms

        idx_arr = 1
        DO i = 1, nAtoms
            ptr = supersymbols(i)
            atom_idx(idx_arr(ptr),ptr) = i
            fps(i,1:nGs(ptr)) = &
            ! fps(i,1:nGs(ptr))*magnitude(1:nGs(ptr),idx_arr(ptr),ptr) + interceptScale(1:nGs(ptr),idx_arr(ptr),ptr)
            fps(i,1:nGs(ptr))*magnitude(1:nGs(ptr),ptr) + interceptScale(1:nGs(ptr),ptr)
            idx_arr2 = 1
            DO j = 1, nneighbors(i)+1
                IF (j == 1) THEN
                    DO j2 = 1, 3
                        ! dfps(i,j,j2,1:nGs(ptr)) = dfps(i,j,j2,1:nGs(ptr))*magnitude(1:nGs(ptr),idx_arr(ptr),ptr)
                        dfps(i,j,j2,1:nGs(ptr)) = dfps(i,j,j2,1:nGs(ptr))*magnitude(1:nGs(ptr),ptr)
                    END DO
                ELSE
                    m = neighborlists(i,j-1)
                    IF (m <= nAtoms) THEN
                        ptr2 = supersymbols(m)
                        DO j2 = 1, 3
                            dfps(i,j,j2,1:nGs(ptr2)) = &
                            ! dfps(i,j,j2,1:nGs(ptr2))*magnitude(1:nGs(ptr2),idx_arr2(ptr2),ptr2)
                            dfps(i,j,j2,1:nGs(ptr2))*magnitude(1:nGs(ptr2),ptr2)
                        END DO
                        idx_arr2(ptr2) = idx_arr2(ptr2) + 1
                    END IF
                END IF
            END DO
            idx_arr(ptr) = idx_arr(ptr) + 1
        END DO

    END SUBROUTINE

    SUBROUTINE loadnormalizeParas(nAtoms, nelement, MAX_FPS, symbols, uniq_elements)
        IMPLICIT NONE

        ! Input
        INTEGER :: nAtoms, nelement, MAX_FPS
        INTEGER, DIMENSION(nAtoms) :: symbols
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements

        ! Variables
        INTEGER :: nG1, nG2, numGs, cidx, i, k
        INTEGER, DIMENSION(nelement) :: g1_endpts, g2_endpts
        DOUBLE PRECISION :: djunk, fpmin, fpmax
        CHARACTER*3 :: center, cjunk
        CHARACTER(LEN=30) :: line, G_type

        ! Output
        !DOUBLE PRECISION, DIMENSION(MAX_FPS, nelement) :: fpminvs, fpmaxvs, diffs

        ALLOCATE (natoms_arr(nelement))
        ALLOCATE (nGs(nelement))
        ALLOCATE (fpminvs(MAX_FPS, nelement))
        ALLOCATE (fpmaxvs(MAX_FPS, nelement))
        ALLOCATE (diffs(MAX_FPS, nelement))
        !print *, 'allocated done'
        g1_endpts = 0
        g2_endpts = 0
        nGs = 0
        fpminvs = 0
        fpmaxvs = 0
        diffs = 0
        DO k = 1, nelement
            natoms_arr(k) = COUNT(symbols .EQ. k)
        END DO

        OPEN (11, FILE='mlff.pyamff', status='old')
        READ (11,*) !skip #Fingerprint type
        READ (11,*) !fp_type 
        DO WHILE (line .NE. "#MachineLearning")
            READ (11,*) line !skip #type number
            IF (line .EQ. "#MachineLearning") GOTO 40
            READ (11,*) G_type, numGs

            IF (G_type .EQ. 'G1') THEN
                nG1 = numGs
                READ (11,*) !skip # center neighbor ...
                DO i = 1, nG1
                    READ (11,*) center, cjunk, djunk, djunk, djunk, fpmin, fpmax
                    cidx = find_loc_char(uniq_elements, center, SIZE(uniq_elements))
                    nGs(cidx) = nGs(cidx) + 1
                    IF (cidx .EQ. 1) THEN
                        fpminvs(i,cidx) = fpmin
                        fpmaxvs(i,cidx) = fpmax
                        g1_endpts(cidx) = g1_endpts(cidx)+1
                    ELSE
                        fpminvs(i-g1_endpts(cidx-1), cidx) = fpmin
                        fpmaxvs(i-g1_endpts(cidx-1), cidx) = fpmax
                        g1_endpts(cidx) = g1_endpts(cidx) + 1
                    END IF
                END DO

            ELSE IF (G_type .EQ. 'G2') THEN
                nG2 = numGs
                READ (11,*) !skip # center neighbor ...
                DO i = 1, nG2
                    READ (11,*) center, cjunk, cjunk, djunk, djunk, djunk, djunk, djunk, fpmin, fpmax
                    cidx = find_loc_char(uniq_elements, center, SIZE(uniq_elements))
                    nGs(cidx) = nGs(cidx) + 1
                    IF (cidx .EQ. 1) THEN
                        fpminvs(g1_endpts(cidx)+i,cidx) = fpmin
                        fpmaxvs(g1_endpts(cidx)+i,cidx) = fpmax
                        g2_endpts(cidx) = g2_endpts(cidx) + 1
                    ELSE
                        fpminvs(g1_endpts(cidx)+i-g2_endpts(cidx-1),cidx) = fpmin
                        fpmaxvs(g1_endpts(cidx)+i-g2_endpts(cidx-1),cidx) = fpmax
                        g2_endpts(cidx) = g2_endpts(cidx) + 1
                    END IF
                END DO
            END IF
        END DO
 40     BACKSPACE 11

    END SUBROUTINE

    SUBROUTINE normalizeParas(nelement)
        IMPLICIT NONE
        ! Inputs
        INTEGER :: nelement

        ! Variables
        INTEGER :: i, j, k, numfps
        DO i = 1, nelement
            numfps = nGs(i)
            DO j = 1, numfps
                IF (fpminvs(j,i) .EQ. -1.0d0) THEN
                    diffs(j,i) = 2.d0
                ELSE
                    diffs(j,i) = fpmaxvs(j,i) - fpminvs(j,i)
                    IF (diffs(j,i) .LT. 1.0E-8) THEN
                        fpminvs(j,i) = -1.0d0
                        diffs(j,i) = 2.d0
                    END IF
                END IF
            END DO
            magnitude(1:numfps,i) = 2.d0/diffs(1:numfps,i)
            interceptScale(1:numfps,i) = -2.d0*fpminvs(1:numfps,i)/diffs(1:numfps,i) - 1.0d0
        END DO

    END SUBROUTINE


    SUBROUTINE ghost_dfps_correct(nelement, nAtoms, MAX_FPS, max_nneighs, nneighbors, &
    neighborlists, sub_nneighbors, sub_nlist)
        IMPLICIT NONE
        INTEGER :: MAX_FPS, max_nneighs, nelement, nAtoms, len_subnlist
        INTEGER, DIMENSION(nAtoms) :: nneighbors, sub_nneighbors
        INTEGER, DIMENSION(nAtoms, max_nneighs) :: neighborlists
        INTEGER, DIMENSION(nAtoms, nAtoms) :: sub_nlist
        DOUBLE PRECISION, DIMENSION(nAtoms, max_nneighs, 3, MAX_FPS) :: in_dfps
        DOUBLE PRECISION, DIMENSION(nAtoms, nAtoms, 3, MAX_FPS) :: out_dfps
        ! Variables
        INTEGER :: ptr, ptr2, i, j, j2, m, max_natoms, m_loc, idx

        sub_nlist=0
        sub_nneighbors=0
        DO i = 1, nAtoms
          idx=1
          DO j = 1, nneighbors(i)
            IF (idx <= nAtoms) THEN
              m=neighborlists(i,j)
              IF ((ANY(sub_nlist(i,:)==m)) .OR. (m==i)) THEN
                CONTINUE
              ELSE 
                sub_nlist(i,idx)=m
                sub_nneighbors(i)=sub_nneighbors(i)+1
                idx=idx+1
              END IF
            END IF
          END DO
        END DO

        out_dfps=0
    END SUBROUTINE

END MODULE
