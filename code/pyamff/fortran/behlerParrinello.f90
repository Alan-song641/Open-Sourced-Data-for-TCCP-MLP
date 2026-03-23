MODULE fbp
    USE nlist
    USE atomsProp
    !USE ieee_arithmetic
    IMPLICIT NONE
    PUBLIC

    CONTAINS

    SUBROUTINE fg1(Rijs, nFPs, etas,r_ss,r_cuts, outputs)

        IMPLICIT NONE
        !Define input values
        INTEGER :: nFPs
        DOUBLE PRECISION, DIMENSION(nFPs) :: etas, r_ss, r_cuts
        DOUBLE PRECISION :: Rijs

        !Define values needed for calculation
        DOUBLE PRECISION, DIMENSION(nFPs) :: outputs
        DOUBLE PRECISION, PARAMETER :: Pi = 4.*ATAN(1.0_8)

        outputs = EXP(-((Rijs-r_ss)**2/(r_cuts**2))*etas)*(0.5*(COS(Pi*Rijs/r_cuts)+1))

        RETURN
    END SUBROUTINE

    SUBROUTINE fdg1(Rij, nFPs, etas,r_ss,r_cuts, outputs)

        IMPLICIT NONE
        !Define input values
        INTEGER :: nFPs
        DOUBLE PRECISION :: Rij
        DOUBLE PRECISION, DIMENSION(nFPs) :: etas, r_ss, r_cuts
        !Define parameters
        DOUBLE PRECISION, PARAMETER :: Pi = 4.*ATAN(1.0_8)
        !Define outputs
        DOUBLE PRECISION, DIMENSION(nFPs) :: outputs

        outputs = (((-etas*(Rij-r_ss)*(COS(Pi*Rij/r_cuts) + 1))/(r_cuts**2)) &
          - (0.5*(Pi/r_cuts)*SIN(Pi*Rij/r_cuts)))&
          * EXP(-etas*((Rij-r_ss)**2)/(r_cuts**2))

    END SUBROUTINE


    SUBROUTINE commonparts(npairs, pairs, inv_rcut, &
                           pair_sins, fcuts, fexp)
        IMPLICIT NONE
        !Define input vales
        INTEGER :: npairs
        DOUBLE PRECISION, DIMENSION(npairs) :: pairs
        DOUBLE PRECISION :: inv_rcut

        !Define output vales
        DOUBLE PRECISION, DIMENSION(npairs) :: pair_sins
        DOUBLE PRECISION, DIMENSION(npairs) :: fcuts, fexp

        !Define parameters
        DOUBLE PRECISION, PARAMETER :: Pi = 4.*ATAN(1.0_8)
        
        pair_sins = SIN(Pi*pairs*inv_rcut)
        fcuts = 0.5*(COS(Pi*pairs*inv_rcut)+1)
        !only for g2
        fexp = EXP(-pairs**2)
    END SUBROUTINE

    SUBROUTINE calcTriplet(nAtoms, tnAtoms, MAX_NEIGHS, rcut, npairs, &
                           ntriplets)
                           !, angles, angle_indices)
        IMPLICIT NONE

        ! Define input values
        INTEGER :: nAtoms, MAX_NEIGHS   !nAtoms: number of atoms in original cell
        INTEGER :: tnAtoms !number of atoms in the whole supercell
        INTEGER :: npairs !number of pairs in the whole supercell
        DOUBLE PRECISION :: rcut, rcut2

        ! Define variables
        INTEGER :: i , i_neigh_1, j_neigh, k_neigh, ij_pair_loc, ik_pair_loc, jk_pair_loc
        INTEGER :: p_size, i_neigh_2, s_neigh, l_neigh, s_pair_loc, l_pair_loc
        INTEGER, DIMENSION(1) :: jk_index !used to fetch the 3rd side
        DOUBLE PRECISION, DIMENSION(3) :: v1_unit, v2_unit, v3_unit

        ! Define output values
        INTEGER :: ntriplets
        ! Place holders
        DOUBLE PRECISION :: a1, b1, c1, dist_3

        rcut2 = rcut**2
        ntriplets = 0
        angles = 0
        angles_indices = 0
        DO i = 1, nAtoms !for each atom 
            DO j_neigh = 1, num_eachpair(i) !for each pair for that atom
                ij_pair_loc = pair_info(i,j_neigh)
                IF (ij_pair_loc==0) THEN
                   CYCLE
                END IF
                v1_unit(:)=unitvects_pair(1,ij_pair_loc,:)
                DO k_neigh = j_neigh+1, num_eachpair(i) !for next neighbor atom
                    !only calculate triplet that has all distance in the cutoff range
                    !select the pairs that are led by neighbors of atom i
                    !pair_info(pair_global_indices(ij_pair),:) give index of pairs
                    ik_pair_loc = pair_info(i,k_neigh)
                    s_neigh = pair_global_indices(2, ij_pair_loc)
                    l_neigh = pair_global_indices(2, ik_pair_loc)

                    IF (l_neigh < s_neigh) THEN ! make sure j is always the smaller side
                        CYCLE
                    END IF

                    IF (s_neigh > tnAtoms) THEN
                       CYCLE
                    END IF
                    ! when pair_indices(j) < nAtoms but all pairs belong to this atom have been included in atoms with smaller index,
                    ! then j_ps =0; for pair_indices(j)= nAtoms case hase been exclued in j loop, because ps = 0
                    p_size = SIZE(pair_global_indices(2, pair_info(s_neigh,1:num_eachpair(s_neigh))))
                    IF (p_size .EQ. 0) THEN
                        CYCLE
                    END IF
                    ! jk_index = find_loc_int(pair_global_indices(2, pair_info(s_neigh,1:num_eachpair(s_neigh))), &
                    !                         l_neigh, &
                    !                         p_size)
                    jk_index = findloc(pair_global_indices(2, pair_info(s_neigh,1:num_eachpair(s_neigh))), l_neigh)
                    IF (jk_index(1) /= 0) THEN !i.e find_loc_int found an index, i.e atom in pairs array
                        jk_pair_loc = pair_info(s_neigh,jk_index(1))
                        v2_unit(:) = unitvects_pair(1, ik_pair_loc, :)  !vect_ik
                        IF ((gvects(ik_pair_loc) < 0).OR.(gvects(ij_pair_loc)<0)) THEN !!!IF ghost vector is a vector in triangle 
                           a1 = SUM(v1_unit * v2_unit) !!Dot Product, i.e Theta
                           dist_3 = pairs(ij_pair_loc) * pairs(ij_pair_loc) +  &
                                    pairs(ik_pair_loc) * pairs(ik_pair_loc) -  &
                                    2 * pairs(ij_pair_loc) * pairs(ik_pair_loc) * a1

                           v3_unit(:) = (pairs(ik_pair_loc) * v2_unit(:)  - pairs(ij_pair_loc)*v1_unit(:))/sqrt(dist_3) 
                        ELSE !!Otherwise there is not ghost vector in the Triangle; BUT WE NEED TO COMPUTE DIST3 STILL,WE CAN HAVE DIST3>Rc but NO GHOST VECTOR
                          !cosine is even, can do this for sure
                           a1 = SUM(v1_unit*v2_unit)
                           v3_unit(:) = unitvects_pair(1, jk_pair_loc, :) !vect_jk
                           dist_3 = pairs(ij_pair_loc) * pairs(ij_pair_loc) +  &
                                    pairs(ik_pair_loc) * pairs(ik_pair_loc) -  &
                                    2 * pairs(ij_pair_loc) * pairs(ik_pair_loc) * a1
                        END IF
                        IF (dist_3 < rcut2) THEN
                           ntriplets = ntriplets + 1
                           IF (ABS(a1) > 0.999999999999) THEN
                               angles(1, ntriplets) = 0.d0
                           ELSE 
                               angles(1, ntriplets) = ACOS(a1)
                           END IF
                           !angles(1, ntriplets) = ACOS(SUM(v1_unit*v2_unit))
                           b1 = SUM(-v1_unit*v3_unit)
                           IF (ABS(b1) > 0.999999999999) THEN
                               angles(2, ntriplets) = 0.d0
                           ELSE 
                               angles(2, ntriplets) = ACOS(b1)
                           END IF
                           c1 = SUM(v2_unit*v3_unit)
                           IF (ABS(c1) > 0.999999999999) THEN
                               angles(3, ntriplets) = 0.d0
                           ELSE 
                               angles(3, ntriplets) = ACOS(c1)
                           END IF
                           IF (ij_pair_loc<0) THEN
                              print*, 'NAN', ij_pair_loc
                           END IF
                           angles_indices(1, ntriplets) = ij_pair_loc  !j
                           angles_indices(2, ntriplets) = ik_pair_loc  !k
                           angles_indices(3, ntriplets) = jk_pair_loc  !jk_index

                        END IF
 
                    END IF
                END DO
            END DO
        END DO

    END SUBROUTINE

    SUBROUTINE fg2(nFPs, theta_ss, etas, zetas, lambdas, theta, fexpt, fct, &
                   outputs, f_thetas_1, f_thetas, f_rs)
        IMPLICIT NONE
        !Define input values
        INTEGER :: nFPs
        DOUBLE PRECISION, DIMENSION(nFPs) :: etas, theta_ss, zetas, lambdas
        DOUBLE PRECISION :: theta
        !DOUBLE PRECISION, DIMENSION(3) :: Rs   !(Rij, Rik, Rjk)
        DOUBLE PRECISION :: fexpt !fexp_ij, fexp_ik, fexp_jk
        DOUBLE PRECISION :: fct   !fc_rij, fc_rik, fc_rjk
        DOUBLE PRECISION, DIMENSION(3, 3) :: unitvects

        !Define values needed for calculation
        DOUBLE PRECISION, PARAMETER :: Pi = 4.*ATAN(1.0_8)
        DOUBLE PRECISION, PARAMETER :: radians = Pi/180.0_8 !same as np.radians

        !Define output arrays
        DOUBLE PRECISION, DIMENSION(nFPs) :: outputs
        DOUBLE PRECISION, DIMENSION(nFPs) :: f_thetas, f_rs, f_thetas_1

        f_thetas_1 = 1 + lambdas * COS(theta - theta_ss)
        f_thetas = f_thetas_1 ** (zetas-1)
        f_rs = fexpt ** etas
        outputs = f_thetas * f_thetas_1 * f_rs * fct * 2 ** (1-zetas)

    END SUBROUTINE

    SUBROUTINE fg2_dg(nFPs, theta, inv_rcut, fexpt, fct, &
                      dexps, dth_ijk, dfc, calc_dth, &
                      theta_ss, etas, zetas, lambdas, &
                      g2, dg2)
        IMPLICIT NONE
        !Define input values
        INTEGER :: nFPs
        DOUBLE PRECISION :: theta, inv_rcut, fexpt, fct
        DOUBLE PRECISION, DIMENSION(3,3) :: dexps, dfc !dexp_i, dexp_j, dexp_k
        DOUBLE PRECISION, DIMENSION(3,3) :: dth_ijk
        DOUBLE PRECISION, DIMENSION(nFPs) :: etas, reduced_etas, theta_ss, zetas, lambdas

        !Define values needed for calculation
        INTEGER :: i, j, k
        DOUBLE PRECISION, PARAMETER :: Pi = 4.*ATAN(1.0_8)
        DOUBLE PRECISION, PARAMETER :: radians = Pi/180.0_8 !same as np.radians
        DOUBLE PRECISION :: cos_theta, inv_sin
        LOGICAL :: calc_dth
  
        DOUBLE PRECISION, DIMENSION(nFPs) :: thetas_shft, f_thetas_m1, f_rs, f_thetas_base,f_thetas
        DOUBLE PRECISION, DIMENSION(nFPs) :: dth_coeff, temp, temp1, temp2,f_rs_fct
        DOUBLE PRECISION, DIMENSION(3,nFPs) :: dth_i
        DOUBLE PRECISION, DIMENSION(3,3,nFPs) :: dfrs_i
  
        !Define output arrays
        DOUBLE PRECISION, DIMENSION(nFPs) :: g2
        DOUBLE PRECISION, DIMENSION(3,3,nFPs) :: dg2
 
        !For derivative calculation 
        dg2 = 0.0
        !inv_rcut = 1/(r_cut * r_cut)
        reduced_etas = etas * inv_rcut *inv_rcut
        thetas_shft = theta - theta_ss
        f_thetas_base = 1.0 + lambdas * COS(thetas_shft)
        f_thetas_m1 = f_thetas_base ** (zetas-1.0)
        f_thetas = f_thetas_m1 * f_thetas_base
        f_rs = fexpt ** reduced_etas !TODO: reduced_etas
        temp1 = 2.0**(1-zetas)
        temp2 = f_thetas * temp1
        f_rs_fct = f_rs * fct
        !calculate G2
        g2 = temp2 * f_rs_fct  !fct=fcuts(1)*fcuts(2)*fcuts(3)
        !for derivative of f_rs
        temp = 2.0 * reduced_etas * f_rs * fct
        ! swap order for a speed improvement 2025-02
        DO j = 1, 3 
           DO i = 1, 3 !center on i
              dfrs_i(i,j,:) = dexps(i, j) * temp !center on i w.r.t 1
           END DO
        END DO
        !derivative of f_theta
        IF (calc_dth) THEN
            !   swap order for speed
            dth_coeff = f_thetas_m1 * zetas * (- lambdas * SIN(thetas_shft)) * f_rs_fct * temp1 
            DO j = 1, 3  !loop over xyz direction
                DO i =1, 3   !w.r.t.
                    dg2(i,j,:) = dth_coeff * dth_ijk(i,j) + & !center on _i w.r.t i
                        (dfrs_i(i,j,:) + f_rs * dfc(i,j))*temp2
                END DO
            END DO
        ELSE
            ! swap order for speed
            DO j = 1, 3  
              DO i=1,3 !w.r.t.
                 dg2(i,j,:) = (dfrs_i(i,j,:) + f_rs *dfc(i,j))*temp2
              END DO
           END DO
        END IF
    END SUBROUTINE

!------------------------------------------------------------------------------------!
! find loc function
!------------------------------------------------------------------------------------!

    FUNCTION find_loc_int(arr, val, size_arr)

        INTEGER,INTENT(IN) :: size_arr   
        INTEGER,INTENT(IN),DIMENSION(size_arr) :: arr
        INTEGER,INTENT(IN) :: val
        INTEGER :: i
        INTEGER :: find_loc_int

        find_loc_int = 0
        DO i = 1, size_arr
            IF (arr(i) .EQ. val) THEN
                find_loc_int = i
                EXIT
            END IF
        END DO

        RETURN
    END FUNCTION

    FUNCTION find_loc_char(arr, val, size_arr)

        INTEGER,INTENT(IN) :: size_arr
        CHARACTER*2, INTENT(IN),DIMENSION(size_arr) :: arr
        CHARACTER*2,INTENT(IN) :: val
        INTEGER :: i
        INTEGER :: find_loc_char

        find_loc_char = 0
        DO i = 1, size_arr
            IF (arr(i) .EQ. val) THEN
                find_loc_char = i
                EXIT
            END IF
        END DO

        RETURN
    END FUNCTION

    FUNCTION find_loc_dbl(arr, val, size_arr)

        INTEGER,INTENT(IN) :: size_arr
        DOUBLE PRECISION,INTENT(IN),DIMENSION(size_arr) :: arr
        DOUBLE PRECISION,INTENT(IN) :: val
        INTEGER :: i
        INTEGER :: find_loc_dbl
        find_loc_dbl = 0
        DO i = 1, size_arr
            IF (arr(i) .EQ. val) THEN
                find_loc_dbl = i
                EXIT
            END IF
        END DO

        RETURN
    END FUNCTION


END MODULE
