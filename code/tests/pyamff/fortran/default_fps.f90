module default_fps
    use fptype, only : fingerprints_simple
    implicit none
    private
    public :: fp_paras_gen

    contains
        subroutine fp_paras_gen(file_name)

            ! WRITE THE FPPARAS

            !declare var
            character(len = 2), dimension(:), allocatable  :: elements
            character(len = *), intent(in)                 :: file_name
            integer                                        :: i, num_g1, num_g2
            type(fingerprints_simple)                      :: fp

            elements = get_elements()

            ! If size elements > 4 then exit out to save memory
            if (size(elements) > 4) then
                stop 'ERROR: TOO MANY ELEMENTS TO CREATE A PROPER DEFAULT ' // file_name &
                     // ' POVIDE A USER BUILT ' // file_name
            end if

            print *, 'Created fingerprints at fpParas.dat'
            ! get finger prints
            fp = make_default_symmetry_functions(elements)
            ! get number of g1 and g2
            num_g1 = size(fp%g1)
            num_g2 = size(fp%g2)

            ! open file
            open(12, file = file_name, status = 'replace')

            write(12, '(A9)') '# FP type'
            write(12, '(A2)') 'BP'

            ! write elements
            write(12, '(A9)') '#Elements'
            do i = 1, size(elements)
                if (i < size(elements)) then
                    write(12, '(A2, A1)', advance = 'no') adjustl(elements(i)), ' '
                else
                    write(12, '(A2)') adjustl(elements(i))
                end if
            end do

            ! write cohesive terms, by default zero
            do i = 1, size(elements)
                if (i < size(elements)) then
                    write(12, '(A4)', advance = 'no') '0.0 '
                else
                    write(12, '(A3)') '0.0'
                end if
            end do

            ! write number of g1 and g2
            write(12, '(A11)') '#nG1s #nG2s'
            write(12, '(I0, A1)', advance = 'no') num_g1, ' '
            write(12, '(I0)') num_g2

            ! header line for g1
            write(12, '(A57)') '#type   CentralElement   Neighbor   eta      r_s      r_c'
            ! write g1 fingerprints
            do i = 1, size(fp%g1)
               write(12, '(A2, A4)', advance = 'no') 'G1', '    '
               write(12, *)fp%g1(i)
            end do

            ! header line for g2
            write(12, '(A89)') '#type   CentralElement   Neighbor1   Neighbor2   eta      zetas    lambdas   r_s      r_c'
            ! write g2 fingerprints
            do i = 1, size(fp%g2)
                write(12, '(A2, A4)', advance = 'no') 'G2', '    '
               write(12, *) fp%g2(i)
            end do

            ! close file
            close(12)

            ! deallocate memory
            deallocate(elements)
            deallocate(fp%g1)
            deallocate(fp%g2)

            contains
                function make_symmetry_functions(elements, etas, a_etas, r_s, zetas, lambdas, theta_s, r_c) result(fp_arr)
                    ! Helper function to create Gaussian symmetry functions.
                    ! Returns a list of dictionaries with symmetry function parameters
                    ! in the format expected by the Gaussian class.

                    ! Parameters
                    ! ----------
                    ! elements : list of str
                    !     List of element types to be observed in this fingerprint.
                    ! f_type : str
                    !     Either G2, G4 fingerprints
                    ! etas : list of floats but strings
                    !     eta values to use in G2 fingerprints
                    ! a_etas : list of floats but strngs
                    !     eta values to use in G4 fingerprints
                    ! r_s: list of floats but strings
                    !     offset values to use in G2 fingerprints
                    ! zetas : list of floats but strings
                    !     zeta values to use in G4 fingerprints
                    ! lambdas : list of floats but strings
                    !     lambda values to use in G4 fingerprints
                    ! r_c : list of floats but strings
                    !     cutoff values to use in G2 and G4 fingerprints
                    ! Returns
                    ! -------
                    ! fp_arr : nested arr where each line is a fingerprint line
                    !     example G1: [[f_type], [element], [eta], [r_s], [r_c]]
                    !     example G2: [[f_type], [element1],[element2], [eta], [zeta], [lambda], [r_s], [r_c]]

                    ! Declare input variables
                    character(len = 9), dimension(:), allocatable, intent(in) :: etas, a_etas, r_s, zetas, lambdas, &
                                                                                 theta_s, r_c
                    character(len = 2), dimension(:), allocatable, intent(in)  :: elements
                    integer                                                    :: num_fp, idx, n_ele_1, n_ele_2, &
                                                                                  n_ele_3, n_etas, n_rs, n_zetas,&
                                                                                  n_lambdas, n_theta_s, n_rc
                    character(len = 100)                                       :: string ! dummy to hold fp string

                    ! Declare output variable
                    type(fingerprints_simple)                                         :: fp_arr

                    ! allocate memory for g1 and g2
                    num_fp = (size(elements) ** 2)  * size(etas) * size(r_s)
                    allocate(fp_arr%g1(num_fp))
                    num_fp = (size(elements) ** 3) * size(a_etas) * size(r_s) * size(zetas) * size(lambdas) * &
                              size(theta_s) * size(r_c)
                    allocate(fp_arr%g2(num_fp))

                    ! Loop for fps
                    idx = 1 ! used to get correct index into fp_arr

                    ! Loop for G1 functions
                    do n_ele_1 = 1, size(elements)
                        do n_ele_2 = 1, size(elements)
                            do n_etas = 1, size(etas)
                                do n_rs = 1, size(r_s)
                                    do n_rc = 1, size(r_c)
                                        write(string, '(A10, A10, A12, A12, A12)') elements(n_ele_1), elements(n_ele_2),&
                                                                                   etas(n_etas), r_s(n_rs), r_c(n_rc)
                                        fp_arr%g1(idx) = trim(string)
                                        idx = idx + 1
                                    end do
                                end do
                            end do
                        end do
                    end do


                    ! Loop for G2 functions
                    if (size(fp_arr%g2) > 0) then
                        idx = 1 ! used to get correct index into fp_arr
                        do n_ele_1 = 1, size(elements)
                            do n_ele_2 = 1, size(elements)
                                do n_ele_3 = 1, size(elements)
                                    do n_etas = 1, size(a_etas)
                                        do n_zetas = 1, size(zetas)
                                            do n_lambdas = 1, size(lambdas)
                                                do n_theta_s = 1, size(theta_s)
                                                    do n_rc = 1, size(r_c)
                                                        write(string, '(A10, A10, A10, A12, A12, A12, A12, A12)') &
                                                              elements(n_ele_1), elements(n_ele_2), elements(n_ele_3),&
                                                              a_etas(n_etas), zetas(n_zetas), lambdas(n_lambdas), & 
                                                              theta_s(n_theta_s), r_c(n_rc)   
                                                        fp_arr%g2(idx) = trim(string) 
                                                        idx = idx + 1
                                                    end do 
                                                end do 
                                            end do 
                                        end do 
                                    end do 
                                end do 
                            end do 
                        end do
                    end if
                    
                end function make_symmetry_functions

                function logspace(start, stop, num) result(logspace_arr)
                    ! returns a logspace_arr from 10^start_exp to 10^stop_exp
                    !
                    ! python reference : np.logspace(np.log10(start_exp), np.log10(stop_exp), num = 4)
                    !
                    ! Parameters
                    ! ----------
                    ! start : float
                    !         should be log10(some starting exponent)
                    ! stop  : float
                    !         should be log10(some ending exponent)
                    ! num       : integer
                    ! ----------
                    ! logspace formula : 10^(start + (i + 1) * (stop - start) / (num - 1)

                    ! Declare input variables
                    real, intent(in)                               :: start, stop 
                    integer, intent(in)                            :: num
                    integer                                        :: i
                    character(len = 25), dimension(:), allocatable :: logspace_arr
                    real(8)                                        :: val 
                    
                    ! Allocate the array
                    allocate(logspace_arr(num))

                    ! Fill logspace array using formula
                    if (num == 1) then
                        write(logspace_arr(num), '(F9.6)') 10.0d0 ** start
                    else
                        do i = 1, num
                            val = 10.0d0 ** (start + (i - 1) * (stop - start) / (num - 1))
                            write(logspace_arr(i), '(F9.6)') val ! convert logspace val to string
                        end do
                    end if

                end function logspace

                function make_default_symmetry_functions(elements, fp_level) result(fp_arr)
                    !Makes default set of G2 and G4 symmetry functions.
                    !
                    !Parameters
                    !----------
                    !elements : list of str
                    !    List of the elements, as in: ["C", "O", "H", "Cu"].
                    !fp_level : string
                    !    "high", "normal", or "low" 
                    !Returns
                    !-------
                    !G : derived type with attributes of g1 and g2 that are both string arrays
                    !    The generated symmetry function in strings
                    !    example g1: ['C C eta r_s r_c', 'C H eta r_s, r_c', ...]
                    !    example g2: ['C C H eta zeta lambda r_s r_c', 'C H H, eta, zeta, lambda, r_s, r_c', ...]

                    ! Declare input variables
                    character(len = 2), dimension(:), allocatable, intent(in) :: elements
                    character(len = 6), optional, intent(in)                  :: fp_level
                    character(len = 6)                                        :: fp_level_local

                    ! Declare default variables
                    character(len = 9), dimension(:), allocatable             :: etas, a_etas, zetas, lambdas, r_s, &
                                                                                 theta_s, r_c
                    ! Declare output variables
                    type(fingerprints_simple)                                        :: fp_arr

                    ! Set default variables
                    allocate(theta_s(1))
                    allocate(r_s(1))
                    allocate(r_c(1))
                    write(r_s(1), '(F9.6)') 0.
                    write(r_c(1), '(F9.6)') 6.5
                    write(theta_s(1), '(F9.6)') 0.

                    if (present(fp_level)) then
                        fp_level_local = trim(adjustl(fp_level))
                    else
                        fp_level_local = 'high'
                    end if

                    fp_level_local = trim(fp_level_local)
                    
                    if (fp_level_local == 'high') then
                        allocate(zetas(2))
                        allocate(lambdas(2))
                        allocate(a_etas(1))
                        etas = logspace(log10(0.05), log10(5.), 4)
                        write(zetas(1), '(F9.6)') 1.
                        write(zetas(2), '(F9.6)') 4.
                        write(lambdas(1), '(F9.6)') 1.
                        write(lambdas(2), '(F9.6)') -1.
                        write(a_etas(1), '(F9.6)') 0.005

                    else if (fp_level_local == 'normal') then
                        allocate(zetas(1))
                        allocate(lambdas(1))
                        allocate(a_etas(1))
                        etas = logspace(log10(0.05), log10(5.), 2)
                        write(zetas(1), '(F9.6)') 1.
                        write(lambdas(1), '(F9.6)') 1.
                        write(a_etas(1), '(F9.6)') 0.005

                    else if (fp_level_local == 'low') then
                        allocate(zetas(0))
                        allocate(lambdas(0))
                        allocate(a_etas(0))
                        etas = logspace(log10(0.05), log10(5.), 1)
                    
                    end if

                    fp_arr = make_symmetry_functions(elements, etas, a_etas, r_s, zetas, lambdas, theta_s, r_c) 
                    
                    deallocate(etas)
                    deallocate(zetas)
                    deallocate(lambdas)
                    deallocate(a_etas)
                    deallocate(theta_s)
                    deallocate(r_s)
                    deallocate(r_c)

                end function make_default_symmetry_functions

                function get_elements() result(elements)
                    ! Reads the POSCAR and returns an array of elements

                    ! Declare local variables
                    character(len = 200)                          :: line ! store the lines of POSCAR
                    integer                                       :: ele_count, i, idx, ios
                    character(len = 2)                            :: element
                    ! Declare output variable
                    character(len = 2), dimension(:), allocatable :: elements

                    ! Open the POSCAR
                    open(unit = 10, file ='POSCAR', status = 'old', action = 'read')

                    ! Skip to the 6th line
                    do i = 1, 5
                        read(10, '(A)', iostat=ios)
                        if (ios /= 0) then
                           print *, "Error: Could not reach the 6th line."
                           close(10)
                           stop
                        end if
                    end do

                    ! Read the 6th line
                    read(10, '(A)', iostat=ios) line
                    if (ios /= 0) then
                        print *, "Error reading line 6."
                        close(10)
                        stop
                    end if

                    ! Initialize index and element count
                    ele_count = 0
                    idx = 1

                    ! find number of elements
                    line = trim(adjustl(line))
                    do i = 1, len(line)
                        ! checks if current character in line is not a space and 
                        ! if next character is also not a space, if so then i:i+1
                        ! is an element of len ==2 like Au or Pt
                        if (line(i : i) /= ' ' .and. line(i + 1 : i + 1) /= ' ') then
                            ele_count = ele_count + 1
                        ! checks if character is a capital letter and if i + 1 character
                        ! is a space, if so then i:i is an element of len == 1
                        ! for example, H or C
                        else if ((line(i : i) >= 'A' .and. line(i : i) <= 'Z') .and. line(i + 1 : i + 1) == ' ') then
                            ele_count = ele_count + 1
                        end if
                    end do

                    ! Allocate array based on the counted words
                    allocate(elements(ele_count))

                    ! Assign element to elements
                    do i = 1, len(line)
                        if (line(i:i) /= ' ' .and. line(i + 1 : i + 1) /= ' ') then
                            write(elements(idx), '(A2)') line(i : i + 1)
                            idx = idx + 1
                        else if ((line(i : i) >= 'A' .and. line(i : i) <= 'Z') .and. line(i + 1 : i + 1) == ' ') then
                            write(elements(idx), '(A2)') line(i : i)
                            idx = idx + 1
                        end if
                    end do

                end function get_elements
        end subroutine fp_paras_gen
end module default_fps
