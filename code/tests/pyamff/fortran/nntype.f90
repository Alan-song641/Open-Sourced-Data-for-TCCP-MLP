MODULE nnType

    IMPLICIT NONE
    PUBLIC

    !CHARACTER*8 :: fp_type
    ! FP paras
    INTEGER :: nelements, total_natoms
    INTEGER, DIMENSION(:), ALLOCATABLE :: natoms_arr
    INTEGER, DIMENSION(:), ALLOCATABLE :: nGs
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: atom_idx
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: fpminvs, fpmaxvs, diffs
    ! DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: magnitude, interceptScale
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: magnitude, interceptScale
    ! NN structure (dim = nhiddenlayers)
    INTEGER, DIMENSION(:), ALLOCATABLE :: nhidneurons
    ! NN paras
    INTEGER :: nhidlayers
    CHARACTER*8 :: actfuncId
    CHARACTER*16 :: scaler_type
    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: biases
    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: weights

    DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: in_weights
    ! DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: in_biases
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: in_biases
    DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: out_weights
    ! DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: out_biases
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: out_biases
    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: hid_weights
    ! DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: hid_biases
    DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: hid_biases

    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: in_gradients
    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: out_gradients
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:), ALLOCATABLE :: hid_gradients
    DOUBLE PRECISION :: slope, intercept

    ! Outputs
    DOUBLE PRECISION :: Etotal
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: forces
    ! Cohesive 
    LOGICAL :: use_cohesive_energy
    DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: coheEs

    CONTAINS 

    SUBROUTINE dummynn()
    END SUBROUTINE

END MODULE
