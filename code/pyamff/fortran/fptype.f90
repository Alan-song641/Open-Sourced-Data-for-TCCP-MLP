MODULE fpType
    !TYPE :: fingerprint_paras
    TYPE :: g1_paras
        CHARACTER*8 :: fp_type    !G1 or G2
        CHARACTER*8 :: species1
        INTEGER :: species1_code
        CHARACTER*8 :: species2
        INTEGER :: nFPs
        INTEGER :: startpoint
        INTEGER :: endpoint
        INTEGER :: currIndex
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: etas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: gammas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: zetas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: rss
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: r_cuts !probably we need to take 1/r_cuts
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: rmins
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: theta_ss
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: fp_maxs
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: fp_mins
    END TYPE g1_paras

    TYPE :: g2_paras
        CHARACTER*8 :: fp_type    !G1 or G2
        CHARACTER*8 :: species1
        INTEGER :: species1_code
        CHARACTER*8 :: species2
        INTEGER :: nFPs
        INTEGER :: startpoint
        INTEGER :: endpoint
        INTEGER :: currIndex
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: etas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: gammas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: lambdas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: zetas
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: r_cuts !probably we need to take 1/r_cuts
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: rmins
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: theta_ss
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: fp_maxs
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: fp_mins
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: g2   
        DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: dg2
    END TYPE g2_paras

    TYPE :: fingerprints
        INTEGER :: tnFPs
        INTEGER :: g1_startpoint, g1_endpoint
        INTEGER :: g2_startpoint, g2_endpoint
        TYPE(g1_paras), DIMENSION(:), ALLOCATABLE :: g1s
        TYPE(g2_paras), DIMENSION(:,:), ALLOCATABLE :: g2s
    END TYPE

    TYPE :: derivatives
        DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: dgdx
    END TYPE

    TYPE :: fingerprintsData
        !INTEGER :: ndists
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: pre_dgdxs
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: gs
        !DOUBLE PRECISION, DIMENSION(:, :, :), ALLOCATABLE :: arm
        !DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: dgdxs
        TYPE(derivatives), DIMENSION(:), ALLOCATABLE :: dgdxs
    END TYPE

    TYPE :: fingerprints_simple
        character(len = 100), dimension(:), allocatable :: g1
        character(len = 100), dimension(:), allocatable :: g2
    END TYPE fingerprints_simple

    CONTAINS

    SUBROUTINE dummy()
    END SUBROUTINE

END MODULE
