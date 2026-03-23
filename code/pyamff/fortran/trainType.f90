MODULE trainType

    IMPLICIT NONE
    PUBLIC

    LOGICAL :: energy_training, force_training
    INTEGER :: nimages, nAtimg, img_idx, curr_epoch
    INTEGER :: epoch_img_idx
    INTEGER, DIMENSION(:), ALLOCATABLE :: natomsE, natomsF, nAtimg_ptr
    INTEGER, DIMENSION(:,:), ALLOCATABLE :: nneighbors_img, symbols_img
    INTEGER, DIMENSION(:,:,:), ALLOCATABLE :: neighborlists_img
    DOUBLE PRECISION :: floss_const
    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: input_fps !calculated fps of all images 
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:), ALLOCATABLE :: input_dfps !calculated dfps of all images 
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:), ALLOCATABLE :: all_neurons !all neuron values of all images
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:), ALLOCATABLE :: in_gradients_img !in_gradients of all images
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:,:), ALLOCATABLE :: hid_gradients_img !hid_gradients of all images
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:), ALLOCATABLE :: out_gradients_img ! out_gradients of all images
    DOUBLE PRECISION, DIMENSION(:,:,:,:,:,:), ALLOCATABLE :: layer_backgrad
    DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: bias_grad, bias_grad_e, bias_grad_f
    DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: weight_grad, weight_grad_e, weight_grad_f
    DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: inputE, targetE
    DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: inputF, targetF

    CONTAINS

    SUBROUTINE dummyTrain()
    END SUBROUTINE

END MODULE
