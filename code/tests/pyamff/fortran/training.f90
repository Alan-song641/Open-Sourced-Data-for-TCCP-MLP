MODULE training
    USE nnType
    USE trainType
    USE neuralnetwork
    USE fnnmodule
    USE lossgrad
    !USE adam
    USE opts
    IMPLICIT NONE

    CONTAINS

    SUBROUTINE train_init(nAtoms,nelement,atomicNrs,uniqueNrs)
        USE fpCalc
        IMPLICIT NONE
        INTEGER :: nAtoms, nelement
        INTEGER, PARAMETER :: max_fps = 100
        INTEGER, DIMENSION(nAtoms) :: atomicNrs
        INTEGER, DIMENSION(nelement) :: uniqueNrs

        !Variables
        INTEGER :: i, j
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements
        CHARACTER*3, DIMENSION(92) :: elementArray
        CHARACTER*100 :: mlff_file
        INTEGER, DIMENSION(nAtoms) :: symbols

        !Temporary due to the ASE-Calc code changes !!
        !TODO: Need better way
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
                IF (atomicNrs(i) == uniqueNrs(j)) THEN
                    symbols(i) = j
                END IF
            END DO
        END DO

        mlff_file='mlff.pyamff'
        CALL read_mlff(nelement, max_fps, mlff_file)
        CALL update_atomInfo(nAtoms, nelement, MAX_FPS, symbols)
        CALL init_backward

    END SUBROUTINE

    SUBROUTINE trainExec(nAtoms,pos_car,box,atomicNumbers,nelement,uniqueNrs,opt_type,max_epoch,force_coeff)
        USE nlist
        USE fpCalc
        USE normalize
        IMPLICIT NONE
        CHARACTER(*) :: opt_type
        INTEGER :: nAtoms, max_epoch
        INTEGER :: NAT, i, j
        REAL, DIMENSION(9), INTENT(IN) :: box
        INTEGER, DIMENSION(nAtoms) :: atomicNumbers
        DOUBLE PRECISION, OPTIONAL :: force_coeff

        INTEGER, PARAMETER :: MAX_FPs = 100
        INTEGER, PARAMETER :: MAX_NEIGHS = 100
        INTEGER, PARAMETER ::forceEngine = 1
        DOUBLE PRECISION, DIMENSION(nAtoms,3) :: pos_car
        INTEGER, DIMENSION(nAtoms) :: symbols
        CHARACTER*3, DIMENSION(nAtoms) :: atomicSymbols

        DOUBLE PRECISION, DIMENSION(3,3) :: cell

        INTEGER :: nelement
        INTEGER, DIMENSION(nelement) :: uniqueNrs
        CHARACTER*20 :: filename
        CHARACTER*3, DIMENSION(nelement) :: uniq_elements
        DOUBLE PRECISION, DIMENSION(nAtoms,3) :: pos_dir
        DOUBLE PRECISION, DIMENSION(3,3) :: dir2car

        DOUBLE PRECISION, DIMENSION(nAtoms, MAX_FPS) :: fps
        DOUBLE PRECISION, DIMENSION(nAtoms, MAX_NEIGHS, 3, MAX_FPS) :: dfps
        CHARACTER*3, DIMENSION(92) :: elementArray
        INTEGER, DIMENSION(nAtoms, MAX_NEIGHS) :: neighs
        INTEGER, DIMENSION(nAtoms) :: num_neigh

        REAL :: start, finish
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

        nat = nAtoms

        dfps = 0.0
        fps = 0.0

        CALL calcfps(nAt, pos_car, cell, symbols, MAX_FPs, nelement, forceEngine, &
                     fps, dfps, neighs, num_neigh)

        CALL normalizeFPs(nelement, nAt, uniq_elements, symbols, MAX_FPS, MAXVAL(num_neigh), &
                          num_neigh, neighs, fps, dfps(:,1:MAXVAL(num_neigh)+1,:,:))

        CALL forwardCalc(num_neigh, MAXVAL(num_neigh),neighs(:,1:MAXVAL(num_neigh)), symbols, &
                         fps, dfps(:,1:MAXVAL(num_neigh)+1,:,:), &
                         MAXVAL(nGs), MAXVAL(natoms_arr), MAXVAL(nhidneurons))
        IF (energy_training .EQV. .TRUE.) THEN
            inputE(img_idx) = Etotal-intercept
        END IF

        IF (force_training .EQV. .TRUE.) THEN
            IF (img_idx==1) THEN
                inputF(1:3,1:total_natoms)=forces(1:3,1:total_natoms)
            ELSE
                inputF(1:3,nAtimg_ptr(img_idx)+1:nAtimg_ptr(img_idx)+total_natoms)=forces
            END IF
        END IF

        ! Call trainer when computations of all images are done
        IF (img_idx == nimages) THEN
            !Timings
            CALL cpu_time(start)
            CALL Trainer(opt_type,max_epoch,MAXVAL(nGs),MAXVAL(natoms_arr),MAXVAL(nhidneurons),force_coeff)
            CALL cpu_time(finish)
            print *, 'Training Time: ', finish-start, "seconds"
        END IF

    END SUBROUTINE

    SUBROUTINE Trainer(opt_type,maxepochs,max_nGs,max_natoms_arr, max_hidneurons,force_coeff,&
    energyRMSEtol,forceRMSEtol,learningRate)
        IMPLICIT NONE
        !Inputs
        CHARACTER(*) :: opt_type
        INTEGER, INTENT(IN) :: maxepochs, max_nGs, max_natoms_arr, max_hidneurons
        DOUBLE PRECISION, OPTIONAL :: energyRMSEtol, forceRMSEtol
        DOUBLE PRECISION, OPTIONAL :: force_coeff, learningRate

        !Must be input variables eventually
        DOUBLE PRECISION :: beta1, beta2, eps, weight_decay
        !variables
        INTEGER :: max_num_neigh
        INTEGER :: epoch, time, i
        INTEGER, DIMENSION(total_natoms) :: num_neigh, symbols
        INTEGER, DIMENSION(total_natoms, total_natoms) :: neighs
        DOUBLE PRECISION :: fconst, etol, ftol, lr
        DOUBLE PRECISION :: energyloss, forceloss, loss
        DOUBLE PRECISION :: energyRMSE, forceRMSE

        IF (PRESENT(force_coeff)) THEN
            fconst=force_coeff
        ELSE
            fconst=0.1
        END IF
        IF (PRESENT(energyRMSEtol)) THEN
            etol=energyRMSEtol
        ELSE
            etol=0.0001
        END IF
        IF (PRESENT(forceRMSEtol)) THEN
            ftol=forceRMSEtol
        ELSE
            ftol=0.01
        END IF
        IF (PRESENT(learningRate)) THEN
            lr=learningRate
        ELSE
            lr=0.01
        END IF

        DO epoch=1, maxepochs
            curr_epoch=epoch
            print *, '*********************************'
            print *, 'epoch=', epoch
            print *, '*********************************'
            ! Compute energy and forces with updated parameters
            IF (epoch /= 1) THEN
                DO i=1, nimages
                  epoch_img_idx=i
                  num_neigh=nneighbors_img(:,i)
                  max_num_neigh=MAXVAL(num_neigh)
                  neighs=neighborlists_img(:,1:max_num_neigh,i)
                  symbols=symbols_img(:,i)
                  !print *, 'img_idx=',i
                  CALL forward(num_neigh, max_num_neigh, neighs, symbols, input_fps(:,:,:,i),&
                  input_dfps(1:total_natoms,1:max_num_neigh+1,1:3,1:max_nGs,i),&
                  max_nGs, max_natoms_arr, max_hidneurons)

                  !Update calculated energy of image i
                  IF (energy_training) THEN
                    inputE(i)=Etotal-intercept
                  END IF
                  !Update calculated force of image i
                  IF (force_training) THEN
                    IF (i==1) THEN
                      inputF(1:3,1:total_natoms)=forces(1:3,1:total_natoms)
                    ELSE
                      inputF(1:3,nAtimg_ptr(i)+1:nAtimg_ptr(i)+total_natoms)=forces(1:3,1:total_natoms)
                    END IF
                  END IF
                END DO
            END IF

            forceloss=0.
            ! Compute loss and backward propagation
            CALL LossFunction(fconst,energyloss,forceloss,loss)
            CALL backward(fconst)

            ! Step of optimizer
            IF (epoch == 1) THEN
              CALL opt_init(opt_type)
              OPEN(unit=10, file='pyamff.log', status='unknown')
              WRITE(10,'(a)') 'epoch    lossValue   EnergyRMSE    ForceRMSE'
            END IF
            !TODO: how to set parameters of each optimizers
            CALL opt_step(opt_type,epoch)

            energyRMSE=sqrt(energyloss/nimages)
            forceRMSE=sqrt(forceloss/nimages)
            WRITE(10,'(I6,F16.8,F16.8,F16.8)') epoch, loss, energyRMSE, forceRMSE
            IF (energyRMSE < etol .and. forceRMSE < ftol) THEN
                WRITE(10,'(a)') 'Minimization converged'
                PRINT *, 'Minimization converged'
                GOTO 30
                !STOP
            END IF
        END DO
        PRINT *, 'Minimization NOT converged. Check your config values' 

        ! Clean up arrays used for optimizer
  30    CALL opt_cleanup(opt_type)

    END SUBROUTINE 

!    SUBROUTINE write_mlff
!        !This is currently temporary format. Only prints out Model parameters
!        IMPLICIT NONE
!
!        OPEN(unit=11, file='trained.pyamff', status='unknown')
!        WRITE(11,'(a)') '#Model Parameters'
!        WRITE(11,'(A2)') 
!
!    END SUBROUTINE

    SUBROUTINE traincleanup
        USE fpCalc
        IMPLICIT NONE

        CALL cleanup
        CALL nncleanup
        CALL backcleanup
    END SUBROUTINE

END MODULE

