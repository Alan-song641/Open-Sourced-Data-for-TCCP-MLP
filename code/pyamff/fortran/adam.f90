MODULE adam
  USE nnType
  USE trainType
  IMPLICIT NONE
  DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: bias_v, bias_m
  DOUBLE PRECISION, DIMENSION(:,:,:,:), ALLOCATABLE :: weight_v, weight_m

  CONTAINS

  SUBROUTINE adam_init
    IMPLICIT NONE
    ! Allocate first moment and second moment arrays
    ALLOCATE(bias_m(MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(bias_v(MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(weight_m(max(MAXVAL(nGs),MAXVAL(nhidneurons)),MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(weight_v(max(MAXVAL(nGs),MAXVAL(nhidneurons)),MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ! Zeros arrays
    bias_v=0.
    bias_m=0.
    weight_v=0.
    weight_m=0.
  END SUBROUTINE

  SUBROUTINE adam_step(time,b1,b2,lr1,wd,eps1)
    IMPLICIT NONE
    !Input
    INTEGER, INTENT(IN) :: time
    DOUBLE PRECISION, OPTIONAL :: b1, b2, lr1, wd, eps1
    !Variables
    INTEGER :: i, k, l
    DOUBLE PRECISION :: beta1,beta2, lr, weight_decay, eps 
    DOUBLE PRECISION :: bias_corr1, bias_corr2

    !Set parameters if not specified
    !Default value comes from pytorch
    IF (PRESENT(b1)) THEN
      beta1=b1
    ELSE
      beta1=0.9d0
    END IF
    IF (PRESENT(b2)) THEN
      beta2=b2
    ELSE
      beta2=0.999d0
    END IF
    IF (PRESENT(lr1)) THEN
      lr=lr1
    ELSE
      lr=0.01d0
    END IF
    IF (PRESENT(wd)) THEN
      weight_decay=wd
    ELSE
      weight_decay=0.
    END IF
    IF (PRESENT(eps1)) THEN
      eps=eps1
    ELSE
      eps=1.0e-8
    END IF

    ! Update decay factors w.r.t time
    bias_corr1=1.d0-beta1**time
    bias_corr2=1.d0-beta2**time
    lr=lr/bias_corr1

    !TODO: need if statement for the case where weight_decya is not equal 0.0

    ! Update parameters
    DO i=1, nelements
      ! Update first/second moments of input weights and biases
      weight_m(1:nGs(i),1:nhidneurons(1),1,i)=&
      beta1*weight_m(1:nGs(i),1:nhidneurons(1),1,i)+(1.d0-beta1)*weight_grad(1:nGs(i),1:nhidneurons(1),1,i)
      weight_v(1:nGs(i),1:nhidneurons(1),1,i)=&
      beta2*weight_v(1:nGs(i),1:nhidneurons(1),1,i)+(1.d0-beta2)*((weight_grad(1:nGs(i),1:nhidneurons(1),1,i))**2)

      bias_m(1:nhidneurons(1),1,i)=&
      beta1*bias_m(1:nhidneurons(1),1,i)+(1-beta1)*bias_grad(1:nhidneurons(1),1,i)
      bias_v(1:nhidneurons(1),1,i)=&
      beta2*bias_v(1:nhidneurons(1),1,i)+(1-beta2)*((bias_grad(1:nhidneurons(1),1,i))**2)

      ! Update input weights and biases
      in_weights(1:nGs(i),1:nhidneurons(1),i)=&
      in_weights(1:nGs(i),1:nhidneurons(1),i)&
      -lr*weight_m(1:nGs(i),1:nhidneurons(1),1,i)/&
      (sqrt(weight_v(1:nGs(i),1:nhidneurons(1),1,i)/bias_corr2)+eps)
      !print *, 'updated input weights'
      !print *, in_weights(1:nGs(i),1:nhidneurons(1),i)
        !   DO k=1, natoms_arr(i)
        !     in_biases(k,1:nhidneurons(1),i)=&
        !     in_biases(k,1:nhidneurons(1),i)&
        !     -lr*bias_m(1:nhidneurons(1),1,i)/(sqrt(bias_v(1:nhidneurons(1),1,i)/bias_corr2)+eps)
        !   END DO
      in_biases(1:nhidneurons(1),i)= in_biases(1:nhidneurons(1),i)&
        -lr*bias_m(1:nhidneurons(1),1,i)/(sqrt(bias_v(1:nhidneurons(1),1,i)/bias_corr2)+eps)
      !print *, 'updated input biases'
      !print *, in_biases(1,1:nhidneurons(1),i)
      DO l=1, nhidlayers-1
        ! Update first/second moments of input weights and biases 
        weight_m(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)=&
        beta1*weight_m(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)&
        +(1-beta1)*weight_grad(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)
        weight_v(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)=&
        beta2*weight_v(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)&
        +(1-beta2)*(weight_grad(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)**2)

        bias_m(1:nhidneurons(l+1),l+1,i)=&
        beta1*bias_m(1:nhidneurons(l+1),l+1,i)+(1-beta1)*bias_grad(1:nhidneurons(l+1),l+1,i)
        bias_v(1:nhidneurons(l+1),l+1,i)=&
        beta2*bias_v(1:nhidneurons(l+1),l+1,i)+(1-beta2)*(bias_grad(1:nhidneurons(l+1),l+1,i)**2)

        ! Update lth hidden layer's weights and biases
        hid_weights(1:nhidneurons(l),1:nhidneurons(l+1),l,i)=&
        hid_weights(1:nhidneurons(l),1:nhidneurons(l+1),l,i)&
        -lr*weight_m(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)/&
        (sqrt(weight_v(1:nhidneurons(l),1:nhidneurons(l+1),l+1,i)/bias_corr2)+eps)
        !print *, l,'th hiddenlayer weights'
        !print *, hid_weights(1:nhidneurons(l),1:nhidneurons(l+1),l,i)
        ! DO k=1, natoms_arr(i)
        !   hid_biases(k,1:nhidneurons(l+1),l,i)=&
        !   hid_biases(k,1:nhidneurons(l+1),l,i)&
        !   -lr*bias_m(1:nhidneurons(l+1),l+1,i)/(sqrt(bias_v(1:nhidneurons(l+1),l+1,i)/bias_corr2)+eps)
        ! END DO
        hid_biases(1:nhidneurons(l+1),l,i)= hid_biases(1:nhidneurons(l+1),l,i)&
          -lr*bias_m(1:nhidneurons(l+1),l+1,i)/(sqrt(bias_v(1:nhidneurons(l+1),l+1,i)/bias_corr2)+eps)
        !print *, l,'th hiddenlayer biases'
        !print *, hid_biases(1,1:nhidneurons(l+1),l,i)
      END DO

      ! Update first moments of input weights and biases  
      weight_m(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)=&
      beta1*weight_m(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)&
      +(1-beta1)*weight_grad(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)
      weight_v(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)=&
      beta2*weight_v(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)&
      +(1-beta2)*(weight_grad(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)**2)

      bias_m(1,nhidlayers+1,i)=&
      beta1*bias_m(1,nhidlayers+1,i)+(1-beta1)*bias_grad(1,nhidlayers+1,i)
      bias_v(1,nhidlayers+1,i)=&
      beta2*bias_v(1,nhidlayers+1,i)+(1-beta2)*(bias_grad(1,nhidlayers+1,i)**2)

      ! Update output layer's weights and biases
      out_weights(1:nhidneurons(nhidlayers),1,i)=&
      out_weights(1:nhidneurons(nhidlayers),1,i)&
      -lr*weight_m(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)/&
      (sqrt(weight_v(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)/bias_corr2)+eps)
      !print *, 'output layer weight'
      !print *, out_weights(1:nhidneurons(nhidlayers),1,i)
      !   DO k=1, natoms_arr(i)
      !     out_biases(k,1,i)=out_biases(k,1,i)&
      !     -lr*bias_m(1,nhidlayers+1,i)/(sqrt(bias_v(1,nhidlayers+1,i)/bias_corr2)+eps)
      !   END DO
      out_biases(1,i)=out_biases(1,i)&
        -lr*bias_m(1,nhidlayers+1,i)/(sqrt(bias_v(1,nhidlayers+1,i)/bias_corr2)+eps)
      !print *, 'output layer biases'
      !print *, out_biases(1,1,i)
    END DO  

  END SUBROUTINE

  SUBROUTINE adam_cleanup
    IMPLICIT NONE

    DEALLOCATE(weight_v)
    DEALLOCATE(weight_m)
    DEALLOCATE(bias_v)
    DEALLOCATE(bias_m)

  END SUBROUTINE

END MODULE
