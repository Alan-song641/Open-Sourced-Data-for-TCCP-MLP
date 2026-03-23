MODULE lossgrad
  USE nnType
  USE trainType
  IMPLICIT NONE
  
  CONTAINS

  SUBROUTINE init_backward
    IMPLICIT NONE
    !Allocate arrays
    ALLOCATE(natomsE(nimages))
    ALLOCATE(inputE(nimages))
    ALLOCATE(targetE(nimages))
    ALLOCATE(natomsF(nAtimg))
    ALLOCATE(nAtimg_ptr(nimages))
    ALLOCATE(inputF(3,nAtimg))
    ALLOCATE(targetF(3,nAtimg))
    ! TODO:Followings should be generalized w.r.t multiple images
    ! TODO:Should consider a case where each image has different number of atoms, etc
    ! TODO:Allocate arrays with if energy_training or force_training
    ALLOCATE(layer_backgrad(MAXVAL(natoms_arr),MAXVAL(nhidneurons),MAXVAL(nhidneurons),nhidlayers,nelements,nimages))
    ALLOCATE(bias_grad(MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(weight_grad(max(MAXVAL(nGs),MAXVAL(nhidneurons)),MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(bias_grad_e(MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(weight_grad_e(max(MAXVAL(nGs),MAXVAL(nhidneurons)),MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(bias_grad_f(MAXVAL(nhidneurons),nhidlayers+1,nelements))
    ALLOCATE(weight_grad_f(max(MAXVAL(nGs),MAXVAL(nhidneurons)),MAXVAL(nhidneurons),nhidlayers+1,nelements)) 
    
    ALLOCATE(all_neurons(MAXVAL(natoms_arr),MAXVAL(nhidneurons),nhidlayers+1,nelements,nimages))
    ALLOCATE(input_fps(MAXVAL(natoms_arr),MAXVAL(nGs),nelements,nimages))
    ALLOCATE(input_dfps(total_natoms,total_natoms,3,MAXVAL(nGs),nimages))
    ALLOCATE(in_gradients_img(MAXVAL(natoms_arr),MAXVAL(nGs), nhidneurons(1),nelements,nimages))
    ALLOCATE(hid_gradients_img(MAXVAL(natoms_arr),MAXVAL(nhidneurons),MAXVAL(nhidneurons),nhidlayers-1,nelements,nimages))
    ALLOCATE(out_gradients_img(MAXVAL(natoms_arr),nhidneurons(nhidlayers),1,nelements,nimages))
    ALLOCATE(symbols_img(total_natoms,nimages))
    ALLOCATE(nneighbors_img(total_natoms,nimages))
    ALLOCATE(neighborlists_img(total_natoms,total_natoms,nimages))

    ! Initialize arrays with zero
    curr_epoch=0
    layer_backgrad=0
    bias_grad=0
    weight_grad=0
    bias_grad_e=0
    weight_grad_e=0
    bias_grad_f=0
    weight_grad_f=0
    all_neurons=0
    input_fps=0
    input_dfps=0
    in_gradients_img=0
    hid_gradients_img=0
    out_gradients_img=0
    natomsE=0
    natomsF=0
    inputE=0
    targetE=0
    inputF=0.
    targetF=0.

  END SUBROUTINE

  SUBROUTINE backward(fconst)
    !inputs
    DOUBLE PRECISION :: fconst
    !variables
    INTEGER :: i, k, l, img, j, n

    ! Zero arrays for every epoch
    layer_backgrad=0
    bias_grad=0
    weight_grad=0
    bias_grad_e=0
    weight_grad_e=0
    bias_grad_f=0
    weight_grad_f=0

    DO i=1, nelements
      DO img=1, nimages
        !print *, 'img=',img
        IF (nhidlayers==1) THEN
          ! No hidweights
          GOTO 10
        ELSE
          layer_backgrad(1:natoms_arr(i),1:nhidneurons(1),1:nhidneurons(2),1,i,img) = &
          layer_backgrad(1:natoms_arr(i),1:nhidneurons(1),1:nhidneurons(2),1,i,img)+ &
          backwardgrad(all_neurons(1:natoms_arr(i),1:nhidneurons(1),1,i,img),&
          hid_weights(1:nhidneurons(1),1:nhidneurons(2),1,i),natoms_arr(i),nhidneurons(2),nhidneurons(1))
          IF (nhidlayers==2) THEN
            ! Only one hidweight
            GOTO 10
          ELSE
            DO l=1, nhidlayers-2
              layer_backgrad(1:natoms_arr(i),1:nhidneurons(l+1),1:nhidneurons(l+2),l+1,i,img)=&
              layer_backgrad(1:natoms_arr(i),1:nhidneurons(l+1),1:nhidneurons(l+2),l+1,i,img)+&
              backwardgrad(all_neurons(1:natoms_arr(i),1:nhidneurons(l+1),l+1,i,img),&
              hid_weights(1:nhidneurons(l+1),1:nhidneurons(l+2),l+1,i),natoms_arr(i),nhidneurons(l+2),nhidneurons(l+1))
            END DO
          END IF
        END IF
 10     CONTINUE
        layer_backgrad(1:natoms_arr(i),1:nhidneurons(nhidlayers),:1,nhidlayers,i,img)=&
        layer_backgrad(1:natoms_arr(i),1:nhidneurons(nhidlayers),:1,nhidlayers,i,img)+&
        backwardgrad(all_neurons(1:natoms_arr(i),1:nhidneurons(nhidlayers),nhidlayers,i,img),&
        out_weights(1:nhidneurons(nhidlayers),:1,i),natoms_arr(i),1,nhidneurons(nhidlayers))
      END DO
    END DO

    DO img=1,nimages
      IF (energy_training) THEN
        CALL eloss_grad(img)
      END IF
      IF (force_training) THEN
        CALL floss_grad(img,fconst)
      END IF
    END DO
    !Sum NN parameter grads from energy loss and force loss
    bias_grad=bias_grad_e+fconst*bias_grad_f
    weight_grad=weight_grad_e+fconst*weight_grad_f

!    DO i=1, nelements
!      DO l=1, nhidlayers+1
!        if (l==nhidlayers+1) then
!          print *, 'output layer weight grads of element', i
!          print *, weight_grad(1:nhidneurons(l-1),1,l,i)
!          print *, 'output layer bias grads of element', i
!          print *, bias_grad(1,l,i)
!        else if (l==1) then
!          print *, l,'th layer weight grads of element', i
!          do j=1,nhidneurons(l)
!            print *, weight_grad(1:nGs(i),j,l,i)
!          end do
!          print *, l,'th layer bias grads of element', i
!          print *, bias_grad(1:nhidneurons(l),l,i)
!        else
!          print *, l,'th layer weight grads of element', i
!          print *, weight_grad(1:nhidneurons(l-1),1:nhidneurons(l),l,i)
!          print *, l,'th layer bias grads of element', i
!          print *, bias_grad(1:nhidneurons(l),l,i)
!        end if
!      END DO
!    END DO

  END SUBROUTINE

  SUBROUTINE eloss_grad(img)
    !-----------------------------------------------------------------------!
    !Compute energy loss gradients w.r.t bias and weights of 'img'th image  !
    !bias_grad_e and weight_grad_e are summed over all images one by one    !
    !Note that this subroutine is within do loop over the number of images  !
    !-----------------------------------------------------------------------!
    !TODO: Currently, all of images has to have same number of atoms, elements, etc. 
    IMPLICIT NONE
    !Inputs
    INTEGER, INTENT(IN) :: img
    !variables
    INTEGER :: i, k, l, j, n
    DOUBLE PRECISION :: dloss_dE
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),MAXVAL(nhidneurons),nhidlayers+1,nelements) :: bias_grad_e_helper 
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),max(MAXVAL(nGs),MAXVAL(nhidneurons)),MAXVAL(nhidneurons),&
    nhidlayers+1,nelements) :: weight_grad_e_helper

    !Zeros arrays 
    bias_grad_e_helper=0.
    weight_grad_e_helper=0.

    !dLoss/dE for each atom 
    dloss_dE=((2*(inputE(img)-targetE(img))/natomsE(img))*slope)/total_natoms

    !Calculate bias grad for each atom seperately
    DO i=1, nelements
      DO k=1, natoms_arr(i)
        !output layer
        bias_grad_e_helper(k,1,nhidlayers+1,i)=dloss_dE
        !Hidden layers to input layer
        DO l=nhidlayers,1,-1
          IF (l==nhidlayers) THEN
            bias_grad_e_helper(k,1:nhidneurons(l),l,i)=&
            layer_backgrad(k,1:nhidneurons(l),1,l,i,img)*bias_grad_e_helper(k,1,nhidlayers+1,i)
          ELSE
            bias_grad_e_helper(k,1:nhidneurons(l),l,i)=&
            MATMUL(layer_backgrad(k,1:nhidneurons(l),1:nhidneurons(l+1),l,i,img),&
            bias_grad_e_helper(k,1:nhidneurons(l+1),l+1,i))
          END IF
        END DO  
      END DO
    END DO

    !Calculate bias_grad for each element 
    DO i=1, nelements
      ! Bias grad of output layer
      bias_grad_e(1,nhidlayers+1,i)=bias_grad_e(1,nhidlayers+1,i)+dloss_dE*natoms_arr(i)
      ! Bias grad of hidden layers and input layer
      DO l=nhidlayers,1,-1
        DO k=1, natoms_arr(i)
          bias_grad_e(1:nhidneurons(l),l,i)=& 
          bias_grad_e(1:nhidneurons(l),l,i)+bias_grad_e_helper(k,1:nhidneurons(l),l,i)
        END DO
      END DO
    END DO

    !Calculate weight grad for each atom 
    DO i=1, nelements 
      DO k=1, natoms_arr(i)
        !output layer
        weight_grad_e_helper(k,1:nhidneurons(nhidlayers),1,nhidlayers+1,i)=&
        all_neurons(k,1:nhidneurons(nhidlayers),nhidlayers,i,img)*bias_grad_e_helper(k,1,nhidlayers+1,i)

        !hidden layers'
        DO l=nhidlayers,1,-1
          IF (l==1) THEN
            weight_grad_e_helper(k,1:nGs(i),1:nhidneurons(l),l,i)=&
            MATMUL(reshape(input_fps(k,1:nGs(i),i,img),(/nGs(i),1/)),&
            reshape(bias_grad_e_helper(k,1:nhidneurons(l),l,i),(/1,nhidneurons(l)/)))
          ELSE
            weight_grad_e_helper(k,1:nhidneurons(l-1),1:nhidneurons(l),l,i)=&
            MATMUL(reshape(all_neurons(k,1:nhidneurons(l-1),l-1,i,img),(/nhidneurons(l-1),1/)),&
            reshape(bias_grad_e_helper(k,1:nhidneurons(l),l,i),(/1,nhidneurons(l)/)))
          END IF
        END DO
      END DO
    END DO

    !Calculate weight grads for each element 
    DO i=1, nelements
      !output layer's weight grads
      DO k=1, natoms_arr(i)
        !Derivative Eloss w.r.t output layer's weights
        weight_grad_e(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)=&
        weight_grad_e(1:nhidneurons(nhidlayers),1,nhidlayers+1,i)+&
        weight_grad_e_helper(k,1:nhidneurons(nhidlayers),1,nhidlayers+1,i)
      END DO

      DO l=nhidlayers,1,-1
        DO k=1, natoms_arr(i)
          IF (l == 1) THEN
            weight_grad_e(1:nGs(i),1:nhidneurons(l),l,i)=&
            weight_grad_e(1:nGs(i),1:nhidneurons(l),l,i)+&
            weight_grad_e_helper(k,1:nGs(i),1:nhidneurons(l),l,i)
          ELSE
            weight_grad_e(1:nhidneurons(l-1),1:nhidneurons(l),l,i)=&
            weight_grad_e(1:nhidneurons(l-1),1:nhidneurons(l),l,i)+&  
            weight_grad_e_helper(k,1:nhidneurons(l-1),1:nhidneurons(l),l,i) 
          END IF
        END DO
      END DO
    END DO  

  END SUBROUTINE

  SUBROUTINE floss_grad(img,fconst)
    !-----------------------------------------------------------------------!
    !Compute force loss gradients w.r.t bias and weights of 'img'th image   !
    !bias_grad_f and weight_grad_f are summed over all images one by one    !
    !Note that this subroutine is within do loop over the number of images  !
    !-----------------------------------------------------------------------!
    IMPLICIT NONE
    !Inputs
    INTEGER, INTENT(IN) :: img
    DOUBLE PRECISION :: fconst
    !variables
    INTEGER :: i, k, l, j, n, m, p, ll, ptr, nid, k2, m2, i2, myid
    !Difference between computed force and reference force
    DOUBLE PRECISION, DIMENSION(3,total_natoms) :: diffF
    !1st, 2nd derivative of activation values in diagonal
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),MAXVAL(nhidneurons),MAXVAL(nhidneurons),&
    nhidlayers,nelements) :: actval_dp, actval_p
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),MAXVAL(nGs),MAXVAL(nhidneurons),nhidlayers,nelements) ::&
    front_chain_b, front_chain_w
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),MAXVAL(nhidneurons),nhidlayers,nelements) :: &
    back_chain_b, back_chain_w 
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),MAXVAL(nhidneurons),MAXVAL(nhidneurons),&
    nhidlayers,nhidlayers,nelements) :: node_chains 
    DOUBLE PRECISION, DIMENSION(MAXVAL(natoms_arr),MAXVAL(nhidneurons),MAXVAL(nhidneurons),MAXVAL(nhidneurons),&
    nhidlayers,nhidlayers,nelements) :: node_grad_b
    DOUBLE PRECISION, DIMENSION(total_natoms,MAXVAL(nGs),MAXVAL(nhidneurons),nhidlayers,nelements) :: dbias_dEdGs
    DOUBLE PRECISION, DIMENSION(total_natoms,MAXVAL(nGs),max(MAXVAL(nGs),MAXVAL(nhidneurons)),&
    MAXVAL(nhidneurons),nhidlayers+1,nelements) :: dweight_dEdGs

    !Zeros arrays for each image
    actval_dp=0
    actval_p=0
    front_chain_b=0
    front_chain_w=0
    back_chain_b=0
    back_chain_w=0
    node_chains=0
    node_grad_b=0
    dbias_dEdGs=0
    dweight_dEdGs=0
    diffF=0

    !calculate common part of force loss gradient first
    floss_const=-slope*(2./3.)/total_natoms

    !calculate difference between computed force and reference's of 'img'th image
    IF (img == 1) THEN
      diffF(1:3,1:total_natoms)=inputF(1:3,1:total_natoms)-targetF(1:3,1:total_natoms)
    ELSE
      diffF(1:3,1:total_natoms)=inputF(1:3,nAtimg_ptr(img)+1:nAtimg_ptr(img)+total_natoms)&
      -targetF(1:3,nAtimg_ptr(img)+1:nAtimg_ptr(img)+total_natoms)
    END IF

    !Compute 1st, 2nd derivative of activation values for all images
    DO i=1, nelements
      DO l=1, nhidlayers
        actval_p(1:natoms_arr(i),1:nhidneurons(l),1:nhidneurons(l),l,i)=&
        actfunc_p(all_neurons(1:natoms_arr(i),1:nhidneurons(l),l,i,img),natoms_arr(i),nhidneurons(l))
        actval_dp(1:natoms_arr(i),1:nhidneurons(l),1:nhidneurons(l),l,i)=&
        actfunc_dp(all_neurons(1:natoms_arr(i),1:nhidneurons(l),l,i,img),natoms_arr(i),nhidneurons(l))
      END DO
    END DO

    !Compute front_chain and back_chains for bias and weights of all layers
    DO i=1,nelements
      DO k=1, natoms_arr(i)
        !Input layer
        front_chain_b(k,1:nGs(i),1:nhidneurons(1),1,i)=in_weights(1:nGs(i),1:nhidneurons(1),i)
        !front_chain_w is just 1 so skipped
        IF (nhidlayers == 1) THEN !If only 1 hidden layer
          back_chain_b(k,1:nhidneurons(1),1,i)=out_weights(1:nhidneurons(1),1,i)
        ELSE
          back_chain_b(k,1:nhidneurons(1),1,i)=&
          back_chain(hid_gradients_img(k,:,:,:,i,img),out_gradients_img(k,:,1,i,img),1,nhidlayers-1)
        END IF
        back_chain_w(k,1:nhidneurons(1),:1,i)=&
        MATMUL(actval_p(k,1:nhidneurons(1),1:nhidneurons(1),1,i),back_chain_b(k,1:nhidneurons(1),:1,i))
        !hidden layers
        DO l=2, nhidlayers
          front_chain_b(k,1:nGs(i),1:nhidneurons(l),l,i)=&
          front_chain(in_gradients_img(k,1:nGs(i),1:nhidneurons(1),i,img),&
          hid_gradients_img(k,:,:,:,i,img),hid_weights(1:nhidneurons(l-1),1:nhidneurons(l),l-1,i),i,l)
          front_chain_w(k,1:nGs(i),1:nhidneurons(l-1),l-1,i)=&
          MATMUL(front_chain_b(k,1:nGs(i),1:nhidneurons(l-1),l-1,i),&
          actval_p(k,1:nhidneurons(l-1),1:nhidneurons(l-1),l-1,i))
          IF (l == nhidlayers) THEN 
            back_chain_b(k,1:nhidneurons(nhidlayers),nhidlayers,i)=&
            out_weights(1:nhidneurons(nhidlayers),1,i)
          ELSE
            back_chain_b(k,1:nhidneurons(l),l,i)=&
            back_chain(hid_gradients_img(k,:,:,:,i,img),out_gradients_img(k,:,:1,i,img),l,nhidlayers-l)
          END IF
          back_chain_w(k,1:nhidneurons(l),l,i)=&
          MATMUL(actval_p(k,1:nhidneurons(l),1:nhidneurons(l),l,i),back_chain_b(k,1:nhidneurons(l),l,i))
        END DO

        !Output layer
        !bias grad is 0 so just weight grad only
        front_chain_w(k,1:nGs(i),1:nhidneurons(nhidlayers),nhidlayers,i)=&
        MATMUL(front_chain_b(k,1:nGs(i),1:nhidneurons(nhidlayers),nhidlayers,i),&
        actval_p(k,1:nhidneurons(nhidlayers),1:nhidneurons(nhidlayers),nhidlayers,i))
        !back_chain_w is just 1 so skipped
      END DO
    END DO

    !Compute the chain for node gradients about later nodes w.r.t using layer_backgrads
    !This only requires when nhidlayers >= 2
    IF (nhidlayers == 1) THEN 
      !do nothing
      CONTINUE 
    ELSE
      DO i=1, nelements
        DO k=1, natoms_arr(i)
          DO l=1, nhidlayers-1
            !initiate pointer for node_chains array
            ptr=1
            DO ll=l+1, nhidlayers
              !Compute node chain first (layer_backgrad chains)
              node_chains(k,1:nhidneurons(l),1:nhidneurons(ll),ptr,l,i)=&
              node_chain(layer_backgrad(k,:,:,:,i,img),l,ll,ll-l)
              !Compute node gradient w.r.t bias
              DO n=1, nhidneurons(l)
                DO m=1, nhidneurons(ll)
                  node_grad_b(k,n,m,m,ptr,l,i)=actval_dp(k,m,m,ll,i)*node_chains(k,n,m,ptr,l,i)
                END DO
              END DO
              !Update pointer
              ptr=ptr+1
            END DO  
          END DO  
        END DO
      END DO
    END IF

    !Compute dbias_dEdGs first
    DO i=1, nelements
      DO l=nhidlayers,1,-1
        DO k=1, natoms_arr(i)
          DO n=1, nhidneurons(l)
            !w.r.t the layer itself
            dbias_dEdGs(atom_idx(k,i),1:nGs(i),n,l,i)=dbias_dEdGs(atom_idx(k,i),1:nGs(i),n,l,i)+&
            front_chain_b(k,1:nGs(i),n,l,i)*actval_dp(k,n,n,l,i)*back_chain_b(k,n,l,i)
            !w.r.t. the later layers
            ptr=1
            DO ll=l+1,nhidlayers
              dbias_dEdGs(atom_idx(k,i),1:nGs(i),n,l,i)=dbias_dEdGs(atom_idx(k,i),1:nGs(i),n,l,i)+& 
              MATMUL(front_chain_b(k,1:nGs(i),1:nhidneurons(ll),ll,i),&
              MATMUL(node_grad_b(k,n,1:nhidneurons(ll),1:nhidneurons(ll),ptr,l,i),&
              back_chain_b(k,1:nhidneurons(ll),ll,i)))
              ptr=ptr+1
            END DO
          END DO
        END DO
      END DO
    END DO

    !Compute dweight_dEdGs now
    DO i=1, nelements
      DO k=1, natoms_arr(i)
        !Output layer's
        DO n=1, nhidneurons(nhidlayers)
          dweight_dEdGs(atom_idx(k,i),1:nGs(i),n,1,nhidlayers+1,i)=&
          front_chain_w(k,1:nGs(i),n,nhidlayers,i)
        END DO
      END DO

      !Hidden layers
      DO l=nhidlayers,2,-1 
        DO k=1, natoms_arr(i)
          DO n=1, nhidneurons(l)
            DO m=1, nhidneurons(l-1)
              !w.r.t. layer's weight itself
              dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,l,i)=&
              front_chain_w(k,1:nGs(i),m,l-1,i)*back_chain_w(k,n,l,i)

              !w.r.t layer's neuron itself
              dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,l,i)=&
              dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,l,i)+&
              front_chain_b(k,1:nGs(i),n,l,i)&
              *actval_dp(k,n,n,l,i)*all_neurons(k,m,l-1,i,img)*back_chain_b(k,n,l,i)

              IF (l==nhidlayers) THEN 
                !Do nothing
                CONTINUE
              ELSE
                !w.r.t. later neurons
                !Initiate pointer
                ptr=1
                DO ll=l+1,nhidlayers
                  dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,l,i)=&
                  dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,l,i)+&
                  MATMUL(front_chain_b(k,1:nGs(i),1:nhidneurons(ll),ll,i),&
                  MATMUL(node_grad_b(k,n,1:nhidneurons(ll),1:nhidneurons(ll),ptr,l,i),&
                  back_chain_b(k,1:nhidneurons(ll),ll,i)))*all_neurons(k,m,l-1,i,img)
                  ptr=ptr+1
                END DO
              END IF
            END DO
          END DO
        END DO
      END DO

      !Input layer's
      DO k=1, natoms_arr(i)
        DO n=1,nhidneurons(1)
          DO m=1, nGs(i)
            !w.r.t layer's weight itself
            dweight_dEdGs(atom_idx(k,i),m,m,n,1,i)=&
            dweight_dEdGs(atom_idx(k,i),m,m,n,1,i)+back_chain_w(k,n,1,i)
            !w.r.t layer's neuron itself
            dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,1,i)=&
            dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,1,i)+&
            front_chain_b(k,1:nGs(i),n,1,i)&
            *actval_dp(k,n,n,1,i)*input_fps(k,m,i,img)*back_chain_b(k,n,1,i)
            !w.r.t. later neurons
            ptr=1
            DO ll=2,nhidlayers
              dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,1,i)=&
              dweight_dEdGs(atom_idx(k,i),1:nGs(i),m,n,1,i)+&
              MATMUL(front_chain_b(k,1:nGs(i),1:nhidneurons(ll),ll,i),&
              MATMUL(node_grad_b(k,n,1:nhidneurons(ll),1:nhidneurons(ll),ptr,1,i),&
              back_chain_b(k,1:nhidneurons(ll),ll,i)))*input_fps(k,m,i,img)
              ptr=ptr+1
            END DO
          END DO
        END DO
      END DO
    END DO

    !compute bias gradients for each element
    DO i=1, nelements
      !Output layer's  is 0
      !Hidden layers and input layer
      DO l=nhidlayers,1,-1
        DO n=1, nhidneurons(l)
          DO i2=1, total_natoms
            myid=symbols_img(i2,img)
            !dfps from itself
            bias_grad_f(n:n,l,i)=bias_grad_f(n:n,l,i)+&
            floss_const*MATMUL(diffF(1:3,i2),&
            MATMUL(input_dfps(i2,1,1:3,1:nGs(myid),img),dbias_dEdGs(i2,1:nGs(myid),n:n,l,i)))
            !dfps from neighbors
            DO j=1, nneighbors_img(i2,img)
              p=neighborlists_img(i2,j,img)
              nid=symbols_img(p,img)
              bias_grad_f(n:n,l,i)=bias_grad_f(n:n,l,i)+&
              floss_const*MATMUL(diffF(1:3,i2),&
              MATMUL(input_dfps(i2,j+1,1:3,1:nGs(nid),img),&
              dbias_dEdGs(p,1:nGs(nid),n:n,l,i)))
            END DO
          END DO
        END DO
      END DO
    END DO

    !Now compute weight gradients
    DO i=1, nelements
      !Output layer's
      DO n=1, nhidneurons(nhidlayers)
        DO i2=1, total_natoms
          myid=symbols_img(i2,img)
          !dfps from itself
          weight_grad_f(n:n,1,nhidlayers+1,i)=weight_grad_f(n:n,1,nhidlayers+1,i)+&
          floss_const*MATMUL(diffF(1:3,i2),&
          MATMUL(input_dfps(i2,1,1:3,1:nGs(myid),img),&
          dweight_dEdGs(i2,1:nGs(myid),n:n,1,nhidlayers+1,i)))

          !dfps from neighbors
          DO j=1, nneighbors_img(i2,img) 
            p=neighborlists_img(i2,j,img)
            nid=symbols_img(p,img)
            weight_grad_f(n:n,1,nhidlayers+1,i)=&
            weight_grad_f(n:n,1,nhidlayers+1,i)+&
            floss_const*MATMUL(diffF(1:3,i2),&
            MATMUL(input_dfps(i2,j+1,1:3,1:nGs(nid),img),&
            dweight_dEdGs(p,1:nGs(nid),n:n,1,nhidlayers+1,i))) 
          END DO
        END DO
      END DO

      !Hidden layers
      DO l=nhidlayers,2,-1 
        DO n=1, nhidneurons(l)
          DO m=1, nhidneurons(l-1)
            DO i2=1, total_natoms  
              myid=symbols_img(i2,img)
              !w.r.t. layer's weight itself + dfps from itself
              weight_grad_f(m:m,n,l,i)=weight_grad_f(m:m,n,l,i)+&
              floss_const*MATMUL(diffF(1:3,i2),&
              MATMUL(input_dfps(i2,1,1:3,1:nGs(myid),img),& 
              dweight_dEdGs(i2,1:nGs(myid),m:m,n,l,i)))
              !dfps from neighbors
              DO j=1, nneighbors_img(i2,img)
                p=neighborlists_img(i2,j,img)
                nid=symbols_img(p,img)
                weight_grad_f(m:m,n,l,i)=weight_grad_f(m:m,n,l,i)+&
                floss_const*MATMUL(diffF(1:3,i2),&
                MATMUL(input_dfps(i2,j+1,1:3,1:nGs(nid),img),&
                dweight_dEdGs(p,1:nGs(nid),m:m,n,l,i)))
              END DO
            END DO
          END DO
        END DO
      END DO

      !Input layer's
      DO n=1,nhidneurons(1)
        DO i2=1, total_natoms
          myid=symbols_img(i2,img)
          !w.r.t layer's weight itself + dfps from itself
          weight_grad_f(1:nGs(i),n,1,i)=weight_grad_f(1:nGs(i),n,1,i)+&
          floss_const*MATMUL(diffF(1:3,i2),&
          MATMUL(input_dfps(i2,1,1:3,1:nGs(myid),img),&
          dweight_dEdGs(i2,1:nGs(myid),1:nGs(i),n,1,i)))
          !dfps from neighbors
          DO j=1, nneighbors_img(i2,img)
            p=neighborlists_img(i2,j,img)
            nid=symbols_img(p,img)
            weight_grad_f(1:nGs(i),n,1,i)=weight_grad_f(1:nGs(i),n,1,i)+&
            floss_const*MATMUL(diffF(1:3,i2),&
            MATMUL(input_dfps(i2,j+1,1:3,1:nGs(nid),img),&
            dweight_dEdGs(p,1:nGs(nid),1:nGs(i),n,1,i)))
          END DO
        END DO
      END DO
    END DO !cycle over nelements

  END SUBROUTINE

  FUNCTION backwardgrad(actval, weight, i, j, k) RESULT (backgrad)
    IMPLICIT NONE
    DOUBLE PRECISION, DIMENSION(i, k) :: actval
    DOUBLE PRECISION, DIMENSION(k, k) :: diagonal
    DOUBLE PRECISION, DIMENSION(k, j) :: weight
    DOUBLE PRECISION, DIMENSION(i,k,j) :: backgrad
    INTEGER :: a, b, i, j, k

    IF (actfuncId == 'sigmoid') THEN
      DO a = 1, i
        diagonal = 0.0d0
        DO b = 1, k
          diagonal(b,b) = actval(a,b)*(1.0d0 - actval(a,b))
        END DO
        backgrad(a,1:k,1:j) = MATMUL(diagonal,weight)
      END DO
    ELSE IF (actfuncId == 'tanh') THEN
      DO a = 1, i
        diagonal = 0.0d0
        DO b = 1, k
          diagonal(b,b) = 1.0d0-actval(a,b)**2
        END DO
        backgrad(a,1:k,1:j) = MATMUL(diagonal,weight)
      END DO
    ELSE IF (actfuncId == 'relu') THEN
      DO a = 1, i
        diagonal = 0.0d0
        DO b = 1, k
          IF (actval(a,b) > 0) THEN
            diagonal(b,b) = 1.0d0
          ELSE
            diagonal(b,b) = 0.0d0
          END IF
        END DO
        backgrad(a,1:k,1:j) = MATMUL(diagonal,weight)
      END DO
    END IF

  END FUNCTION

  SUBROUTINE LossFunction(force_coeff,energyloss, forceloss,loss)
      !Inputs
      DOUBLE PRECISION, OPTIONAL :: force_coeff
      !Outputs
      DOUBLE PRECISION, INTENT(OUT) :: energyloss, forceloss, loss

      IF (PRESENT(force_coeff)) THEN
        CONTINUE
      ELSE
        force_coeff=0.05
      END IF
      !print *, 'inputE=', inputE
      !print *, 'targetE=', targetE
      !print *, 'inputF=', inputF
      !print *, 'targetF=', targetF
      IF (energy_training) THEN
        energyloss=SUM(((inputE-targetE)/natomsE)**(2.0)) 
      ELSE
        energyloss=0.
      END IF
      IF (force_training) THEN
        forceloss=SUM(SUM((inputF-targetF)**(2.0),dim=1)/natomsF)/3.0
      ELSE
        forceloss=0.
      END IF
      loss=energyloss+forceloss*force_coeff 

      !print *, 'force_coeff=', force_coeff
      print *, 'energyloss=', energyloss
      print *, 'forceloss=', forceloss
      print *, 'loss=', loss

  END SUBROUTINE

  FUNCTION actfunc_p(actval,len1,len2) RESULT(outarray)
    IMPLICIT NONE
    INTEGER, intent(in) :: len1,len2
    DOUBLE PRECISION, intent(in) :: actval(len1,len2)
    DOUBLE PRECISION :: outarray(len1,len2,len2)
    INTEGER :: a,b

    outarray = 0.0
    DO a=1, len1
      outarray(a,:,:)=0.
      IF (actfuncId .EQ. 'relu') THEN
        DO b=1,len2
          IF (actval(a,b) > 0.) THEN
            outarray(a,1:len2,1:len2) = 1.0
          ELSE
            outarray(a,1:len2,1:len2) = 0.0
          END IF
        END DO
      ELSE IF (actfuncId .EQ. 'sigmoid') THEN
        DO b=1,len2
           outarray(a,b,b) = actval(a,b)*(1.0d0-actval(a,b)) 
        END DO
      ELSE IF (actfuncId .EQ. 'tanh') THEN
        DO b=1, len2
          outarray(a,b,b) = 1.0d0-actval(a,b)**2 
        END DO
      ELSE
        PRINT *, 'Unknown function: ', actfuncId, 'for force training'
        STOP
      END IF
    END DO
  END FUNCTION actfunc_p

  FUNCTION actfunc_dp(actval,len1,len2) RESULT (outarray)
    IMPLICIT NONE
    INTEGER, intent(in) :: len1,len2
    DOUBLE PRECISION, intent(in) :: actval(len1,len2)
    DOUBLE PRECISION :: outarray(len1,len2,len2)
    INTEGER :: a,b

    outarray = 0.0
    DO a=1, len1
      outarray(a,:,:)=0.
      IF (actfuncId .EQ. 'relu') THEN
        CONTINUE
        !outarray(a,1:len2,1:len2) = 0.0
      ELSE IF (actfuncId .EQ. 'sigmoid') THEN
        DO b=1,len2
           outarray(a,b,b) = actval(a,b)*(1.0d0-actval(a,b))*(1.0d0-2.d0*actval(a,b))
        END DO
      ELSE IF (actfuncId .EQ. 'tanh') THEN
        DO b=1, len2
          outarray(a,b,b) = -2.d0*actval(a,b)*(1.0d0-actval(a,b)**2)
        END DO  
      ELSE
        PRINT *, 'Unknown function: ', actfuncId, 'for force training'
        STOP
      END IF
    END DO
  END FUNCTION actfunc_dp

  FUNCTION node_chain(layer_backgrad_in,layer_idx,layer_idx2,layers) RESULT(outarr)
    IMPLICIT NONE
    INTEGER :: layer_idx, layer_idx2, layers
    DOUBLE PRECISION, DIMENSION(MAXVAL(nhidneurons),MAXVAL(nhidneurons),nhidlayers) :: layer_backgrad_in
    DOUBLE PRECISION, DIMENSION(nhidneurons(layer_idx),nhidneurons(layer_idx2)) :: outarr
    DOUBLE PRECISION, DIMENSION(MAXVAL(nhidneurons),MAXVAL(nhidneurons)) :: chain
    !Variables
    INTEGER :: l

    !Initiate node chain
    chain(1:nhidneurons(layer_idx),1:nhidneurons(layer_idx+1))=&
    layer_backgrad_in(1:nhidneurons(layer_idx),1:nhidneurons(layer_idx+1),layer_idx)
    IF (layers == 1) THEN
      outarr=chain(1:nhidneurons(layer_idx),1:nhidneurons(layer_idx2))
    ELSE
      DO l=1, layers-1
        chain(1:nhidneurons(layer_idx),1:nhidneurons(layer_idx+l+1))=&
        MATMUL(chain(1:nhidneurons(layer_idx),1:nhidneurons(layer_idx+l)),&
        layer_backgrad_in(1:nhidneurons(layer_idx+l),1:nhidneurons(layer_idx+l+1),layer_idx+l))
      END DO
      outarr=chain(1:nhidneurons(layer_idx),1:nhidneurons(layer_idx2))
    END IF

  END FUNCTION node_chain 

  FUNCTION front_chain(in_gradient,hid_gradient_in,weight,idx,curr_layer) RESULT(outarr) 
    IMPLICIT NONE
    INTEGER :: idx, curr_layer
    DOUBLE PRECISION, DIMENSION(nGs(idx),nhidneurons(1)) :: in_gradient
    DOUBLE PRECISION, DIMENSION(MAXVAL(nhidneurons),MAXVAL(nhidneurons),nhidlayers-1) :: hid_gradient_in
    DOUBLE PRECISION, DIMENSION(nhidneurons(curr_layer-1),nhidneurons(curr_layer)) :: weight
    DOUBLE PRECISION, DIMENSION(nGs(idx),nhidneurons(curr_layer)) :: outarr
    DOUBLE PRECISION, DIMENSION(MAXVAL(nhidneurons),MAXVAL(nhidneurons)) :: hid_gradient
    !Variables
    INTEGER :: l

    hid_gradient(1:nhidneurons(1),1:nhidneurons(2))=&
    hid_gradient_in(1:nhidneurons(1),1:nhidneurons(2),1)
    IF (curr_layer-2 .EQ. 0) THEN
      outarr=MATMUL(in_gradient,weight)
    ELSE
      DO l=1, curr_layer-3
        hid_gradient(1:nhidneurons(1),1:nhidneurons(l+2))=&
        MATMUL(hid_gradient(1:nhidneurons(1),1:nhidneurons(l+1)),&
        hid_gradient_in(1:nhidneurons(l+1),1:nhidneurons(l+2),l+1))
      END DO
      outarr=MATMUL(in_gradient,&
      MATMUL(hid_gradient(1:nhidneurons(1),1:nhidneurons(curr_layer-1)),weight))
    END IF

  END FUNCTION front_chain

  FUNCTION back_chain(hid_gradient_in,out_gradient,curr_layer,rest_layers) RESULT(outarr)
    IMPLICIT NONE
    INTEGER :: curr_layer, rest_layers
    DOUBLE PRECISION, DIMENSION(MAXVAL(nhidneurons),MAXVAL(nhidneurons),nhidlayers-1) :: hid_gradient_in
    DOUBLE PRECISION, DIMENSION(nhidneurons(nhidlayers)) :: out_gradient
    DOUBLE PRECISION, DIMENSION(nhidneurons(curr_layer)) :: outarr
    DOUBLE PRECISION, DIMENSION(MAXVAL(nhidneurons),MAXVAL(nhidneurons)) :: hid_gradient
    !Variables
    INTEGER :: l

    hid_gradient(1:nhidneurons(curr_layer),1:nhidneurons(curr_layer+1))=&
    hid_gradient_in(1:nhidneurons(curr_layer),1:nhidneurons(curr_layer+1),curr_layer)
    IF (rest_layers == 1) THEN 
      CONTINUE
    ELSE
      DO l=curr_layer, nhidlayers-2
        hid_gradient(1:nhidneurons(curr_layer),1:nhidneurons(l+2))=&
        MATMUL(hid_gradient(1:nhidneurons(curr_layer),1:nhidneurons(l+1)),& 
        hid_gradient_in(1:nhidneurons(l+1),1:nhidneurons(l+2),l+1))
      END DO
    END IF
    outarr=MATMUL(hid_gradient(1:nhidneurons(curr_layer),1:nhidneurons(nhidlayers)),&
    out_gradient(1:nhidneurons(nhidlayers)))

  END FUNCTION back_chain

  SUBROUTINE backcleanup
    IMPLICIT NONE

    DEALLOCATE(input_fps)
    DEALLOCATE(all_neurons)
    DEALLOCATE(layer_backgrad)
    DEALLOCATE(bias_grad)
    DEALLOCATE(weight_grad)
    DEALLOCATE(bias_grad_e)
    DEALLOCATE(weight_grad_e)
    DEALLOCATE(bias_grad_f)
    DEALLOCATE(weight_grad_f)
    DEALLOCATE(natomsE)
    DEALLOCATE(natomsF)
    DEALLOCATE(inputE)
    DEALLOCATE(targetE)
    DEALLOCATE(inputF)
    DEALLOCATE(targetF)

  END SUBROUTINE
END MODULE

