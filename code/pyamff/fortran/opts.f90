MODULE opts
  USE adam
  IMPLICIT NONE

  CONTAINS

  SUBROUTINE opt_init(opt_type)
    IMPLICIT NONE
    !Inputs
    CHARACTER(*) :: opt_type

    IF (opt_type == 'adam') THEN
      CALL adam_init
    ELSE
      print *, 'Fortran error: Not available optimizer type:', opt_type
      STOP
    END IF

  END SUBROUTINE

  SUBROUTINE opt_step(opt_type,epoch)
    !All default values follow pytorch
    IMPLICIT NONE
    !Inputs
    CHARACTER(*) :: opt_type
    INTEGER :: epoch
    !Variables
    !DOUBLE PRECISION :: beta1, beta2, learning_rate, weight_decay, eps, time

    IF (opt_type == 'adam') THEN
      !Call adam step
      !TODO: values for adam parameters has to come from inputs
      !At this point, we use pytorch default parameters 
      CALL adam_step(epoch)
    ELSE
      print *, 'Fortran error: Not available optimizer type:', opt_type
      STOP
    END IF

  END SUBROUTINE

  SUBROUTINE opt_cleanup(opt_type) 
    IMPLICIT NONE
    !Inputs
    CHARACTER(*) :: opt_type

    IF (opt_type == 'adam') THEN
      CALL adam_cleanup
    ELSE
      print *, 'Fortran error: Not available optimizer type:', opt_type
      STOP
    END IF

  END SUBROUTINE
END MODULE
