(define (stream continuous-tamp)
  (:stream s-region
    :inputs (?b ?r)
    :domain (Placeable ?b ?r)
    :outputs (?p)
    :certified (and (Contain ?b ?p ?r) (Pose ?b ?p))
  )
  
  (:stream s-grasp
    :inputs (?b)
    :domain (Block ?b)
    :outputs (?g)
    :certified (Grasp ?b ?g)
  )
  
  (:stream s-ik ;;Inverse Kinematics (IK)
    :inputs (?b ?p ?g)
    :domain (and (Pose ?b ?p) (Grasp ?b ?g))
    :outputs (?q)
    :certified (and (Kin ?b ?q ?p ?g) (Conf ?q))
  )
  
  (:stream s-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose AtGrasp)
    :outputs (?t)
    :certified (and (Motion ?q1 ?t ?q2) (Traj ?t))
  )
  
  (:stream t-cfree
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CFree ?b1 ?p1 ?b2 ?p2)
  )

  (:stream s-stack
    :inputs (?b_above ?b_below ?p_below)
    :domain (and 
      (Block ?b_above) 
      (Block ?b_below)
      (Pose ?b_below ?p_below)
    )
    :outputs (?p_above)
    :certified (and 
      (Pose ?b_above ?p_above)
      (StackPose ?p_above ?p_below)
    )
  )

  (:stream t-on-ground
    :inputs (?b ?p)
    :domain (Pose ?b ?p)
    :certified (PoseOnGround ?p)
  )
  
  (:stream t-on-platform
    :inputs (?b ?p)
    :domain (Pose ?b ?p)
    :certified (PoseOnPlatform ?p)
  )
  
  (:function (Dist ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))
  )
)