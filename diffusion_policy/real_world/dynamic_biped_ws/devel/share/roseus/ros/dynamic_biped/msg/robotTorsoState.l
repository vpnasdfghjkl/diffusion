;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robotTorsoState)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robotTorsoState (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOTTORSOSTATE")
  (make-package "DYNAMIC_BIPED::ROBOTTORSOSTATE"))

(in-package "ROS")
;;//! \htmlinclude robotTorsoState.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass dynamic_biped::robotTorsoState
  :super ros::object
  :slots (_torsoR _torsoRd _torsoRdd _r _rd _rdd ))

(defmethod dynamic_biped::robotTorsoState
  (:init
   (&key
    ((:torsoR __torsoR) (instance geometry_msgs::Vector3 :init))
    ((:torsoRd __torsoRd) (instance geometry_msgs::Vector3 :init))
    ((:torsoRdd __torsoRdd) (instance geometry_msgs::Vector3 :init))
    ((:r __r) (instance geometry_msgs::Vector3 :init))
    ((:rd __rd) (instance geometry_msgs::Vector3 :init))
    ((:rdd __rdd) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _torsoR __torsoR)
   (setq _torsoRd __torsoRd)
   (setq _torsoRdd __torsoRdd)
   (setq _r __r)
   (setq _rd __rd)
   (setq _rdd __rdd)
   self)
  (:torsoR
   (&rest __torsoR)
   (if (keywordp (car __torsoR))
       (send* _torsoR __torsoR)
     (progn
       (if __torsoR (setq _torsoR (car __torsoR)))
       _torsoR)))
  (:torsoRd
   (&rest __torsoRd)
   (if (keywordp (car __torsoRd))
       (send* _torsoRd __torsoRd)
     (progn
       (if __torsoRd (setq _torsoRd (car __torsoRd)))
       _torsoRd)))
  (:torsoRdd
   (&rest __torsoRdd)
   (if (keywordp (car __torsoRdd))
       (send* _torsoRdd __torsoRdd)
     (progn
       (if __torsoRdd (setq _torsoRdd (car __torsoRdd)))
       _torsoRdd)))
  (:r
   (&rest __r)
   (if (keywordp (car __r))
       (send* _r __r)
     (progn
       (if __r (setq _r (car __r)))
       _r)))
  (:rd
   (&rest __rd)
   (if (keywordp (car __rd))
       (send* _rd __rd)
     (progn
       (if __rd (setq _rd (car __rd)))
       _rd)))
  (:rdd
   (&rest __rdd)
   (if (keywordp (car __rdd))
       (send* _rdd __rdd)
     (progn
       (if __rdd (setq _rdd (car __rdd)))
       _rdd)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Vector3 _torsoR
    (send _torsoR :serialization-length)
    ;; geometry_msgs/Vector3 _torsoRd
    (send _torsoRd :serialization-length)
    ;; geometry_msgs/Vector3 _torsoRdd
    (send _torsoRdd :serialization-length)
    ;; geometry_msgs/Vector3 _r
    (send _r :serialization-length)
    ;; geometry_msgs/Vector3 _rd
    (send _rd :serialization-length)
    ;; geometry_msgs/Vector3 _rdd
    (send _rdd :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Vector3 _torsoR
       (send _torsoR :serialize s)
     ;; geometry_msgs/Vector3 _torsoRd
       (send _torsoRd :serialize s)
     ;; geometry_msgs/Vector3 _torsoRdd
       (send _torsoRdd :serialize s)
     ;; geometry_msgs/Vector3 _r
       (send _r :serialize s)
     ;; geometry_msgs/Vector3 _rd
       (send _rd :serialize s)
     ;; geometry_msgs/Vector3 _rdd
       (send _rdd :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Vector3 _torsoR
     (send _torsoR :deserialize buf ptr-) (incf ptr- (send _torsoR :serialization-length))
   ;; geometry_msgs/Vector3 _torsoRd
     (send _torsoRd :deserialize buf ptr-) (incf ptr- (send _torsoRd :serialization-length))
   ;; geometry_msgs/Vector3 _torsoRdd
     (send _torsoRdd :deserialize buf ptr-) (incf ptr- (send _torsoRdd :serialization-length))
   ;; geometry_msgs/Vector3 _r
     (send _r :deserialize buf ptr-) (incf ptr- (send _r :serialization-length))
   ;; geometry_msgs/Vector3 _rd
     (send _rd :deserialize buf ptr-) (incf ptr- (send _rd :serialization-length))
   ;; geometry_msgs/Vector3 _rdd
     (send _rdd :deserialize buf ptr-) (incf ptr- (send _rdd :serialization-length))
   ;;
   self)
  )

(setf (get dynamic_biped::robotTorsoState :md5sum-) "0cabdc260bffa0da5323abf719a81f3e")
(setf (get dynamic_biped::robotTorsoState :datatype-) "dynamic_biped/robotTorsoState")
(setf (get dynamic_biped::robotTorsoState :definition-)
      "geometry_msgs/Vector3 torsoR
geometry_msgs/Vector3 torsoRd
geometry_msgs/Vector3 torsoRdd
geometry_msgs/Vector3 r
geometry_msgs/Vector3 rd
geometry_msgs/Vector3 rdd
================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :dynamic_biped/robotTorsoState "0cabdc260bffa0da5323abf719a81f3e")


