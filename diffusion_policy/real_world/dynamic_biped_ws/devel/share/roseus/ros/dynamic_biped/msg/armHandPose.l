;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::armHandPose)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'armHandPose (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ARMHANDPOSE")
  (make-package "DYNAMIC_BIPED::ARMHANDPOSE"))

(in-package "ROS")
;;//! \htmlinclude armHandPose.msg.html


(defclass dynamic_biped::armHandPose
  :super ros::object
  :slots (_pos_xyz _quat_xyzw _joint_angles ))

(defmethod dynamic_biped::armHandPose
  (:init
   (&key
    ((:pos_xyz __pos_xyz) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:quat_xyzw __quat_xyzw) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:joint_angles __joint_angles) (make-array 7 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _pos_xyz __pos_xyz)
   (setq _quat_xyzw __quat_xyzw)
   (setq _joint_angles __joint_angles)
   self)
  (:pos_xyz
   (&optional __pos_xyz)
   (if __pos_xyz (setq _pos_xyz __pos_xyz)) _pos_xyz)
  (:quat_xyzw
   (&optional __quat_xyzw)
   (if __quat_xyzw (setq _quat_xyzw __quat_xyzw)) _quat_xyzw)
  (:joint_angles
   (&optional __joint_angles)
   (if __joint_angles (setq _joint_angles __joint_angles)) _joint_angles)
  (:serialization-length
   ()
   (+
    ;; float64[3] _pos_xyz
    (* 8    3)
    ;; float64[4] _quat_xyzw
    (* 8    4)
    ;; float64[7] _joint_angles
    (* 8    7)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[3] _pos_xyz
     (dotimes (i 3)
       (sys::poke (elt _pos_xyz i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _quat_xyzw
     (dotimes (i 4)
       (sys::poke (elt _quat_xyzw i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[7] _joint_angles
     (dotimes (i 7)
       (sys::poke (elt _joint_angles i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[3] _pos_xyz
   (dotimes (i (length _pos_xyz))
     (setf (elt _pos_xyz i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _quat_xyzw
   (dotimes (i (length _quat_xyzw))
     (setf (elt _quat_xyzw i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[7] _joint_angles
   (dotimes (i (length _joint_angles))
     (setf (elt _joint_angles i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get dynamic_biped::armHandPose :md5sum-) "ae653a36c40e6220e166614d7037aae2")
(setf (get dynamic_biped::armHandPose :datatype-) "dynamic_biped/armHandPose")
(setf (get dynamic_biped::armHandPose :definition-)
      "float64[3] pos_xyz
float64[4] quat_xyzw

float64[7] joint_angles
")



(provide :dynamic_biped/armHandPose "ae653a36c40e6220e166614d7037aae2")


