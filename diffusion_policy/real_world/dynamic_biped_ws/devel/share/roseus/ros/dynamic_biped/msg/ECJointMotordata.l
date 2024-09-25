;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::ECJointMotordata)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'ECJointMotordata (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ECJOINTMOTORDATA")
  (make-package "DYNAMIC_BIPED::ECJOINTMOTORDATA"))

(in-package "ROS")
;;//! \htmlinclude ECJointMotordata.msg.html


(defclass dynamic_biped::ECJointMotordata
  :super ros::object
  :slots (_l_pos_pitch _l_pos_yaw _l_pos_end _r_pos_pitch _r_pos_yaw _r_pos_end ))

(defmethod dynamic_biped::ECJointMotordata
  (:init
   (&key
    ((:l_pos_pitch __l_pos_pitch) 0)
    ((:l_pos_yaw __l_pos_yaw) 0)
    ((:l_pos_end __l_pos_end) 0)
    ((:r_pos_pitch __r_pos_pitch) 0)
    ((:r_pos_yaw __r_pos_yaw) 0)
    ((:r_pos_end __r_pos_end) 0)
    )
   (send-super :init)
   (setq _l_pos_pitch (round __l_pos_pitch))
   (setq _l_pos_yaw (round __l_pos_yaw))
   (setq _l_pos_end (round __l_pos_end))
   (setq _r_pos_pitch (round __r_pos_pitch))
   (setq _r_pos_yaw (round __r_pos_yaw))
   (setq _r_pos_end (round __r_pos_end))
   self)
  (:l_pos_pitch
   (&optional __l_pos_pitch)
   (if __l_pos_pitch (setq _l_pos_pitch __l_pos_pitch)) _l_pos_pitch)
  (:l_pos_yaw
   (&optional __l_pos_yaw)
   (if __l_pos_yaw (setq _l_pos_yaw __l_pos_yaw)) _l_pos_yaw)
  (:l_pos_end
   (&optional __l_pos_end)
   (if __l_pos_end (setq _l_pos_end __l_pos_end)) _l_pos_end)
  (:r_pos_pitch
   (&optional __r_pos_pitch)
   (if __r_pos_pitch (setq _r_pos_pitch __r_pos_pitch)) _r_pos_pitch)
  (:r_pos_yaw
   (&optional __r_pos_yaw)
   (if __r_pos_yaw (setq _r_pos_yaw __r_pos_yaw)) _r_pos_yaw)
  (:r_pos_end
   (&optional __r_pos_end)
   (if __r_pos_end (setq _r_pos_end __r_pos_end)) _r_pos_end)
  (:serialization-length
   ()
   (+
    ;; int32 _l_pos_pitch
    4
    ;; int32 _l_pos_yaw
    4
    ;; int32 _l_pos_end
    4
    ;; int32 _r_pos_pitch
    4
    ;; int32 _r_pos_yaw
    4
    ;; int32 _r_pos_end
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _l_pos_pitch
       (write-long _l_pos_pitch s)
     ;; int32 _l_pos_yaw
       (write-long _l_pos_yaw s)
     ;; int32 _l_pos_end
       (write-long _l_pos_end s)
     ;; int32 _r_pos_pitch
       (write-long _r_pos_pitch s)
     ;; int32 _r_pos_yaw
       (write-long _r_pos_yaw s)
     ;; int32 _r_pos_end
       (write-long _r_pos_end s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _l_pos_pitch
     (setq _l_pos_pitch (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _l_pos_yaw
     (setq _l_pos_yaw (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _l_pos_end
     (setq _l_pos_end (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _r_pos_pitch
     (setq _r_pos_pitch (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _r_pos_yaw
     (setq _r_pos_yaw (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _r_pos_end
     (setq _r_pos_end (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get dynamic_biped::ECJointMotordata :md5sum-) "da126c0fb22352a555bc24cbdebe8a85")
(setf (get dynamic_biped::ECJointMotordata :datatype-) "dynamic_biped/ECJointMotordata")
(setf (get dynamic_biped::ECJointMotordata :definition-)
      "int32 l_pos_pitch 
int32 l_pos_yaw
int32 l_pos_end
int32 r_pos_pitch 
int32 r_pos_yaw
int32 r_pos_end
")



(provide :dynamic_biped/ECJointMotordata "da126c0fb22352a555bc24cbdebe8a85")


