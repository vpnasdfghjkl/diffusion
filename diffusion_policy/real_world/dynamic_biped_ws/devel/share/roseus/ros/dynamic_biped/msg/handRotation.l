;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::handRotation)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'handRotation (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::HANDROTATION")
  (make-package "DYNAMIC_BIPED::HANDROTATION"))

(in-package "ROS")
;;//! \htmlinclude handRotation.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass dynamic_biped::handRotation
  :super ros::object
  :slots (_quaternions ))

(defmethod dynamic_biped::handRotation
  (:init
   (&key
    ((:quaternions __quaternions) ())
    )
   (send-super :init)
   (setq _quaternions __quaternions)
   self)
  (:quaternions
   (&rest __quaternions)
   (if (keywordp (car __quaternions))
       (send* _quaternions __quaternions)
     (progn
       (if __quaternions (setq _quaternions (car __quaternions)))
       _quaternions)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Quaternion[] _quaternions
    (apply #'+ (send-all _quaternions :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Quaternion[] _quaternions
     (write-long (length _quaternions) s)
     (dolist (elem _quaternions)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Quaternion[] _quaternions
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _quaternions (let (r) (dotimes (i n) (push (instance geometry_msgs::Quaternion :init) r)) r))
     (dolist (elem- _quaternions)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get dynamic_biped::handRotation :md5sum-) "c666021c5a7330bd53d9827c2e91d9f6")
(setf (get dynamic_biped::handRotation :datatype-) "dynamic_biped/handRotation")
(setf (get dynamic_biped::handRotation :definition-)
      "geometry_msgs/Quaternion[] quaternions

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

")



(provide :dynamic_biped/handRotation "c666021c5a7330bd53d9827c2e91d9f6")


