;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robotHeadMotionData)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robotHeadMotionData (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOTHEADMOTIONDATA")
  (make-package "DYNAMIC_BIPED::ROBOTHEADMOTIONDATA"))

(in-package "ROS")
;;//! \htmlinclude robotHeadMotionData.msg.html


(defclass dynamic_biped::robotHeadMotionData
  :super ros::object
  :slots (_joint_data ))

(defmethod dynamic_biped::robotHeadMotionData
  (:init
   (&key
    ((:joint_data __joint_data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _joint_data __joint_data)
   self)
  (:joint_data
   (&optional __joint_data)
   (if __joint_data (setq _joint_data __joint_data)) _joint_data)
  (:serialization-length
   ()
   (+
    ;; float64[] _joint_data
    (* 8    (length _joint_data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _joint_data
     (write-long (length _joint_data) s)
     (dotimes (i (length _joint_data))
       (sys::poke (elt _joint_data i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _joint_data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _joint_data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _joint_data i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get dynamic_biped::robotHeadMotionData :md5sum-) "400001e7cf73111efbced59084cb481a")
(setf (get dynamic_biped::robotHeadMotionData :datatype-) "dynamic_biped/robotHeadMotionData")
(setf (get dynamic_biped::robotHeadMotionData :definition-)
      "float64[] joint_data
")



(provide :dynamic_biped/robotHeadMotionData "400001e7cf73111efbced59084cb481a")


