;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robotHandPosition)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robotHandPosition (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOTHANDPOSITION")
  (make-package "DYNAMIC_BIPED::ROBOTHANDPOSITION"))

(in-package "ROS")
;;//! \htmlinclude robotHandPosition.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass dynamic_biped::robotHandPosition
  :super ros::object
  :slots (_header _left_hand_position _right_hand_position ))

(defmethod dynamic_biped::robotHandPosition
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:left_hand_position __left_hand_position) (make-array 0 :initial-element 0 :element-type :char))
    ((:right_hand_position __right_hand_position) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _header __header)
   (setq _left_hand_position __left_hand_position)
   (setq _right_hand_position __right_hand_position)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:left_hand_position
   (&optional __left_hand_position)
   (if __left_hand_position (setq _left_hand_position __left_hand_position)) _left_hand_position)
  (:right_hand_position
   (&optional __right_hand_position)
   (if __right_hand_position (setq _right_hand_position __right_hand_position)) _right_hand_position)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8[] _left_hand_position
    (* 1    (length _left_hand_position)) 4
    ;; uint8[] _right_hand_position
    (* 1    (length _right_hand_position)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8[] _left_hand_position
     (write-long (length _left_hand_position) s)
     (princ _left_hand_position s)
     ;; uint8[] _right_hand_position
     (write-long (length _right_hand_position) s)
     (princ _right_hand_position s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8[] _left_hand_position
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _left_hand_position (make-array n :element-type :char))
     (replace _left_hand_position buf :start2 ptr-) (incf ptr- n))
   ;; uint8[] _right_hand_position
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _right_hand_position (make-array n :element-type :char))
     (replace _right_hand_position buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(setf (get dynamic_biped::robotHandPosition :md5sum-) "ee45ac8dfb6e0c9a4c7712025fd69d4f")
(setf (get dynamic_biped::robotHandPosition :datatype-) "dynamic_biped/robotHandPosition")
(setf (get dynamic_biped::robotHandPosition :definition-)
      "Header header
uint8[] left_hand_position
uint8[] right_hand_position

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :dynamic_biped/robotHandPosition "ee45ac8dfb6e0c9a4c7712025fd69d4f")


