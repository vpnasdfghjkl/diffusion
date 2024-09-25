;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robotArmInfo)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robotArmInfo (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOTARMINFO")
  (make-package "DYNAMIC_BIPED::ROBOTARMINFO"))

(in-package "ROS")
;;//! \htmlinclude robotArmInfo.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass dynamic_biped::robotArmInfo
  :super ros::object
  :slots (_header _q _v _vd _tau ))

(defmethod dynamic_biped::robotArmInfo
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:q __q) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:v __v) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:vd __vd) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:tau __tau) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _q __q)
   (setq _v __v)
   (setq _vd __vd)
   (setq _tau __tau)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:q
   (&optional __q)
   (if __q (setq _q __q)) _q)
  (:v
   (&optional __v)
   (if __v (setq _v __v)) _v)
  (:vd
   (&optional __vd)
   (if __vd (setq _vd __vd)) _vd)
  (:tau
   (&optional __tau)
   (if __tau (setq _tau __tau)) _tau)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64[] _q
    (* 8    (length _q)) 4
    ;; float64[] _v
    (* 8    (length _v)) 4
    ;; float64[] _vd
    (* 8    (length _vd)) 4
    ;; float64[] _tau
    (* 8    (length _tau)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64[] _q
     (write-long (length _q) s)
     (dotimes (i (length _q))
       (sys::poke (elt _q i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _v
     (write-long (length _v) s)
     (dotimes (i (length _v))
       (sys::poke (elt _v i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _vd
     (write-long (length _vd) s)
     (dotimes (i (length _vd))
       (sys::poke (elt _vd i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _tau
     (write-long (length _tau) s)
     (dotimes (i (length _tau))
       (sys::poke (elt _tau i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64[] _q
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _q (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _q i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _v
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _v (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _v i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _vd
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _vd (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _vd i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _tau
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _tau (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _tau i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get dynamic_biped::robotArmInfo :md5sum-) "3871141b674f003bc326e4d8da08f4ad")
(setf (get dynamic_biped::robotArmInfo :datatype-) "dynamic_biped/robotArmInfo")
(setf (get dynamic_biped::robotArmInfo :definition-)
      "Header header
float64[] q
float64[] v
float64[] vd
float64[] tau

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



(provide :dynamic_biped/robotArmInfo "3871141b674f003bc326e4d8da08f4ad")


