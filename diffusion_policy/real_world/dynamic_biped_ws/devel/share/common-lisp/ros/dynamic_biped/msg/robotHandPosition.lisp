; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude robotHandPosition.msg.html

(cl:defclass <robotHandPosition> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (left_hand_position
    :reader left_hand_position
    :initarg :left_hand_position
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0))
   (right_hand_position
    :reader right_hand_position
    :initarg :right_hand_position
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass robotHandPosition (<robotHandPosition>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <robotHandPosition>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'robotHandPosition)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<robotHandPosition> is deprecated: use dynamic_biped-msg:robotHandPosition instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <robotHandPosition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:header-val is deprecated.  Use dynamic_biped-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'left_hand_position-val :lambda-list '(m))
(cl:defmethod left_hand_position-val ((m <robotHandPosition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:left_hand_position-val is deprecated.  Use dynamic_biped-msg:left_hand_position instead.")
  (left_hand_position m))

(cl:ensure-generic-function 'right_hand_position-val :lambda-list '(m))
(cl:defmethod right_hand_position-val ((m <robotHandPosition>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:right_hand_position-val is deprecated.  Use dynamic_biped-msg:right_hand_position instead.")
  (right_hand_position m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <robotHandPosition>) ostream)
  "Serializes a message object of type '<robotHandPosition>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'left_hand_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'left_hand_position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'right_hand_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'right_hand_position))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <robotHandPosition>) istream)
  "Deserializes a message object of type '<robotHandPosition>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'left_hand_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'left_hand_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'right_hand_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'right_hand_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<robotHandPosition>)))
  "Returns string type for a message object of type '<robotHandPosition>"
  "dynamic_biped/robotHandPosition")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'robotHandPosition)))
  "Returns string type for a message object of type 'robotHandPosition"
  "dynamic_biped/robotHandPosition")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<robotHandPosition>)))
  "Returns md5sum for a message object of type '<robotHandPosition>"
  "ee45ac8dfb6e0c9a4c7712025fd69d4f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'robotHandPosition)))
  "Returns md5sum for a message object of type 'robotHandPosition"
  "ee45ac8dfb6e0c9a4c7712025fd69d4f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<robotHandPosition>)))
  "Returns full string definition for message of type '<robotHandPosition>"
  (cl:format cl:nil "Header header~%uint8[] left_hand_position~%uint8[] right_hand_position~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'robotHandPosition)))
  "Returns full string definition for message of type 'robotHandPosition"
  (cl:format cl:nil "Header header~%uint8[] left_hand_position~%uint8[] right_hand_position~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <robotHandPosition>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'left_hand_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'right_hand_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <robotHandPosition>))
  "Converts a ROS message object to a list"
  (cl:list 'robotHandPosition
    (cl:cons ':header (header msg))
    (cl:cons ':left_hand_position (left_hand_position msg))
    (cl:cons ':right_hand_position (right_hand_position msg))
))
