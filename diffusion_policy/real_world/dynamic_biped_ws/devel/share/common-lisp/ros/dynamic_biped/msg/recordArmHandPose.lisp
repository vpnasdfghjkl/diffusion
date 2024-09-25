; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude recordArmHandPose.msg.html

(cl:defclass <recordArmHandPose> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (left_pose
    :reader left_pose
    :initarg :left_pose
    :type dynamic_biped-msg:armHandPose
    :initform (cl:make-instance 'dynamic_biped-msg:armHandPose))
   (right_pose
    :reader right_pose
    :initarg :right_pose
    :type dynamic_biped-msg:armHandPose
    :initform (cl:make-instance 'dynamic_biped-msg:armHandPose)))
)

(cl:defclass recordArmHandPose (<recordArmHandPose>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <recordArmHandPose>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'recordArmHandPose)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<recordArmHandPose> is deprecated: use dynamic_biped-msg:recordArmHandPose instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <recordArmHandPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:header-val is deprecated.  Use dynamic_biped-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'left_pose-val :lambda-list '(m))
(cl:defmethod left_pose-val ((m <recordArmHandPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:left_pose-val is deprecated.  Use dynamic_biped-msg:left_pose instead.")
  (left_pose m))

(cl:ensure-generic-function 'right_pose-val :lambda-list '(m))
(cl:defmethod right_pose-val ((m <recordArmHandPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:right_pose-val is deprecated.  Use dynamic_biped-msg:right_pose instead.")
  (right_pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <recordArmHandPose>) ostream)
  "Serializes a message object of type '<recordArmHandPose>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'left_pose) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'right_pose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <recordArmHandPose>) istream)
  "Deserializes a message object of type '<recordArmHandPose>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'left_pose) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'right_pose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<recordArmHandPose>)))
  "Returns string type for a message object of type '<recordArmHandPose>"
  "dynamic_biped/recordArmHandPose")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'recordArmHandPose)))
  "Returns string type for a message object of type 'recordArmHandPose"
  "dynamic_biped/recordArmHandPose")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<recordArmHandPose>)))
  "Returns md5sum for a message object of type '<recordArmHandPose>"
  "ebe76034e32233813bca8e9a0bd81c2c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'recordArmHandPose)))
  "Returns md5sum for a message object of type 'recordArmHandPose"
  "ebe76034e32233813bca8e9a0bd81c2c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<recordArmHandPose>)))
  "Returns full string definition for message of type '<recordArmHandPose>"
  (cl:format cl:nil "Header header~%armHandPose  left_pose~%armHandPose  right_pose~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: dynamic_biped/armHandPose~%float64[3] pos_xyz~%float64[4] quat_xyzw~%~%float64[7] joint_angles~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'recordArmHandPose)))
  "Returns full string definition for message of type 'recordArmHandPose"
  (cl:format cl:nil "Header header~%armHandPose  left_pose~%armHandPose  right_pose~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: dynamic_biped/armHandPose~%float64[3] pos_xyz~%float64[4] quat_xyzw~%~%float64[7] joint_angles~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <recordArmHandPose>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'left_pose))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'right_pose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <recordArmHandPose>))
  "Converts a ROS message object to a list"
  (cl:list 'recordArmHandPose
    (cl:cons ':header (header msg))
    (cl:cons ':left_pose (left_pose msg))
    (cl:cons ':right_pose (right_pose msg))
))
