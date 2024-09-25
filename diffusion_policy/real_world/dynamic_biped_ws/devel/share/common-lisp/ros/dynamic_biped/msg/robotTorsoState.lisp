; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude robotTorsoState.msg.html

(cl:defclass <robotTorsoState> (roslisp-msg-protocol:ros-message)
  ((torsoR
    :reader torsoR
    :initarg :torsoR
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (torsoRd
    :reader torsoRd
    :initarg :torsoRd
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (torsoRdd
    :reader torsoRdd
    :initarg :torsoRdd
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (r
    :reader r
    :initarg :r
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (rd
    :reader rd
    :initarg :rd
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (rdd
    :reader rdd
    :initarg :rdd
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass robotTorsoState (<robotTorsoState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <robotTorsoState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'robotTorsoState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<robotTorsoState> is deprecated: use dynamic_biped-msg:robotTorsoState instead.")))

(cl:ensure-generic-function 'torsoR-val :lambda-list '(m))
(cl:defmethod torsoR-val ((m <robotTorsoState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:torsoR-val is deprecated.  Use dynamic_biped-msg:torsoR instead.")
  (torsoR m))

(cl:ensure-generic-function 'torsoRd-val :lambda-list '(m))
(cl:defmethod torsoRd-val ((m <robotTorsoState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:torsoRd-val is deprecated.  Use dynamic_biped-msg:torsoRd instead.")
  (torsoRd m))

(cl:ensure-generic-function 'torsoRdd-val :lambda-list '(m))
(cl:defmethod torsoRdd-val ((m <robotTorsoState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:torsoRdd-val is deprecated.  Use dynamic_biped-msg:torsoRdd instead.")
  (torsoRdd m))

(cl:ensure-generic-function 'r-val :lambda-list '(m))
(cl:defmethod r-val ((m <robotTorsoState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:r-val is deprecated.  Use dynamic_biped-msg:r instead.")
  (r m))

(cl:ensure-generic-function 'rd-val :lambda-list '(m))
(cl:defmethod rd-val ((m <robotTorsoState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:rd-val is deprecated.  Use dynamic_biped-msg:rd instead.")
  (rd m))

(cl:ensure-generic-function 'rdd-val :lambda-list '(m))
(cl:defmethod rdd-val ((m <robotTorsoState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:rdd-val is deprecated.  Use dynamic_biped-msg:rdd instead.")
  (rdd m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <robotTorsoState>) ostream)
  "Serializes a message object of type '<robotTorsoState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'torsoR) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'torsoRd) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'torsoRdd) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'r) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rd) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rdd) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <robotTorsoState>) istream)
  "Deserializes a message object of type '<robotTorsoState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'torsoR) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'torsoRd) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'torsoRdd) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'r) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rd) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rdd) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<robotTorsoState>)))
  "Returns string type for a message object of type '<robotTorsoState>"
  "dynamic_biped/robotTorsoState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'robotTorsoState)))
  "Returns string type for a message object of type 'robotTorsoState"
  "dynamic_biped/robotTorsoState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<robotTorsoState>)))
  "Returns md5sum for a message object of type '<robotTorsoState>"
  "0cabdc260bffa0da5323abf719a81f3e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'robotTorsoState)))
  "Returns md5sum for a message object of type 'robotTorsoState"
  "0cabdc260bffa0da5323abf719a81f3e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<robotTorsoState>)))
  "Returns full string definition for message of type '<robotTorsoState>"
  (cl:format cl:nil "geometry_msgs/Vector3 torsoR~%geometry_msgs/Vector3 torsoRd~%geometry_msgs/Vector3 torsoRdd~%geometry_msgs/Vector3 r~%geometry_msgs/Vector3 rd~%geometry_msgs/Vector3 rdd~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'robotTorsoState)))
  "Returns full string definition for message of type 'robotTorsoState"
  (cl:format cl:nil "geometry_msgs/Vector3 torsoR~%geometry_msgs/Vector3 torsoRd~%geometry_msgs/Vector3 torsoRdd~%geometry_msgs/Vector3 r~%geometry_msgs/Vector3 rd~%geometry_msgs/Vector3 rdd~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <robotTorsoState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'torsoR))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'torsoRd))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'torsoRdd))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'r))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rd))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rdd))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <robotTorsoState>))
  "Converts a ROS message object to a list"
  (cl:list 'robotTorsoState
    (cl:cons ':torsoR (torsoR msg))
    (cl:cons ':torsoRd (torsoRd msg))
    (cl:cons ':torsoRdd (torsoRdd msg))
    (cl:cons ':r (r msg))
    (cl:cons ':rd (rd msg))
    (cl:cons ':rdd (rdd msg))
))
