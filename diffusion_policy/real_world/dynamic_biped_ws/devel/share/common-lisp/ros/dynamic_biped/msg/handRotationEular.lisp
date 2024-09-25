; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude handRotationEular.msg.html

(cl:defclass <handRotationEular> (roslisp-msg-protocol:ros-message)
  ((eulerAngles
    :reader eulerAngles
    :initarg :eulerAngles
    :type (cl:vector geometry_msgs-msg:Vector3)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Vector3 :initial-element (cl:make-instance 'geometry_msgs-msg:Vector3))))
)

(cl:defclass handRotationEular (<handRotationEular>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <handRotationEular>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'handRotationEular)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<handRotationEular> is deprecated: use dynamic_biped-msg:handRotationEular instead.")))

(cl:ensure-generic-function 'eulerAngles-val :lambda-list '(m))
(cl:defmethod eulerAngles-val ((m <handRotationEular>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:eulerAngles-val is deprecated.  Use dynamic_biped-msg:eulerAngles instead.")
  (eulerAngles m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <handRotationEular>) ostream)
  "Serializes a message object of type '<handRotationEular>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'eulerAngles))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'eulerAngles))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <handRotationEular>) istream)
  "Deserializes a message object of type '<handRotationEular>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'eulerAngles) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'eulerAngles)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Vector3))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<handRotationEular>)))
  "Returns string type for a message object of type '<handRotationEular>"
  "dynamic_biped/handRotationEular")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'handRotationEular)))
  "Returns string type for a message object of type 'handRotationEular"
  "dynamic_biped/handRotationEular")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<handRotationEular>)))
  "Returns md5sum for a message object of type '<handRotationEular>"
  "6f24fd3e9eed11c525f6da164f46e8b2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'handRotationEular)))
  "Returns md5sum for a message object of type 'handRotationEular"
  "6f24fd3e9eed11c525f6da164f46e8b2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<handRotationEular>)))
  "Returns full string definition for message of type '<handRotationEular>"
  (cl:format cl:nil "geometry_msgs/Vector3[] eulerAngles~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'handRotationEular)))
  "Returns full string definition for message of type 'handRotationEular"
  (cl:format cl:nil "geometry_msgs/Vector3[] eulerAngles~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <handRotationEular>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'eulerAngles) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <handRotationEular>))
  "Converts a ROS message object to a list"
  (cl:list 'handRotationEular
    (cl:cons ':eulerAngles (eulerAngles msg))
))
