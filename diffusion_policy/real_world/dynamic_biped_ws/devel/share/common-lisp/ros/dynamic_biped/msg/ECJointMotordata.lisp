; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude ECJointMotordata.msg.html

(cl:defclass <ECJointMotordata> (roslisp-msg-protocol:ros-message)
  ((l_pos_pitch
    :reader l_pos_pitch
    :initarg :l_pos_pitch
    :type cl:integer
    :initform 0)
   (l_pos_yaw
    :reader l_pos_yaw
    :initarg :l_pos_yaw
    :type cl:integer
    :initform 0)
   (l_pos_end
    :reader l_pos_end
    :initarg :l_pos_end
    :type cl:integer
    :initform 0)
   (r_pos_pitch
    :reader r_pos_pitch
    :initarg :r_pos_pitch
    :type cl:integer
    :initform 0)
   (r_pos_yaw
    :reader r_pos_yaw
    :initarg :r_pos_yaw
    :type cl:integer
    :initform 0)
   (r_pos_end
    :reader r_pos_end
    :initarg :r_pos_end
    :type cl:integer
    :initform 0))
)

(cl:defclass ECJointMotordata (<ECJointMotordata>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ECJointMotordata>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ECJointMotordata)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<ECJointMotordata> is deprecated: use dynamic_biped-msg:ECJointMotordata instead.")))

(cl:ensure-generic-function 'l_pos_pitch-val :lambda-list '(m))
(cl:defmethod l_pos_pitch-val ((m <ECJointMotordata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:l_pos_pitch-val is deprecated.  Use dynamic_biped-msg:l_pos_pitch instead.")
  (l_pos_pitch m))

(cl:ensure-generic-function 'l_pos_yaw-val :lambda-list '(m))
(cl:defmethod l_pos_yaw-val ((m <ECJointMotordata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:l_pos_yaw-val is deprecated.  Use dynamic_biped-msg:l_pos_yaw instead.")
  (l_pos_yaw m))

(cl:ensure-generic-function 'l_pos_end-val :lambda-list '(m))
(cl:defmethod l_pos_end-val ((m <ECJointMotordata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:l_pos_end-val is deprecated.  Use dynamic_biped-msg:l_pos_end instead.")
  (l_pos_end m))

(cl:ensure-generic-function 'r_pos_pitch-val :lambda-list '(m))
(cl:defmethod r_pos_pitch-val ((m <ECJointMotordata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:r_pos_pitch-val is deprecated.  Use dynamic_biped-msg:r_pos_pitch instead.")
  (r_pos_pitch m))

(cl:ensure-generic-function 'r_pos_yaw-val :lambda-list '(m))
(cl:defmethod r_pos_yaw-val ((m <ECJointMotordata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:r_pos_yaw-val is deprecated.  Use dynamic_biped-msg:r_pos_yaw instead.")
  (r_pos_yaw m))

(cl:ensure-generic-function 'r_pos_end-val :lambda-list '(m))
(cl:defmethod r_pos_end-val ((m <ECJointMotordata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:r_pos_end-val is deprecated.  Use dynamic_biped-msg:r_pos_end instead.")
  (r_pos_end m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ECJointMotordata>) ostream)
  "Serializes a message object of type '<ECJointMotordata>"
  (cl:let* ((signed (cl:slot-value msg 'l_pos_pitch)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'l_pos_yaw)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'l_pos_end)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'r_pos_pitch)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'r_pos_yaw)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'r_pos_end)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ECJointMotordata>) istream)
  "Deserializes a message object of type '<ECJointMotordata>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'l_pos_pitch) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'l_pos_yaw) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'l_pos_end) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'r_pos_pitch) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'r_pos_yaw) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'r_pos_end) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ECJointMotordata>)))
  "Returns string type for a message object of type '<ECJointMotordata>"
  "dynamic_biped/ECJointMotordata")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ECJointMotordata)))
  "Returns string type for a message object of type 'ECJointMotordata"
  "dynamic_biped/ECJointMotordata")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ECJointMotordata>)))
  "Returns md5sum for a message object of type '<ECJointMotordata>"
  "da126c0fb22352a555bc24cbdebe8a85")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ECJointMotordata)))
  "Returns md5sum for a message object of type 'ECJointMotordata"
  "da126c0fb22352a555bc24cbdebe8a85")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ECJointMotordata>)))
  "Returns full string definition for message of type '<ECJointMotordata>"
  (cl:format cl:nil "int32 l_pos_pitch ~%int32 l_pos_yaw~%int32 l_pos_end~%int32 r_pos_pitch ~%int32 r_pos_yaw~%int32 r_pos_end~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ECJointMotordata)))
  "Returns full string definition for message of type 'ECJointMotordata"
  (cl:format cl:nil "int32 l_pos_pitch ~%int32 l_pos_yaw~%int32 l_pos_end~%int32 r_pos_pitch ~%int32 r_pos_yaw~%int32 r_pos_end~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ECJointMotordata>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ECJointMotordata>))
  "Converts a ROS message object to a list"
  (cl:list 'ECJointMotordata
    (cl:cons ':l_pos_pitch (l_pos_pitch msg))
    (cl:cons ':l_pos_yaw (l_pos_yaw msg))
    (cl:cons ':l_pos_end (l_pos_end msg))
    (cl:cons ':r_pos_pitch (r_pos_pitch msg))
    (cl:cons ':r_pos_yaw (r_pos_yaw msg))
    (cl:cons ':r_pos_end (r_pos_end msg))
))
