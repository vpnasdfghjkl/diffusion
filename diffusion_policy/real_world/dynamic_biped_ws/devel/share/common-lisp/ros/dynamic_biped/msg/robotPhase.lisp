; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude robotPhase.msg.html

(cl:defclass <robotPhase> (roslisp-msg-protocol:ros-message)
  ((mainPhase
    :reader mainPhase
    :initarg :mainPhase
    :type cl:fixnum
    :initform 0)
   (subPhase
    :reader subPhase
    :initarg :subPhase
    :type cl:fixnum
    :initform 0))
)

(cl:defclass robotPhase (<robotPhase>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <robotPhase>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'robotPhase)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<robotPhase> is deprecated: use dynamic_biped-msg:robotPhase instead.")))

(cl:ensure-generic-function 'mainPhase-val :lambda-list '(m))
(cl:defmethod mainPhase-val ((m <robotPhase>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:mainPhase-val is deprecated.  Use dynamic_biped-msg:mainPhase instead.")
  (mainPhase m))

(cl:ensure-generic-function 'subPhase-val :lambda-list '(m))
(cl:defmethod subPhase-val ((m <robotPhase>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:subPhase-val is deprecated.  Use dynamic_biped-msg:subPhase instead.")
  (subPhase m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <robotPhase>) ostream)
  "Serializes a message object of type '<robotPhase>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mainPhase)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'subPhase)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <robotPhase>) istream)
  "Deserializes a message object of type '<robotPhase>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mainPhase)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'subPhase)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<robotPhase>)))
  "Returns string type for a message object of type '<robotPhase>"
  "dynamic_biped/robotPhase")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'robotPhase)))
  "Returns string type for a message object of type 'robotPhase"
  "dynamic_biped/robotPhase")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<robotPhase>)))
  "Returns md5sum for a message object of type '<robotPhase>"
  "26cd19545acfae4dd7729b92456b2c32")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'robotPhase)))
  "Returns md5sum for a message object of type 'robotPhase"
  "26cd19545acfae4dd7729b92456b2c32")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<robotPhase>)))
  "Returns full string definition for message of type '<robotPhase>"
  (cl:format cl:nil "uint8 mainPhase~%uint8 subPhase~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'robotPhase)))
  "Returns full string definition for message of type 'robotPhase"
  (cl:format cl:nil "uint8 mainPhase~%uint8 subPhase~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <robotPhase>))
  (cl:+ 0
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <robotPhase>))
  "Converts a ROS message object to a list"
  (cl:list 'robotPhase
    (cl:cons ':mainPhase (mainPhase msg))
    (cl:cons ':subPhase (subPhase msg))
))
