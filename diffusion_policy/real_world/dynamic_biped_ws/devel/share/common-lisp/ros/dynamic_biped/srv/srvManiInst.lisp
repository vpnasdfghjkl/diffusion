; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude srvManiInst-request.msg.html

(cl:defclass <srvManiInst-request> (roslisp-msg-protocol:ros-message)
  ((intruction
    :reader intruction
    :initarg :intruction
    :type cl:string
    :initform ""))
)

(cl:defclass srvManiInst-request (<srvManiInst-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvManiInst-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvManiInst-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvManiInst-request> is deprecated: use dynamic_biped-srv:srvManiInst-request instead.")))

(cl:ensure-generic-function 'intruction-val :lambda-list '(m))
(cl:defmethod intruction-val ((m <srvManiInst-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:intruction-val is deprecated.  Use dynamic_biped-srv:intruction instead.")
  (intruction m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvManiInst-request>) ostream)
  "Serializes a message object of type '<srvManiInst-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'intruction))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'intruction))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvManiInst-request>) istream)
  "Deserializes a message object of type '<srvManiInst-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'intruction) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'intruction) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvManiInst-request>)))
  "Returns string type for a service object of type '<srvManiInst-request>"
  "dynamic_biped/srvManiInstRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvManiInst-request)))
  "Returns string type for a service object of type 'srvManiInst-request"
  "dynamic_biped/srvManiInstRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvManiInst-request>)))
  "Returns md5sum for a message object of type '<srvManiInst-request>"
  "b4c6768ce40d4cdb4227714a7786bd04")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvManiInst-request)))
  "Returns md5sum for a message object of type 'srvManiInst-request"
  "b4c6768ce40d4cdb4227714a7786bd04")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvManiInst-request>)))
  "Returns full string definition for message of type '<srvManiInst-request>"
  (cl:format cl:nil "string intruction~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvManiInst-request)))
  "Returns full string definition for message of type 'srvManiInst-request"
  (cl:format cl:nil "string intruction~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvManiInst-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'intruction))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvManiInst-request>))
  "Converts a ROS message object to a list"
  (cl:list 'srvManiInst-request
    (cl:cons ':intruction (intruction msg))
))
;//! \htmlinclude srvManiInst-response.msg.html

(cl:defclass <srvManiInst-response> (roslisp-msg-protocol:ros-message)
  ((stateRes
    :reader stateRes
    :initarg :stateRes
    :type cl:fixnum
    :initform 0))
)

(cl:defclass srvManiInst-response (<srvManiInst-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvManiInst-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvManiInst-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvManiInst-response> is deprecated: use dynamic_biped-srv:srvManiInst-response instead.")))

(cl:ensure-generic-function 'stateRes-val :lambda-list '(m))
(cl:defmethod stateRes-val ((m <srvManiInst-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:stateRes-val is deprecated.  Use dynamic_biped-srv:stateRes instead.")
  (stateRes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvManiInst-response>) ostream)
  "Serializes a message object of type '<srvManiInst-response>"
  (cl:let* ((signed (cl:slot-value msg 'stateRes)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvManiInst-response>) istream)
  "Deserializes a message object of type '<srvManiInst-response>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stateRes) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvManiInst-response>)))
  "Returns string type for a service object of type '<srvManiInst-response>"
  "dynamic_biped/srvManiInstResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvManiInst-response)))
  "Returns string type for a service object of type 'srvManiInst-response"
  "dynamic_biped/srvManiInstResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvManiInst-response>)))
  "Returns md5sum for a message object of type '<srvManiInst-response>"
  "b4c6768ce40d4cdb4227714a7786bd04")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvManiInst-response)))
  "Returns md5sum for a message object of type 'srvManiInst-response"
  "b4c6768ce40d4cdb4227714a7786bd04")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvManiInst-response>)))
  "Returns full string definition for message of type '<srvManiInst-response>"
  (cl:format cl:nil "int8 stateRes~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvManiInst-response)))
  "Returns full string definition for message of type 'srvManiInst-response"
  (cl:format cl:nil "int8 stateRes~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvManiInst-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvManiInst-response>))
  "Converts a ROS message object to a list"
  (cl:list 'srvManiInst-response
    (cl:cons ':stateRes (stateRes msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'srvManiInst)))
  'srvManiInst-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'srvManiInst)))
  'srvManiInst-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvManiInst)))
  "Returns string type for a service object of type '<srvManiInst>"
  "dynamic_biped/srvManiInst")