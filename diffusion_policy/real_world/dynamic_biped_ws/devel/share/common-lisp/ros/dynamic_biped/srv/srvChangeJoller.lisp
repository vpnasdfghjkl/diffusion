; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude srvChangeJoller-request.msg.html

(cl:defclass <srvChangeJoller-request> (roslisp-msg-protocol:ros-message)
  ((l_pos
    :reader l_pos
    :initarg :l_pos
    :type cl:integer
    :initform 0)
   (r_pos
    :reader r_pos
    :initarg :r_pos
    :type cl:integer
    :initform 0))
)

(cl:defclass srvChangeJoller-request (<srvChangeJoller-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvChangeJoller-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvChangeJoller-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvChangeJoller-request> is deprecated: use dynamic_biped-srv:srvChangeJoller-request instead.")))

(cl:ensure-generic-function 'l_pos-val :lambda-list '(m))
(cl:defmethod l_pos-val ((m <srvChangeJoller-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:l_pos-val is deprecated.  Use dynamic_biped-srv:l_pos instead.")
  (l_pos m))

(cl:ensure-generic-function 'r_pos-val :lambda-list '(m))
(cl:defmethod r_pos-val ((m <srvChangeJoller-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:r_pos-val is deprecated.  Use dynamic_biped-srv:r_pos instead.")
  (r_pos m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvChangeJoller-request>) ostream)
  "Serializes a message object of type '<srvChangeJoller-request>"
  (cl:let* ((signed (cl:slot-value msg 'l_pos)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'r_pos)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvChangeJoller-request>) istream)
  "Deserializes a message object of type '<srvChangeJoller-request>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'l_pos) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'r_pos) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvChangeJoller-request>)))
  "Returns string type for a service object of type '<srvChangeJoller-request>"
  "dynamic_biped/srvChangeJollerRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvChangeJoller-request)))
  "Returns string type for a service object of type 'srvChangeJoller-request"
  "dynamic_biped/srvChangeJollerRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvChangeJoller-request>)))
  "Returns md5sum for a message object of type '<srvChangeJoller-request>"
  "9e5320c57603a9a2673e4a9b69a7b21e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvChangeJoller-request)))
  "Returns md5sum for a message object of type 'srvChangeJoller-request"
  "9e5320c57603a9a2673e4a9b69a7b21e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvChangeJoller-request>)))
  "Returns full string definition for message of type '<srvChangeJoller-request>"
  (cl:format cl:nil "int32 l_pos~%int32 r_pos~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvChangeJoller-request)))
  "Returns full string definition for message of type 'srvChangeJoller-request"
  (cl:format cl:nil "int32 l_pos~%int32 r_pos~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvChangeJoller-request>))
  (cl:+ 0
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvChangeJoller-request>))
  "Converts a ROS message object to a list"
  (cl:list 'srvChangeJoller-request
    (cl:cons ':l_pos (l_pos msg))
    (cl:cons ':r_pos (r_pos msg))
))
;//! \htmlinclude srvChangeJoller-response.msg.html

(cl:defclass <srvChangeJoller-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass srvChangeJoller-response (<srvChangeJoller-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvChangeJoller-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvChangeJoller-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvChangeJoller-response> is deprecated: use dynamic_biped-srv:srvChangeJoller-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <srvChangeJoller-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:result-val is deprecated.  Use dynamic_biped-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvChangeJoller-response>) ostream)
  "Serializes a message object of type '<srvChangeJoller-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'result) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvChangeJoller-response>) istream)
  "Deserializes a message object of type '<srvChangeJoller-response>"
    (cl:setf (cl:slot-value msg 'result) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvChangeJoller-response>)))
  "Returns string type for a service object of type '<srvChangeJoller-response>"
  "dynamic_biped/srvChangeJollerResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvChangeJoller-response)))
  "Returns string type for a service object of type 'srvChangeJoller-response"
  "dynamic_biped/srvChangeJollerResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvChangeJoller-response>)))
  "Returns md5sum for a message object of type '<srvChangeJoller-response>"
  "9e5320c57603a9a2673e4a9b69a7b21e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvChangeJoller-response)))
  "Returns md5sum for a message object of type 'srvChangeJoller-response"
  "9e5320c57603a9a2673e4a9b69a7b21e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvChangeJoller-response>)))
  "Returns full string definition for message of type '<srvChangeJoller-response>"
  (cl:format cl:nil "bool result~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvChangeJoller-response)))
  "Returns full string definition for message of type 'srvChangeJoller-response"
  (cl:format cl:nil "bool result~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvChangeJoller-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvChangeJoller-response>))
  "Converts a ROS message object to a list"
  (cl:list 'srvChangeJoller-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'srvChangeJoller)))
  'srvChangeJoller-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'srvChangeJoller)))
  'srvChangeJoller-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvChangeJoller)))
  "Returns string type for a service object of type '<srvChangeJoller>"
  "dynamic_biped/srvChangeJoller")