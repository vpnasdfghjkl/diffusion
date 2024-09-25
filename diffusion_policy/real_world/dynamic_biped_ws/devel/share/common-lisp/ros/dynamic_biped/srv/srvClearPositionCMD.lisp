; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude srvClearPositionCMD-request.msg.html

(cl:defclass <srvClearPositionCMD-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass srvClearPositionCMD-request (<srvClearPositionCMD-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvClearPositionCMD-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvClearPositionCMD-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvClearPositionCMD-request> is deprecated: use dynamic_biped-srv:srvClearPositionCMD-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvClearPositionCMD-request>) ostream)
  "Serializes a message object of type '<srvClearPositionCMD-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvClearPositionCMD-request>) istream)
  "Deserializes a message object of type '<srvClearPositionCMD-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvClearPositionCMD-request>)))
  "Returns string type for a service object of type '<srvClearPositionCMD-request>"
  "dynamic_biped/srvClearPositionCMDRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvClearPositionCMD-request)))
  "Returns string type for a service object of type 'srvClearPositionCMD-request"
  "dynamic_biped/srvClearPositionCMDRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvClearPositionCMD-request>)))
  "Returns md5sum for a message object of type '<srvClearPositionCMD-request>"
  "4414c67819626a1b8e0f043a9a0d6c9a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvClearPositionCMD-request)))
  "Returns md5sum for a message object of type 'srvClearPositionCMD-request"
  "4414c67819626a1b8e0f043a9a0d6c9a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvClearPositionCMD-request>)))
  "Returns full string definition for message of type '<srvClearPositionCMD-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvClearPositionCMD-request)))
  "Returns full string definition for message of type 'srvClearPositionCMD-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvClearPositionCMD-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvClearPositionCMD-request>))
  "Converts a ROS message object to a list"
  (cl:list 'srvClearPositionCMD-request
))
;//! \htmlinclude srvClearPositionCMD-response.msg.html

(cl:defclass <srvClearPositionCMD-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:fixnum
    :initform 0))
)

(cl:defclass srvClearPositionCMD-response (<srvClearPositionCMD-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvClearPositionCMD-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvClearPositionCMD-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvClearPositionCMD-response> is deprecated: use dynamic_biped-srv:srvClearPositionCMD-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <srvClearPositionCMD-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:result-val is deprecated.  Use dynamic_biped-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvClearPositionCMD-response>) ostream)
  "Serializes a message object of type '<srvClearPositionCMD-response>"
  (cl:let* ((signed (cl:slot-value msg 'result)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvClearPositionCMD-response>) istream)
  "Deserializes a message object of type '<srvClearPositionCMD-response>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'result) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvClearPositionCMD-response>)))
  "Returns string type for a service object of type '<srvClearPositionCMD-response>"
  "dynamic_biped/srvClearPositionCMDResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvClearPositionCMD-response)))
  "Returns string type for a service object of type 'srvClearPositionCMD-response"
  "dynamic_biped/srvClearPositionCMDResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvClearPositionCMD-response>)))
  "Returns md5sum for a message object of type '<srvClearPositionCMD-response>"
  "4414c67819626a1b8e0f043a9a0d6c9a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvClearPositionCMD-response)))
  "Returns md5sum for a message object of type 'srvClearPositionCMD-response"
  "4414c67819626a1b8e0f043a9a0d6c9a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvClearPositionCMD-response>)))
  "Returns full string definition for message of type '<srvClearPositionCMD-response>"
  (cl:format cl:nil "int8 result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvClearPositionCMD-response)))
  "Returns full string definition for message of type 'srvClearPositionCMD-response"
  (cl:format cl:nil "int8 result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvClearPositionCMD-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvClearPositionCMD-response>))
  "Converts a ROS message object to a list"
  (cl:list 'srvClearPositionCMD-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'srvClearPositionCMD)))
  'srvClearPositionCMD-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'srvClearPositionCMD)))
  'srvClearPositionCMD-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvClearPositionCMD)))
  "Returns string type for a service object of type '<srvClearPositionCMD>"
  "dynamic_biped/srvClearPositionCMD")