; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude srvchangeCtlMode-request.msg.html

(cl:defclass <srvchangeCtlMode-request> (roslisp-msg-protocol:ros-message)
  ((masterID
    :reader masterID
    :initarg :masterID
    :type cl:fixnum
    :initform 0)
   (control_mode
    :reader control_mode
    :initarg :control_mode
    :type cl:fixnum
    :initform 0))
)

(cl:defclass srvchangeCtlMode-request (<srvchangeCtlMode-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvchangeCtlMode-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvchangeCtlMode-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvchangeCtlMode-request> is deprecated: use dynamic_biped-srv:srvchangeCtlMode-request instead.")))

(cl:ensure-generic-function 'masterID-val :lambda-list '(m))
(cl:defmethod masterID-val ((m <srvchangeCtlMode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:masterID-val is deprecated.  Use dynamic_biped-srv:masterID instead.")
  (masterID m))

(cl:ensure-generic-function 'control_mode-val :lambda-list '(m))
(cl:defmethod control_mode-val ((m <srvchangeCtlMode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:control_mode-val is deprecated.  Use dynamic_biped-srv:control_mode instead.")
  (control_mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvchangeCtlMode-request>) ostream)
  "Serializes a message object of type '<srvchangeCtlMode-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'masterID)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'control_mode)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvchangeCtlMode-request>) istream)
  "Deserializes a message object of type '<srvchangeCtlMode-request>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'masterID)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'control_mode)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvchangeCtlMode-request>)))
  "Returns string type for a service object of type '<srvchangeCtlMode-request>"
  "dynamic_biped/srvchangeCtlModeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvchangeCtlMode-request)))
  "Returns string type for a service object of type 'srvchangeCtlMode-request"
  "dynamic_biped/srvchangeCtlModeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvchangeCtlMode-request>)))
  "Returns md5sum for a message object of type '<srvchangeCtlMode-request>"
  "55401bc31181d236576d88caa73b57c6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvchangeCtlMode-request)))
  "Returns md5sum for a message object of type 'srvchangeCtlMode-request"
  "55401bc31181d236576d88caa73b57c6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvchangeCtlMode-request>)))
  "Returns full string definition for message of type '<srvchangeCtlMode-request>"
  (cl:format cl:nil "uint8 masterID~%uint8 control_mode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvchangeCtlMode-request)))
  "Returns full string definition for message of type 'srvchangeCtlMode-request"
  (cl:format cl:nil "uint8 masterID~%uint8 control_mode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvchangeCtlMode-request>))
  (cl:+ 0
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvchangeCtlMode-request>))
  "Converts a ROS message object to a list"
  (cl:list 'srvchangeCtlMode-request
    (cl:cons ':masterID (masterID msg))
    (cl:cons ':control_mode (control_mode msg))
))
;//! \htmlinclude srvchangeCtlMode-response.msg.html

(cl:defclass <srvchangeCtlMode-response> (roslisp-msg-protocol:ros-message)
  ((control_mode
    :reader control_mode
    :initarg :control_mode
    :type cl:fixnum
    :initform 0))
)

(cl:defclass srvchangeCtlMode-response (<srvchangeCtlMode-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvchangeCtlMode-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvchangeCtlMode-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvchangeCtlMode-response> is deprecated: use dynamic_biped-srv:srvchangeCtlMode-response instead.")))

(cl:ensure-generic-function 'control_mode-val :lambda-list '(m))
(cl:defmethod control_mode-val ((m <srvchangeCtlMode-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:control_mode-val is deprecated.  Use dynamic_biped-srv:control_mode instead.")
  (control_mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvchangeCtlMode-response>) ostream)
  "Serializes a message object of type '<srvchangeCtlMode-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'control_mode)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvchangeCtlMode-response>) istream)
  "Deserializes a message object of type '<srvchangeCtlMode-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'control_mode)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvchangeCtlMode-response>)))
  "Returns string type for a service object of type '<srvchangeCtlMode-response>"
  "dynamic_biped/srvchangeCtlModeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvchangeCtlMode-response)))
  "Returns string type for a service object of type 'srvchangeCtlMode-response"
  "dynamic_biped/srvchangeCtlModeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvchangeCtlMode-response>)))
  "Returns md5sum for a message object of type '<srvchangeCtlMode-response>"
  "55401bc31181d236576d88caa73b57c6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvchangeCtlMode-response)))
  "Returns md5sum for a message object of type 'srvchangeCtlMode-response"
  "55401bc31181d236576d88caa73b57c6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvchangeCtlMode-response>)))
  "Returns full string definition for message of type '<srvchangeCtlMode-response>"
  (cl:format cl:nil "uint8 control_mode~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvchangeCtlMode-response)))
  "Returns full string definition for message of type 'srvchangeCtlMode-response"
  (cl:format cl:nil "uint8 control_mode~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvchangeCtlMode-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvchangeCtlMode-response>))
  "Converts a ROS message object to a list"
  (cl:list 'srvchangeCtlMode-response
    (cl:cons ':control_mode (control_mode msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'srvchangeCtlMode)))
  'srvchangeCtlMode-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'srvchangeCtlMode)))
  'srvchangeCtlMode-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvchangeCtlMode)))
  "Returns string type for a service object of type '<srvchangeCtlMode>"
  "dynamic_biped/srvchangeCtlMode")