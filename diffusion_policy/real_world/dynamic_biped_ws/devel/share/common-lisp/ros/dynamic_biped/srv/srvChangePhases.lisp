; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude srvChangePhases-request.msg.html

(cl:defclass <srvChangePhases-request> (roslisp-msg-protocol:ros-message)
  ((masterID
    :reader masterID
    :initarg :masterID
    :type cl:fixnum
    :initform 0)
   (stateReq
    :reader stateReq
    :initarg :stateReq
    :type cl:string
    :initform "")
   (subState
    :reader subState
    :initarg :subState
    :type cl:string
    :initform ""))
)

(cl:defclass srvChangePhases-request (<srvChangePhases-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvChangePhases-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvChangePhases-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvChangePhases-request> is deprecated: use dynamic_biped-srv:srvChangePhases-request instead.")))

(cl:ensure-generic-function 'masterID-val :lambda-list '(m))
(cl:defmethod masterID-val ((m <srvChangePhases-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:masterID-val is deprecated.  Use dynamic_biped-srv:masterID instead.")
  (masterID m))

(cl:ensure-generic-function 'stateReq-val :lambda-list '(m))
(cl:defmethod stateReq-val ((m <srvChangePhases-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:stateReq-val is deprecated.  Use dynamic_biped-srv:stateReq instead.")
  (stateReq m))

(cl:ensure-generic-function 'subState-val :lambda-list '(m))
(cl:defmethod subState-val ((m <srvChangePhases-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:subState-val is deprecated.  Use dynamic_biped-srv:subState instead.")
  (subState m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvChangePhases-request>) ostream)
  "Serializes a message object of type '<srvChangePhases-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'masterID)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'stateReq))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'stateReq))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'subState))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'subState))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvChangePhases-request>) istream)
  "Deserializes a message object of type '<srvChangePhases-request>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'masterID)) (cl:read-byte istream))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stateReq) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'stateReq) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'subState) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'subState) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvChangePhases-request>)))
  "Returns string type for a service object of type '<srvChangePhases-request>"
  "dynamic_biped/srvChangePhasesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvChangePhases-request)))
  "Returns string type for a service object of type 'srvChangePhases-request"
  "dynamic_biped/srvChangePhasesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvChangePhases-request>)))
  "Returns md5sum for a message object of type '<srvChangePhases-request>"
  "f89a7d7f1ae7501d38174856aa23b24d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvChangePhases-request)))
  "Returns md5sum for a message object of type 'srvChangePhases-request"
  "f89a7d7f1ae7501d38174856aa23b24d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvChangePhases-request>)))
  "Returns full string definition for message of type '<srvChangePhases-request>"
  (cl:format cl:nil "uint8 masterID~%string stateReq~%string subState~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvChangePhases-request)))
  "Returns full string definition for message of type 'srvChangePhases-request"
  (cl:format cl:nil "uint8 masterID~%string stateReq~%string subState~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvChangePhases-request>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'stateReq))
     4 (cl:length (cl:slot-value msg 'subState))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvChangePhases-request>))
  "Converts a ROS message object to a list"
  (cl:list 'srvChangePhases-request
    (cl:cons ':masterID (masterID msg))
    (cl:cons ':stateReq (stateReq msg))
    (cl:cons ':subState (subState msg))
))
;//! \htmlinclude srvChangePhases-response.msg.html

(cl:defclass <srvChangePhases-response> (roslisp-msg-protocol:ros-message)
  ((stateRes
    :reader stateRes
    :initarg :stateRes
    :type cl:fixnum
    :initform 0))
)

(cl:defclass srvChangePhases-response (<srvChangePhases-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <srvChangePhases-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'srvChangePhases-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<srvChangePhases-response> is deprecated: use dynamic_biped-srv:srvChangePhases-response instead.")))

(cl:ensure-generic-function 'stateRes-val :lambda-list '(m))
(cl:defmethod stateRes-val ((m <srvChangePhases-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:stateRes-val is deprecated.  Use dynamic_biped-srv:stateRes instead.")
  (stateRes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <srvChangePhases-response>) ostream)
  "Serializes a message object of type '<srvChangePhases-response>"
  (cl:let* ((signed (cl:slot-value msg 'stateRes)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <srvChangePhases-response>) istream)
  "Deserializes a message object of type '<srvChangePhases-response>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stateRes) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<srvChangePhases-response>)))
  "Returns string type for a service object of type '<srvChangePhases-response>"
  "dynamic_biped/srvChangePhasesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvChangePhases-response)))
  "Returns string type for a service object of type 'srvChangePhases-response"
  "dynamic_biped/srvChangePhasesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<srvChangePhases-response>)))
  "Returns md5sum for a message object of type '<srvChangePhases-response>"
  "f89a7d7f1ae7501d38174856aa23b24d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'srvChangePhases-response)))
  "Returns md5sum for a message object of type 'srvChangePhases-response"
  "f89a7d7f1ae7501d38174856aa23b24d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<srvChangePhases-response>)))
  "Returns full string definition for message of type '<srvChangePhases-response>"
  (cl:format cl:nil "int16 stateRes~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'srvChangePhases-response)))
  "Returns full string definition for message of type 'srvChangePhases-response"
  (cl:format cl:nil "int16 stateRes~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <srvChangePhases-response>))
  (cl:+ 0
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <srvChangePhases-response>))
  "Converts a ROS message object to a list"
  (cl:list 'srvChangePhases-response
    (cl:cons ':stateRes (stateRes msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'srvChangePhases)))
  'srvChangePhases-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'srvChangePhases)))
  'srvChangePhases-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'srvChangePhases)))
  "Returns string type for a service object of type '<srvChangePhases>"
  "dynamic_biped/srvChangePhases")