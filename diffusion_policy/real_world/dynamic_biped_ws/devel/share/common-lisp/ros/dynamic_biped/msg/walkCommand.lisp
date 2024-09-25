; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude walkCommand.msg.html

(cl:defclass <walkCommand> (roslisp-msg-protocol:ros-message)
  ((mode
    :reader mode
    :initarg :mode
    :type cl:fixnum
    :initform 0)
   (values
    :reader values
    :initarg :values
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass walkCommand (<walkCommand>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <walkCommand>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'walkCommand)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<walkCommand> is deprecated: use dynamic_biped-msg:walkCommand instead.")))

(cl:ensure-generic-function 'mode-val :lambda-list '(m))
(cl:defmethod mode-val ((m <walkCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:mode-val is deprecated.  Use dynamic_biped-msg:mode instead.")
  (mode m))

(cl:ensure-generic-function 'values-val :lambda-list '(m))
(cl:defmethod values-val ((m <walkCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:values-val is deprecated.  Use dynamic_biped-msg:values instead.")
  (values m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <walkCommand>) ostream)
  "Serializes a message object of type '<walkCommand>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'values))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'values))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <walkCommand>) istream)
  "Deserializes a message object of type '<walkCommand>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'values) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'values)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<walkCommand>)))
  "Returns string type for a message object of type '<walkCommand>"
  "dynamic_biped/walkCommand")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'walkCommand)))
  "Returns string type for a message object of type 'walkCommand"
  "dynamic_biped/walkCommand")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<walkCommand>)))
  "Returns md5sum for a message object of type '<walkCommand>"
  "5b32312c0e7618ca6c317ce9272f2ca7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'walkCommand)))
  "Returns md5sum for a message object of type 'walkCommand"
  "5b32312c0e7618ca6c317ce9272f2ca7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<walkCommand>)))
  "Returns full string definition for message of type '<walkCommand>"
  (cl:format cl:nil "# mode: 0->PositionCommand | 1->VelocityCommand~%uint8 mode~%float64[] values~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'walkCommand)))
  "Returns full string definition for message of type 'walkCommand"
  (cl:format cl:nil "# mode: 0->PositionCommand | 1->VelocityCommand~%uint8 mode~%float64[] values~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <walkCommand>))
  (cl:+ 0
     1
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'values) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <walkCommand>))
  "Converts a ROS message object to a list"
  (cl:list 'walkCommand
    (cl:cons ':mode (mode msg))
    (cl:cons ':values (values msg))
))
