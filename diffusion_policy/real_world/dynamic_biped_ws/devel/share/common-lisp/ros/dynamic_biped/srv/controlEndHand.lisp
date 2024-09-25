; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude controlEndHand-request.msg.html

(cl:defclass <controlEndHand-request> (roslisp-msg-protocol:ros-message)
  ((left_hand_position
    :reader left_hand_position
    :initarg :left_hand_position
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0))
   (right_hand_position
    :reader right_hand_position
    :initarg :right_hand_position
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass controlEndHand-request (<controlEndHand-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <controlEndHand-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'controlEndHand-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<controlEndHand-request> is deprecated: use dynamic_biped-srv:controlEndHand-request instead.")))

(cl:ensure-generic-function 'left_hand_position-val :lambda-list '(m))
(cl:defmethod left_hand_position-val ((m <controlEndHand-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:left_hand_position-val is deprecated.  Use dynamic_biped-srv:left_hand_position instead.")
  (left_hand_position m))

(cl:ensure-generic-function 'right_hand_position-val :lambda-list '(m))
(cl:defmethod right_hand_position-val ((m <controlEndHand-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:right_hand_position-val is deprecated.  Use dynamic_biped-srv:right_hand_position instead.")
  (right_hand_position m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <controlEndHand-request>) ostream)
  "Serializes a message object of type '<controlEndHand-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'left_hand_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'left_hand_position))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'right_hand_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'right_hand_position))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <controlEndHand-request>) istream)
  "Deserializes a message object of type '<controlEndHand-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'left_hand_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'left_hand_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'right_hand_position) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'right_hand_position)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<controlEndHand-request>)))
  "Returns string type for a service object of type '<controlEndHand-request>"
  "dynamic_biped/controlEndHandRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'controlEndHand-request)))
  "Returns string type for a service object of type 'controlEndHand-request"
  "dynamic_biped/controlEndHandRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<controlEndHand-request>)))
  "Returns md5sum for a message object of type '<controlEndHand-request>"
  "741989817a59889e258aa9e94c7ada8a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'controlEndHand-request)))
  "Returns md5sum for a message object of type 'controlEndHand-request"
  "741989817a59889e258aa9e94c7ada8a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<controlEndHand-request>)))
  "Returns full string definition for message of type '<controlEndHand-request>"
  (cl:format cl:nil "uint8[] left_hand_position~%uint8[] right_hand_position~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'controlEndHand-request)))
  "Returns full string definition for message of type 'controlEndHand-request"
  (cl:format cl:nil "uint8[] left_hand_position~%uint8[] right_hand_position~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <controlEndHand-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'left_hand_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'right_hand_position) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <controlEndHand-request>))
  "Converts a ROS message object to a list"
  (cl:list 'controlEndHand-request
    (cl:cons ':left_hand_position (left_hand_position msg))
    (cl:cons ':right_hand_position (right_hand_position msg))
))
;//! \htmlinclude controlEndHand-response.msg.html

(cl:defclass <controlEndHand-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass controlEndHand-response (<controlEndHand-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <controlEndHand-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'controlEndHand-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<controlEndHand-response> is deprecated: use dynamic_biped-srv:controlEndHand-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <controlEndHand-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:result-val is deprecated.  Use dynamic_biped-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <controlEndHand-response>) ostream)
  "Serializes a message object of type '<controlEndHand-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'result) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <controlEndHand-response>) istream)
  "Deserializes a message object of type '<controlEndHand-response>"
    (cl:setf (cl:slot-value msg 'result) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<controlEndHand-response>)))
  "Returns string type for a service object of type '<controlEndHand-response>"
  "dynamic_biped/controlEndHandResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'controlEndHand-response)))
  "Returns string type for a service object of type 'controlEndHand-response"
  "dynamic_biped/controlEndHandResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<controlEndHand-response>)))
  "Returns md5sum for a message object of type '<controlEndHand-response>"
  "741989817a59889e258aa9e94c7ada8a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'controlEndHand-response)))
  "Returns md5sum for a message object of type 'controlEndHand-response"
  "741989817a59889e258aa9e94c7ada8a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<controlEndHand-response>)))
  "Returns full string definition for message of type '<controlEndHand-response>"
  (cl:format cl:nil "bool result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'controlEndHand-response)))
  "Returns full string definition for message of type 'controlEndHand-response"
  (cl:format cl:nil "bool result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <controlEndHand-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <controlEndHand-response>))
  "Converts a ROS message object to a list"
  (cl:list 'controlEndHand-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'controlEndHand)))
  'controlEndHand-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'controlEndHand)))
  'controlEndHand-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'controlEndHand)))
  "Returns string type for a service object of type '<controlEndHand>"
  "dynamic_biped/controlEndHand")