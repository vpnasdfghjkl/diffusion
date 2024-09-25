; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude robotHeadMotionData.msg.html

(cl:defclass <robotHeadMotionData> (roslisp-msg-protocol:ros-message)
  ((joint_data
    :reader joint_data
    :initarg :joint_data
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass robotHeadMotionData (<robotHeadMotionData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <robotHeadMotionData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'robotHeadMotionData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<robotHeadMotionData> is deprecated: use dynamic_biped-msg:robotHeadMotionData instead.")))

(cl:ensure-generic-function 'joint_data-val :lambda-list '(m))
(cl:defmethod joint_data-val ((m <robotHeadMotionData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:joint_data-val is deprecated.  Use dynamic_biped-msg:joint_data instead.")
  (joint_data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <robotHeadMotionData>) ostream)
  "Serializes a message object of type '<robotHeadMotionData>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'joint_data))))
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
   (cl:slot-value msg 'joint_data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <robotHeadMotionData>) istream)
  "Deserializes a message object of type '<robotHeadMotionData>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'joint_data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'joint_data)))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<robotHeadMotionData>)))
  "Returns string type for a message object of type '<robotHeadMotionData>"
  "dynamic_biped/robotHeadMotionData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'robotHeadMotionData)))
  "Returns string type for a message object of type 'robotHeadMotionData"
  "dynamic_biped/robotHeadMotionData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<robotHeadMotionData>)))
  "Returns md5sum for a message object of type '<robotHeadMotionData>"
  "400001e7cf73111efbced59084cb481a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'robotHeadMotionData)))
  "Returns md5sum for a message object of type 'robotHeadMotionData"
  "400001e7cf73111efbced59084cb481a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<robotHeadMotionData>)))
  "Returns full string definition for message of type '<robotHeadMotionData>"
  (cl:format cl:nil "float64[] joint_data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'robotHeadMotionData)))
  "Returns full string definition for message of type 'robotHeadMotionData"
  (cl:format cl:nil "float64[] joint_data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <robotHeadMotionData>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'joint_data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <robotHeadMotionData>))
  "Converts a ROS message object to a list"
  (cl:list 'robotHeadMotionData
    (cl:cons ':joint_data (joint_data msg))
))
