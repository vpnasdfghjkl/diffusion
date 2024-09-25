; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-msg)


;//! \htmlinclude armHandPose.msg.html

(cl:defclass <armHandPose> (roslisp-msg-protocol:ros-message)
  ((pos_xyz
    :reader pos_xyz
    :initarg :pos_xyz
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (quat_xyzw
    :reader quat_xyzw
    :initarg :quat_xyzw
    :type (cl:vector cl:float)
   :initform (cl:make-array 4 :element-type 'cl:float :initial-element 0.0))
   (joint_angles
    :reader joint_angles
    :initarg :joint_angles
    :type (cl:vector cl:float)
   :initform (cl:make-array 7 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass armHandPose (<armHandPose>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <armHandPose>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'armHandPose)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-msg:<armHandPose> is deprecated: use dynamic_biped-msg:armHandPose instead.")))

(cl:ensure-generic-function 'pos_xyz-val :lambda-list '(m))
(cl:defmethod pos_xyz-val ((m <armHandPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:pos_xyz-val is deprecated.  Use dynamic_biped-msg:pos_xyz instead.")
  (pos_xyz m))

(cl:ensure-generic-function 'quat_xyzw-val :lambda-list '(m))
(cl:defmethod quat_xyzw-val ((m <armHandPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:quat_xyzw-val is deprecated.  Use dynamic_biped-msg:quat_xyzw instead.")
  (quat_xyzw m))

(cl:ensure-generic-function 'joint_angles-val :lambda-list '(m))
(cl:defmethod joint_angles-val ((m <armHandPose>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-msg:joint_angles-val is deprecated.  Use dynamic_biped-msg:joint_angles instead.")
  (joint_angles m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <armHandPose>) ostream)
  "Serializes a message object of type '<armHandPose>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'pos_xyz))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'quat_xyzw))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'joint_angles))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <armHandPose>) istream)
  "Deserializes a message object of type '<armHandPose>"
  (cl:setf (cl:slot-value msg 'pos_xyz) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'pos_xyz)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'quat_xyzw) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'quat_xyzw)))
    (cl:dotimes (i 4)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'joint_angles) (cl:make-array 7))
  (cl:let ((vals (cl:slot-value msg 'joint_angles)))
    (cl:dotimes (i 7)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<armHandPose>)))
  "Returns string type for a message object of type '<armHandPose>"
  "dynamic_biped/armHandPose")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'armHandPose)))
  "Returns string type for a message object of type 'armHandPose"
  "dynamic_biped/armHandPose")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<armHandPose>)))
  "Returns md5sum for a message object of type '<armHandPose>"
  "ae653a36c40e6220e166614d7037aae2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'armHandPose)))
  "Returns md5sum for a message object of type 'armHandPose"
  "ae653a36c40e6220e166614d7037aae2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<armHandPose>)))
  "Returns full string definition for message of type '<armHandPose>"
  (cl:format cl:nil "float64[3] pos_xyz~%float64[4] quat_xyzw~%~%float64[7] joint_angles~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'armHandPose)))
  "Returns full string definition for message of type 'armHandPose"
  (cl:format cl:nil "float64[3] pos_xyz~%float64[4] quat_xyzw~%~%float64[7] joint_angles~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <armHandPose>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'pos_xyz) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'quat_xyzw) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'joint_angles) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <armHandPose>))
  "Converts a ROS message object to a list"
  (cl:list 'armHandPose
    (cl:cons ':pos_xyz (pos_xyz msg))
    (cl:cons ':quat_xyzw (quat_xyzw msg))
    (cl:cons ':joint_angles (joint_angles msg))
))
