; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude changeArmCtrlMode-request.msg.html

(cl:defclass <changeArmCtrlMode-request> (roslisp-msg-protocol:ros-message)
  ((control_mode
    :reader control_mode
    :initarg :control_mode
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass changeArmCtrlMode-request (<changeArmCtrlMode-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <changeArmCtrlMode-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'changeArmCtrlMode-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<changeArmCtrlMode-request> is deprecated: use dynamic_biped-srv:changeArmCtrlMode-request instead.")))

(cl:ensure-generic-function 'control_mode-val :lambda-list '(m))
(cl:defmethod control_mode-val ((m <changeArmCtrlMode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:control_mode-val is deprecated.  Use dynamic_biped-srv:control_mode instead.")
  (control_mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <changeArmCtrlMode-request>) ostream)
  "Serializes a message object of type '<changeArmCtrlMode-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'control_mode) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <changeArmCtrlMode-request>) istream)
  "Deserializes a message object of type '<changeArmCtrlMode-request>"
    (cl:setf (cl:slot-value msg 'control_mode) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<changeArmCtrlMode-request>)))
  "Returns string type for a service object of type '<changeArmCtrlMode-request>"
  "dynamic_biped/changeArmCtrlModeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'changeArmCtrlMode-request)))
  "Returns string type for a service object of type 'changeArmCtrlMode-request"
  "dynamic_biped/changeArmCtrlModeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<changeArmCtrlMode-request>)))
  "Returns md5sum for a message object of type '<changeArmCtrlMode-request>"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'changeArmCtrlMode-request)))
  "Returns md5sum for a message object of type 'changeArmCtrlMode-request"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<changeArmCtrlMode-request>)))
  "Returns full string definition for message of type '<changeArmCtrlMode-request>"
  (cl:format cl:nil "bool control_mode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'changeArmCtrlMode-request)))
  "Returns full string definition for message of type 'changeArmCtrlMode-request"
  (cl:format cl:nil "bool control_mode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <changeArmCtrlMode-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <changeArmCtrlMode-request>))
  "Converts a ROS message object to a list"
  (cl:list 'changeArmCtrlMode-request
    (cl:cons ':control_mode (control_mode msg))
))
;//! \htmlinclude changeArmCtrlMode-response.msg.html

(cl:defclass <changeArmCtrlMode-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass changeArmCtrlMode-response (<changeArmCtrlMode-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <changeArmCtrlMode-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'changeArmCtrlMode-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<changeArmCtrlMode-response> is deprecated: use dynamic_biped-srv:changeArmCtrlMode-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <changeArmCtrlMode-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:result-val is deprecated.  Use dynamic_biped-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <changeArmCtrlMode-response>) ostream)
  "Serializes a message object of type '<changeArmCtrlMode-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'result) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <changeArmCtrlMode-response>) istream)
  "Deserializes a message object of type '<changeArmCtrlMode-response>"
    (cl:setf (cl:slot-value msg 'result) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<changeArmCtrlMode-response>)))
  "Returns string type for a service object of type '<changeArmCtrlMode-response>"
  "dynamic_biped/changeArmCtrlModeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'changeArmCtrlMode-response)))
  "Returns string type for a service object of type 'changeArmCtrlMode-response"
  "dynamic_biped/changeArmCtrlModeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<changeArmCtrlMode-response>)))
  "Returns md5sum for a message object of type '<changeArmCtrlMode-response>"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'changeArmCtrlMode-response)))
  "Returns md5sum for a message object of type 'changeArmCtrlMode-response"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<changeArmCtrlMode-response>)))
  "Returns full string definition for message of type '<changeArmCtrlMode-response>"
  (cl:format cl:nil "bool result~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'changeArmCtrlMode-response)))
  "Returns full string definition for message of type 'changeArmCtrlMode-response"
  (cl:format cl:nil "bool result~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <changeArmCtrlMode-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <changeArmCtrlMode-response>))
  "Converts a ROS message object to a list"
  (cl:list 'changeArmCtrlMode-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'changeArmCtrlMode)))
  'changeArmCtrlMode-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'changeArmCtrlMode)))
  'changeArmCtrlMode-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'changeArmCtrlMode)))
  "Returns string type for a service object of type '<changeArmCtrlMode>"
  "dynamic_biped/changeArmCtrlMode")