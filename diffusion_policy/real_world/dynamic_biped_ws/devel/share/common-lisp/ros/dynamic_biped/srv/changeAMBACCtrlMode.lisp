; Auto-generated. Do not edit!


(cl:in-package dynamic_biped-srv)


;//! \htmlinclude changeAMBACCtrlMode-request.msg.html

(cl:defclass <changeAMBACCtrlMode-request> (roslisp-msg-protocol:ros-message)
  ((control_mode
    :reader control_mode
    :initarg :control_mode
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass changeAMBACCtrlMode-request (<changeAMBACCtrlMode-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <changeAMBACCtrlMode-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'changeAMBACCtrlMode-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<changeAMBACCtrlMode-request> is deprecated: use dynamic_biped-srv:changeAMBACCtrlMode-request instead.")))

(cl:ensure-generic-function 'control_mode-val :lambda-list '(m))
(cl:defmethod control_mode-val ((m <changeAMBACCtrlMode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:control_mode-val is deprecated.  Use dynamic_biped-srv:control_mode instead.")
  (control_mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <changeAMBACCtrlMode-request>) ostream)
  "Serializes a message object of type '<changeAMBACCtrlMode-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'control_mode) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <changeAMBACCtrlMode-request>) istream)
  "Deserializes a message object of type '<changeAMBACCtrlMode-request>"
    (cl:setf (cl:slot-value msg 'control_mode) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<changeAMBACCtrlMode-request>)))
  "Returns string type for a service object of type '<changeAMBACCtrlMode-request>"
  "dynamic_biped/changeAMBACCtrlModeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'changeAMBACCtrlMode-request)))
  "Returns string type for a service object of type 'changeAMBACCtrlMode-request"
  "dynamic_biped/changeAMBACCtrlModeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<changeAMBACCtrlMode-request>)))
  "Returns md5sum for a message object of type '<changeAMBACCtrlMode-request>"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'changeAMBACCtrlMode-request)))
  "Returns md5sum for a message object of type 'changeAMBACCtrlMode-request"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<changeAMBACCtrlMode-request>)))
  "Returns full string definition for message of type '<changeAMBACCtrlMode-request>"
  (cl:format cl:nil "bool control_mode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'changeAMBACCtrlMode-request)))
  "Returns full string definition for message of type 'changeAMBACCtrlMode-request"
  (cl:format cl:nil "bool control_mode~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <changeAMBACCtrlMode-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <changeAMBACCtrlMode-request>))
  "Converts a ROS message object to a list"
  (cl:list 'changeAMBACCtrlMode-request
    (cl:cons ':control_mode (control_mode msg))
))
;//! \htmlinclude changeAMBACCtrlMode-response.msg.html

(cl:defclass <changeAMBACCtrlMode-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass changeAMBACCtrlMode-response (<changeAMBACCtrlMode-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <changeAMBACCtrlMode-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'changeAMBACCtrlMode-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name dynamic_biped-srv:<changeAMBACCtrlMode-response> is deprecated: use dynamic_biped-srv:changeAMBACCtrlMode-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <changeAMBACCtrlMode-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader dynamic_biped-srv:result-val is deprecated.  Use dynamic_biped-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <changeAMBACCtrlMode-response>) ostream)
  "Serializes a message object of type '<changeAMBACCtrlMode-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'result) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <changeAMBACCtrlMode-response>) istream)
  "Deserializes a message object of type '<changeAMBACCtrlMode-response>"
    (cl:setf (cl:slot-value msg 'result) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<changeAMBACCtrlMode-response>)))
  "Returns string type for a service object of type '<changeAMBACCtrlMode-response>"
  "dynamic_biped/changeAMBACCtrlModeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'changeAMBACCtrlMode-response)))
  "Returns string type for a service object of type 'changeAMBACCtrlMode-response"
  "dynamic_biped/changeAMBACCtrlModeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<changeAMBACCtrlMode-response>)))
  "Returns md5sum for a message object of type '<changeAMBACCtrlMode-response>"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'changeAMBACCtrlMode-response)))
  "Returns md5sum for a message object of type 'changeAMBACCtrlMode-response"
  "f89438f9d6f48f748eabe64775a22261")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<changeAMBACCtrlMode-response>)))
  "Returns full string definition for message of type '<changeAMBACCtrlMode-response>"
  (cl:format cl:nil "bool result~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'changeAMBACCtrlMode-response)))
  "Returns full string definition for message of type 'changeAMBACCtrlMode-response"
  (cl:format cl:nil "bool result~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <changeAMBACCtrlMode-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <changeAMBACCtrlMode-response>))
  "Converts a ROS message object to a list"
  (cl:list 'changeAMBACCtrlMode-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'changeAMBACCtrlMode)))
  'changeAMBACCtrlMode-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'changeAMBACCtrlMode)))
  'changeAMBACCtrlMode-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'changeAMBACCtrlMode)))
  "Returns string type for a service object of type '<changeAMBACCtrlMode>"
  "dynamic_biped/changeAMBACCtrlMode")