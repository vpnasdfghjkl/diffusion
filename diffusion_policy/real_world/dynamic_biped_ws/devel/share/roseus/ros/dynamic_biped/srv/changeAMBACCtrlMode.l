;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::changeAMBACCtrlMode)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'changeAMBACCtrlMode (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::CHANGEAMBACCTRLMODE")
  (make-package "DYNAMIC_BIPED::CHANGEAMBACCTRLMODE"))
(unless (find-package "DYNAMIC_BIPED::CHANGEAMBACCTRLMODEREQUEST")
  (make-package "DYNAMIC_BIPED::CHANGEAMBACCTRLMODEREQUEST"))
(unless (find-package "DYNAMIC_BIPED::CHANGEAMBACCTRLMODERESPONSE")
  (make-package "DYNAMIC_BIPED::CHANGEAMBACCTRLMODERESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::changeAMBACCtrlModeRequest
  :super ros::object
  :slots (_control_mode ))

(defmethod dynamic_biped::changeAMBACCtrlModeRequest
  (:init
   (&key
    ((:control_mode __control_mode) nil)
    )
   (send-super :init)
   (setq _control_mode __control_mode)
   self)
  (:control_mode
   (&optional (__control_mode :null))
   (if (not (eq __control_mode :null)) (setq _control_mode __control_mode)) _control_mode)
  (:serialization-length
   ()
   (+
    ;; bool _control_mode
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _control_mode
       (if _control_mode (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _control_mode
     (setq _control_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass dynamic_biped::changeAMBACCtrlModeResponse
  :super ros::object
  :slots (_result ))

(defmethod dynamic_biped::changeAMBACCtrlModeResponse
  (:init
   (&key
    ((:result __result) nil)
    )
   (send-super :init)
   (setq _result __result)
   self)
  (:result
   (&optional (__result :null))
   (if (not (eq __result :null)) (setq _result __result)) _result)
  (:serialization-length
   ()
   (+
    ;; bool _result
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _result
       (if _result (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _result
     (setq _result (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass dynamic_biped::changeAMBACCtrlMode
  :super ros::object
  :slots ())

(setf (get dynamic_biped::changeAMBACCtrlMode :md5sum-) "f89438f9d6f48f748eabe64775a22261")
(setf (get dynamic_biped::changeAMBACCtrlMode :datatype-) "dynamic_biped/changeAMBACCtrlMode")
(setf (get dynamic_biped::changeAMBACCtrlMode :request) dynamic_biped::changeAMBACCtrlModeRequest)
(setf (get dynamic_biped::changeAMBACCtrlMode :response) dynamic_biped::changeAMBACCtrlModeResponse)

(defmethod dynamic_biped::changeAMBACCtrlModeRequest
  (:response () (instance dynamic_biped::changeAMBACCtrlModeResponse :init)))

(setf (get dynamic_biped::changeAMBACCtrlModeRequest :md5sum-) "f89438f9d6f48f748eabe64775a22261")
(setf (get dynamic_biped::changeAMBACCtrlModeRequest :datatype-) "dynamic_biped/changeAMBACCtrlModeRequest")
(setf (get dynamic_biped::changeAMBACCtrlModeRequest :definition-)
      "bool control_mode
---
bool result


")

(setf (get dynamic_biped::changeAMBACCtrlModeResponse :md5sum-) "f89438f9d6f48f748eabe64775a22261")
(setf (get dynamic_biped::changeAMBACCtrlModeResponse :datatype-) "dynamic_biped/changeAMBACCtrlModeResponse")
(setf (get dynamic_biped::changeAMBACCtrlModeResponse :definition-)
      "bool control_mode
---
bool result


")



(provide :dynamic_biped/changeAMBACCtrlMode "f89438f9d6f48f748eabe64775a22261")


