;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::changeArmCtrlMode)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'changeArmCtrlMode (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::CHANGEARMCTRLMODE")
  (make-package "DYNAMIC_BIPED::CHANGEARMCTRLMODE"))
(unless (find-package "DYNAMIC_BIPED::CHANGEARMCTRLMODEREQUEST")
  (make-package "DYNAMIC_BIPED::CHANGEARMCTRLMODEREQUEST"))
(unless (find-package "DYNAMIC_BIPED::CHANGEARMCTRLMODERESPONSE")
  (make-package "DYNAMIC_BIPED::CHANGEARMCTRLMODERESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::changeArmCtrlModeRequest
  :super ros::object
  :slots (_control_mode ))

(defmethod dynamic_biped::changeArmCtrlModeRequest
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

(defclass dynamic_biped::changeArmCtrlModeResponse
  :super ros::object
  :slots (_result ))

(defmethod dynamic_biped::changeArmCtrlModeResponse
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

(defclass dynamic_biped::changeArmCtrlMode
  :super ros::object
  :slots ())

(setf (get dynamic_biped::changeArmCtrlMode :md5sum-) "f89438f9d6f48f748eabe64775a22261")
(setf (get dynamic_biped::changeArmCtrlMode :datatype-) "dynamic_biped/changeArmCtrlMode")
(setf (get dynamic_biped::changeArmCtrlMode :request) dynamic_biped::changeArmCtrlModeRequest)
(setf (get dynamic_biped::changeArmCtrlMode :response) dynamic_biped::changeArmCtrlModeResponse)

(defmethod dynamic_biped::changeArmCtrlModeRequest
  (:response () (instance dynamic_biped::changeArmCtrlModeResponse :init)))

(setf (get dynamic_biped::changeArmCtrlModeRequest :md5sum-) "f89438f9d6f48f748eabe64775a22261")
(setf (get dynamic_biped::changeArmCtrlModeRequest :datatype-) "dynamic_biped/changeArmCtrlModeRequest")
(setf (get dynamic_biped::changeArmCtrlModeRequest :definition-)
      "bool control_mode
---
bool result


")

(setf (get dynamic_biped::changeArmCtrlModeResponse :md5sum-) "f89438f9d6f48f748eabe64775a22261")
(setf (get dynamic_biped::changeArmCtrlModeResponse :datatype-) "dynamic_biped/changeArmCtrlModeResponse")
(setf (get dynamic_biped::changeArmCtrlModeResponse :definition-)
      "bool control_mode
---
bool result


")



(provide :dynamic_biped/changeArmCtrlMode "f89438f9d6f48f748eabe64775a22261")


