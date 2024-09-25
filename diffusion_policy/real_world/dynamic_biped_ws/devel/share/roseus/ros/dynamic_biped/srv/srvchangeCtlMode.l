;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::srvchangeCtlMode)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'srvchangeCtlMode (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGECTLMODE")
  (make-package "DYNAMIC_BIPED::SRVCHANGECTLMODE"))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGECTLMODEREQUEST")
  (make-package "DYNAMIC_BIPED::SRVCHANGECTLMODEREQUEST"))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGECTLMODERESPONSE")
  (make-package "DYNAMIC_BIPED::SRVCHANGECTLMODERESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::srvchangeCtlModeRequest
  :super ros::object
  :slots (_masterID _control_mode ))

(defmethod dynamic_biped::srvchangeCtlModeRequest
  (:init
   (&key
    ((:masterID __masterID) 0)
    ((:control_mode __control_mode) 0)
    )
   (send-super :init)
   (setq _masterID (round __masterID))
   (setq _control_mode (round __control_mode))
   self)
  (:masterID
   (&optional __masterID)
   (if __masterID (setq _masterID __masterID)) _masterID)
  (:control_mode
   (&optional __control_mode)
   (if __control_mode (setq _control_mode __control_mode)) _control_mode)
  (:serialization-length
   ()
   (+
    ;; uint8 _masterID
    1
    ;; uint8 _control_mode
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _masterID
       (write-byte _masterID s)
     ;; uint8 _control_mode
       (write-byte _control_mode s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _masterID
     (setq _masterID (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _control_mode
     (setq _control_mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass dynamic_biped::srvchangeCtlModeResponse
  :super ros::object
  :slots (_control_mode ))

(defmethod dynamic_biped::srvchangeCtlModeResponse
  (:init
   (&key
    ((:control_mode __control_mode) 0)
    )
   (send-super :init)
   (setq _control_mode (round __control_mode))
   self)
  (:control_mode
   (&optional __control_mode)
   (if __control_mode (setq _control_mode __control_mode)) _control_mode)
  (:serialization-length
   ()
   (+
    ;; uint8 _control_mode
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _control_mode
       (write-byte _control_mode s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _control_mode
     (setq _control_mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass dynamic_biped::srvchangeCtlMode
  :super ros::object
  :slots ())

(setf (get dynamic_biped::srvchangeCtlMode :md5sum-) "55401bc31181d236576d88caa73b57c6")
(setf (get dynamic_biped::srvchangeCtlMode :datatype-) "dynamic_biped/srvchangeCtlMode")
(setf (get dynamic_biped::srvchangeCtlMode :request) dynamic_biped::srvchangeCtlModeRequest)
(setf (get dynamic_biped::srvchangeCtlMode :response) dynamic_biped::srvchangeCtlModeResponse)

(defmethod dynamic_biped::srvchangeCtlModeRequest
  (:response () (instance dynamic_biped::srvchangeCtlModeResponse :init)))

(setf (get dynamic_biped::srvchangeCtlModeRequest :md5sum-) "55401bc31181d236576d88caa73b57c6")
(setf (get dynamic_biped::srvchangeCtlModeRequest :datatype-) "dynamic_biped/srvchangeCtlModeRequest")
(setf (get dynamic_biped::srvchangeCtlModeRequest :definition-)
      "uint8 masterID
uint8 control_mode
---
uint8 control_mode

")

(setf (get dynamic_biped::srvchangeCtlModeResponse :md5sum-) "55401bc31181d236576d88caa73b57c6")
(setf (get dynamic_biped::srvchangeCtlModeResponse :datatype-) "dynamic_biped/srvchangeCtlModeResponse")
(setf (get dynamic_biped::srvchangeCtlModeResponse :definition-)
      "uint8 masterID
uint8 control_mode
---
uint8 control_mode

")



(provide :dynamic_biped/srvchangeCtlMode "55401bc31181d236576d88caa73b57c6")


