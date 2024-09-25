;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::srvClearPositionCMD)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'srvClearPositionCMD (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::SRVCLEARPOSITIONCMD")
  (make-package "DYNAMIC_BIPED::SRVCLEARPOSITIONCMD"))
(unless (find-package "DYNAMIC_BIPED::SRVCLEARPOSITIONCMDREQUEST")
  (make-package "DYNAMIC_BIPED::SRVCLEARPOSITIONCMDREQUEST"))
(unless (find-package "DYNAMIC_BIPED::SRVCLEARPOSITIONCMDRESPONSE")
  (make-package "DYNAMIC_BIPED::SRVCLEARPOSITIONCMDRESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::srvClearPositionCMDRequest
  :super ros::object
  :slots ())

(defmethod dynamic_biped::srvClearPositionCMDRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass dynamic_biped::srvClearPositionCMDResponse
  :super ros::object
  :slots (_result ))

(defmethod dynamic_biped::srvClearPositionCMDResponse
  (:init
   (&key
    ((:result __result) 0)
    )
   (send-super :init)
   (setq _result (round __result))
   self)
  (:result
   (&optional __result)
   (if __result (setq _result __result)) _result)
  (:serialization-length
   ()
   (+
    ;; int8 _result
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _result
       (write-byte _result s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _result
     (setq _result (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _result 127) (setq _result (- _result 256)))
   ;;
   self)
  )

(defclass dynamic_biped::srvClearPositionCMD
  :super ros::object
  :slots ())

(setf (get dynamic_biped::srvClearPositionCMD :md5sum-) "4414c67819626a1b8e0f043a9a0d6c9a")
(setf (get dynamic_biped::srvClearPositionCMD :datatype-) "dynamic_biped/srvClearPositionCMD")
(setf (get dynamic_biped::srvClearPositionCMD :request) dynamic_biped::srvClearPositionCMDRequest)
(setf (get dynamic_biped::srvClearPositionCMD :response) dynamic_biped::srvClearPositionCMDResponse)

(defmethod dynamic_biped::srvClearPositionCMDRequest
  (:response () (instance dynamic_biped::srvClearPositionCMDResponse :init)))

(setf (get dynamic_biped::srvClearPositionCMDRequest :md5sum-) "4414c67819626a1b8e0f043a9a0d6c9a")
(setf (get dynamic_biped::srvClearPositionCMDRequest :datatype-) "dynamic_biped/srvClearPositionCMDRequest")
(setf (get dynamic_biped::srvClearPositionCMDRequest :definition-)
      "---
int8 result

")

(setf (get dynamic_biped::srvClearPositionCMDResponse :md5sum-) "4414c67819626a1b8e0f043a9a0d6c9a")
(setf (get dynamic_biped::srvClearPositionCMDResponse :datatype-) "dynamic_biped/srvClearPositionCMDResponse")
(setf (get dynamic_biped::srvClearPositionCMDResponse :definition-)
      "---
int8 result

")



(provide :dynamic_biped/srvClearPositionCMD "4414c67819626a1b8e0f043a9a0d6c9a")


