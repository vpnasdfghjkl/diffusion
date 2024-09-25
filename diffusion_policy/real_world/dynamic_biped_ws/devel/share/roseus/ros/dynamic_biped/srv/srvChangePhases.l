;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::srvChangePhases)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'srvChangePhases (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGEPHASES")
  (make-package "DYNAMIC_BIPED::SRVCHANGEPHASES"))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGEPHASESREQUEST")
  (make-package "DYNAMIC_BIPED::SRVCHANGEPHASESREQUEST"))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGEPHASESRESPONSE")
  (make-package "DYNAMIC_BIPED::SRVCHANGEPHASESRESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::srvChangePhasesRequest
  :super ros::object
  :slots (_masterID _stateReq _subState ))

(defmethod dynamic_biped::srvChangePhasesRequest
  (:init
   (&key
    ((:masterID __masterID) 0)
    ((:stateReq __stateReq) "")
    ((:subState __subState) "")
    )
   (send-super :init)
   (setq _masterID (round __masterID))
   (setq _stateReq (string __stateReq))
   (setq _subState (string __subState))
   self)
  (:masterID
   (&optional __masterID)
   (if __masterID (setq _masterID __masterID)) _masterID)
  (:stateReq
   (&optional __stateReq)
   (if __stateReq (setq _stateReq __stateReq)) _stateReq)
  (:subState
   (&optional __subState)
   (if __subState (setq _subState __subState)) _subState)
  (:serialization-length
   ()
   (+
    ;; uint8 _masterID
    1
    ;; string _stateReq
    4 (length _stateReq)
    ;; string _subState
    4 (length _subState)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _masterID
       (write-byte _masterID s)
     ;; string _stateReq
       (write-long (length _stateReq) s) (princ _stateReq s)
     ;; string _subState
       (write-long (length _subState) s) (princ _subState s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _masterID
     (setq _masterID (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; string _stateReq
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _stateReq (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _subState
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _subState (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass dynamic_biped::srvChangePhasesResponse
  :super ros::object
  :slots (_stateRes ))

(defmethod dynamic_biped::srvChangePhasesResponse
  (:init
   (&key
    ((:stateRes __stateRes) 0)
    )
   (send-super :init)
   (setq _stateRes (round __stateRes))
   self)
  (:stateRes
   (&optional __stateRes)
   (if __stateRes (setq _stateRes __stateRes)) _stateRes)
  (:serialization-length
   ()
   (+
    ;; int16 _stateRes
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int16 _stateRes
       (write-word _stateRes s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int16 _stateRes
     (setq _stateRes (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(defclass dynamic_biped::srvChangePhases
  :super ros::object
  :slots ())

(setf (get dynamic_biped::srvChangePhases :md5sum-) "f89a7d7f1ae7501d38174856aa23b24d")
(setf (get dynamic_biped::srvChangePhases :datatype-) "dynamic_biped/srvChangePhases")
(setf (get dynamic_biped::srvChangePhases :request) dynamic_biped::srvChangePhasesRequest)
(setf (get dynamic_biped::srvChangePhases :response) dynamic_biped::srvChangePhasesResponse)

(defmethod dynamic_biped::srvChangePhasesRequest
  (:response () (instance dynamic_biped::srvChangePhasesResponse :init)))

(setf (get dynamic_biped::srvChangePhasesRequest :md5sum-) "f89a7d7f1ae7501d38174856aa23b24d")
(setf (get dynamic_biped::srvChangePhasesRequest :datatype-) "dynamic_biped/srvChangePhasesRequest")
(setf (get dynamic_biped::srvChangePhasesRequest :definition-)
      "uint8 masterID
string stateReq
string subState
---
int16 stateRes

")

(setf (get dynamic_biped::srvChangePhasesResponse :md5sum-) "f89a7d7f1ae7501d38174856aa23b24d")
(setf (get dynamic_biped::srvChangePhasesResponse :datatype-) "dynamic_biped/srvChangePhasesResponse")
(setf (get dynamic_biped::srvChangePhasesResponse :definition-)
      "uint8 masterID
string stateReq
string subState
---
int16 stateRes

")



(provide :dynamic_biped/srvChangePhases "f89a7d7f1ae7501d38174856aa23b24d")


