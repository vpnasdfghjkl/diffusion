;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::srvManiInst)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'srvManiInst (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::SRVMANIINST")
  (make-package "DYNAMIC_BIPED::SRVMANIINST"))
(unless (find-package "DYNAMIC_BIPED::SRVMANIINSTREQUEST")
  (make-package "DYNAMIC_BIPED::SRVMANIINSTREQUEST"))
(unless (find-package "DYNAMIC_BIPED::SRVMANIINSTRESPONSE")
  (make-package "DYNAMIC_BIPED::SRVMANIINSTRESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::srvManiInstRequest
  :super ros::object
  :slots (_intruction ))

(defmethod dynamic_biped::srvManiInstRequest
  (:init
   (&key
    ((:intruction __intruction) "")
    )
   (send-super :init)
   (setq _intruction (string __intruction))
   self)
  (:intruction
   (&optional __intruction)
   (if __intruction (setq _intruction __intruction)) _intruction)
  (:serialization-length
   ()
   (+
    ;; string _intruction
    4 (length _intruction)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _intruction
       (write-long (length _intruction) s) (princ _intruction s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _intruction
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _intruction (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass dynamic_biped::srvManiInstResponse
  :super ros::object
  :slots (_stateRes ))

(defmethod dynamic_biped::srvManiInstResponse
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
    ;; int8 _stateRes
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _stateRes
       (write-byte _stateRes s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _stateRes
     (setq _stateRes (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _stateRes 127) (setq _stateRes (- _stateRes 256)))
   ;;
   self)
  )

(defclass dynamic_biped::srvManiInst
  :super ros::object
  :slots ())

(setf (get dynamic_biped::srvManiInst :md5sum-) "b4c6768ce40d4cdb4227714a7786bd04")
(setf (get dynamic_biped::srvManiInst :datatype-) "dynamic_biped/srvManiInst")
(setf (get dynamic_biped::srvManiInst :request) dynamic_biped::srvManiInstRequest)
(setf (get dynamic_biped::srvManiInst :response) dynamic_biped::srvManiInstResponse)

(defmethod dynamic_biped::srvManiInstRequest
  (:response () (instance dynamic_biped::srvManiInstResponse :init)))

(setf (get dynamic_biped::srvManiInstRequest :md5sum-) "b4c6768ce40d4cdb4227714a7786bd04")
(setf (get dynamic_biped::srvManiInstRequest :datatype-) "dynamic_biped/srvManiInstRequest")
(setf (get dynamic_biped::srvManiInstRequest :definition-)
      "string intruction
---
int8 stateRes

")

(setf (get dynamic_biped::srvManiInstResponse :md5sum-) "b4c6768ce40d4cdb4227714a7786bd04")
(setf (get dynamic_biped::srvManiInstResponse :datatype-) "dynamic_biped/srvManiInstResponse")
(setf (get dynamic_biped::srvManiInstResponse :definition-)
      "string intruction
---
int8 stateRes

")



(provide :dynamic_biped/srvManiInst "b4c6768ce40d4cdb4227714a7786bd04")


