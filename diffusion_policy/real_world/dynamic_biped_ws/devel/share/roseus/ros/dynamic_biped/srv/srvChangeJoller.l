;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::srvChangeJoller)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'srvChangeJoller (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGEJOLLER")
  (make-package "DYNAMIC_BIPED::SRVCHANGEJOLLER"))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGEJOLLERREQUEST")
  (make-package "DYNAMIC_BIPED::SRVCHANGEJOLLERREQUEST"))
(unless (find-package "DYNAMIC_BIPED::SRVCHANGEJOLLERRESPONSE")
  (make-package "DYNAMIC_BIPED::SRVCHANGEJOLLERRESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::srvChangeJollerRequest
  :super ros::object
  :slots (_l_pos _r_pos ))

(defmethod dynamic_biped::srvChangeJollerRequest
  (:init
   (&key
    ((:l_pos __l_pos) 0)
    ((:r_pos __r_pos) 0)
    )
   (send-super :init)
   (setq _l_pos (round __l_pos))
   (setq _r_pos (round __r_pos))
   self)
  (:l_pos
   (&optional __l_pos)
   (if __l_pos (setq _l_pos __l_pos)) _l_pos)
  (:r_pos
   (&optional __r_pos)
   (if __r_pos (setq _r_pos __r_pos)) _r_pos)
  (:serialization-length
   ()
   (+
    ;; int32 _l_pos
    4
    ;; int32 _r_pos
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _l_pos
       (write-long _l_pos s)
     ;; int32 _r_pos
       (write-long _r_pos s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _l_pos
     (setq _l_pos (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _r_pos
     (setq _r_pos (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(defclass dynamic_biped::srvChangeJollerResponse
  :super ros::object
  :slots (_result ))

(defmethod dynamic_biped::srvChangeJollerResponse
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

(defclass dynamic_biped::srvChangeJoller
  :super ros::object
  :slots ())

(setf (get dynamic_biped::srvChangeJoller :md5sum-) "9e5320c57603a9a2673e4a9b69a7b21e")
(setf (get dynamic_biped::srvChangeJoller :datatype-) "dynamic_biped/srvChangeJoller")
(setf (get dynamic_biped::srvChangeJoller :request) dynamic_biped::srvChangeJollerRequest)
(setf (get dynamic_biped::srvChangeJoller :response) dynamic_biped::srvChangeJollerResponse)

(defmethod dynamic_biped::srvChangeJollerRequest
  (:response () (instance dynamic_biped::srvChangeJollerResponse :init)))

(setf (get dynamic_biped::srvChangeJollerRequest :md5sum-) "9e5320c57603a9a2673e4a9b69a7b21e")
(setf (get dynamic_biped::srvChangeJollerRequest :datatype-) "dynamic_biped/srvChangeJollerRequest")
(setf (get dynamic_biped::srvChangeJollerRequest :definition-)
      "int32 l_pos
int32 r_pos
---
bool result
")

(setf (get dynamic_biped::srvChangeJollerResponse :md5sum-) "9e5320c57603a9a2673e4a9b69a7b21e")
(setf (get dynamic_biped::srvChangeJollerResponse :datatype-) "dynamic_biped/srvChangeJollerResponse")
(setf (get dynamic_biped::srvChangeJollerResponse :definition-)
      "int32 l_pos
int32 r_pos
---
bool result
")



(provide :dynamic_biped/srvChangeJoller "9e5320c57603a9a2673e4a9b69a7b21e")


