;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robotPhase)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robotPhase (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOTPHASE")
  (make-package "DYNAMIC_BIPED::ROBOTPHASE"))

(in-package "ROS")
;;//! \htmlinclude robotPhase.msg.html


(defclass dynamic_biped::robotPhase
  :super ros::object
  :slots (_mainPhase _subPhase ))

(defmethod dynamic_biped::robotPhase
  (:init
   (&key
    ((:mainPhase __mainPhase) 0)
    ((:subPhase __subPhase) 0)
    )
   (send-super :init)
   (setq _mainPhase (round __mainPhase))
   (setq _subPhase (round __subPhase))
   self)
  (:mainPhase
   (&optional __mainPhase)
   (if __mainPhase (setq _mainPhase __mainPhase)) _mainPhase)
  (:subPhase
   (&optional __subPhase)
   (if __subPhase (setq _subPhase __subPhase)) _subPhase)
  (:serialization-length
   ()
   (+
    ;; uint8 _mainPhase
    1
    ;; uint8 _subPhase
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _mainPhase
       (write-byte _mainPhase s)
     ;; uint8 _subPhase
       (write-byte _subPhase s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _mainPhase
     (setq _mainPhase (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _subPhase
     (setq _subPhase (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get dynamic_biped::robotPhase :md5sum-) "26cd19545acfae4dd7729b92456b2c32")
(setf (get dynamic_biped::robotPhase :datatype-) "dynamic_biped/robotPhase")
(setf (get dynamic_biped::robotPhase :definition-)
      "uint8 mainPhase
uint8 subPhase
")



(provide :dynamic_biped/robotPhase "26cd19545acfae4dd7729b92456b2c32")


