;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::walkCommand)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'walkCommand (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::WALKCOMMAND")
  (make-package "DYNAMIC_BIPED::WALKCOMMAND"))

(in-package "ROS")
;;//! \htmlinclude walkCommand.msg.html


(defclass dynamic_biped::walkCommand
  :super ros::object
  :slots (_mode _values ))

(defmethod dynamic_biped::walkCommand
  (:init
   (&key
    ((:mode __mode) 0)
    ((:values __values) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _mode (round __mode))
   (setq _values __values)
   self)
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:values
   (&optional __values)
   (if __values (setq _values __values)) _values)
  (:serialization-length
   ()
   (+
    ;; uint8 _mode
    1
    ;; float64[] _values
    (* 8    (length _values)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _mode
       (write-byte _mode s)
     ;; float64[] _values
     (write-long (length _values) s)
     (dotimes (i (length _values))
       (sys::poke (elt _values i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _mode
     (setq _mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float64[] _values
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _values (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _values i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get dynamic_biped::walkCommand :md5sum-) "5b32312c0e7618ca6c317ce9272f2ca7")
(setf (get dynamic_biped::walkCommand :datatype-) "dynamic_biped/walkCommand")
(setf (get dynamic_biped::walkCommand :definition-)
      "# mode: 0->PositionCommand | 1->VelocityCommand
uint8 mode
float64[] values

")



(provide :dynamic_biped/walkCommand "5b32312c0e7618ca6c317ce9272f2ca7")


