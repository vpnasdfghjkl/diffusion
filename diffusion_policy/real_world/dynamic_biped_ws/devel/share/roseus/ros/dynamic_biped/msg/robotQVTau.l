;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robotQVTau)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robotQVTau (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOTQVTAU")
  (make-package "DYNAMIC_BIPED::ROBOTQVTAU"))

(in-package "ROS")
;;//! \htmlinclude robotQVTau.msg.html


(defclass dynamic_biped::robotQVTau
  :super ros::object
  :slots (_q _v _tau ))

(defmethod dynamic_biped::robotQVTau
  (:init
   (&key
    ((:q __q) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:v __v) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:tau __tau) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _q __q)
   (setq _v __v)
   (setq _tau __tau)
   self)
  (:q
   (&optional __q)
   (if __q (setq _q __q)) _q)
  (:v
   (&optional __v)
   (if __v (setq _v __v)) _v)
  (:tau
   (&optional __tau)
   (if __tau (setq _tau __tau)) _tau)
  (:serialization-length
   ()
   (+
    ;; float64[] _q
    (* 8    (length _q)) 4
    ;; float64[] _v
    (* 8    (length _v)) 4
    ;; float64[] _tau
    (* 8    (length _tau)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64[] _q
     (write-long (length _q) s)
     (dotimes (i (length _q))
       (sys::poke (elt _q i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _v
     (write-long (length _v) s)
     (dotimes (i (length _v))
       (sys::poke (elt _v i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[] _tau
     (write-long (length _tau) s)
     (dotimes (i (length _tau))
       (sys::poke (elt _tau i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64[] _q
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _q (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _q i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _v
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _v (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _v i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64[] _tau
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _tau (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _tau i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get dynamic_biped::robotQVTau :md5sum-) "b3aa74a32b604340b47572dd2a0b70d4")
(setf (get dynamic_biped::robotQVTau :datatype-) "dynamic_biped/robotQVTau")
(setf (get dynamic_biped::robotQVTau :definition-)
      "float64[] q
float64[] v
float64[] tau
")



(provide :dynamic_biped/robotQVTau "b3aa74a32b604340b47572dd2a0b70d4")


