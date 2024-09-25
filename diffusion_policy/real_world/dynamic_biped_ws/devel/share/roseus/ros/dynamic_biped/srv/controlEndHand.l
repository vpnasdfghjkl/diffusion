;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::controlEndHand)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'controlEndHand (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::CONTROLENDHAND")
  (make-package "DYNAMIC_BIPED::CONTROLENDHAND"))
(unless (find-package "DYNAMIC_BIPED::CONTROLENDHANDREQUEST")
  (make-package "DYNAMIC_BIPED::CONTROLENDHANDREQUEST"))
(unless (find-package "DYNAMIC_BIPED::CONTROLENDHANDRESPONSE")
  (make-package "DYNAMIC_BIPED::CONTROLENDHANDRESPONSE"))

(in-package "ROS")





(defclass dynamic_biped::controlEndHandRequest
  :super ros::object
  :slots (_left_hand_position _right_hand_position ))

(defmethod dynamic_biped::controlEndHandRequest
  (:init
   (&key
    ((:left_hand_position __left_hand_position) (make-array 0 :initial-element 0 :element-type :char))
    ((:right_hand_position __right_hand_position) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _left_hand_position __left_hand_position)
   (setq _right_hand_position __right_hand_position)
   self)
  (:left_hand_position
   (&optional __left_hand_position)
   (if __left_hand_position (setq _left_hand_position __left_hand_position)) _left_hand_position)
  (:right_hand_position
   (&optional __right_hand_position)
   (if __right_hand_position (setq _right_hand_position __right_hand_position)) _right_hand_position)
  (:serialization-length
   ()
   (+
    ;; uint8[] _left_hand_position
    (* 1    (length _left_hand_position)) 4
    ;; uint8[] _right_hand_position
    (* 1    (length _right_hand_position)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8[] _left_hand_position
     (write-long (length _left_hand_position) s)
     (princ _left_hand_position s)
     ;; uint8[] _right_hand_position
     (write-long (length _right_hand_position) s)
     (princ _right_hand_position s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8[] _left_hand_position
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _left_hand_position (make-array n :element-type :char))
     (replace _left_hand_position buf :start2 ptr-) (incf ptr- n))
   ;; uint8[] _right_hand_position
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _right_hand_position (make-array n :element-type :char))
     (replace _right_hand_position buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(defclass dynamic_biped::controlEndHandResponse
  :super ros::object
  :slots (_result ))

(defmethod dynamic_biped::controlEndHandResponse
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

(defclass dynamic_biped::controlEndHand
  :super ros::object
  :slots ())

(setf (get dynamic_biped::controlEndHand :md5sum-) "741989817a59889e258aa9e94c7ada8a")
(setf (get dynamic_biped::controlEndHand :datatype-) "dynamic_biped/controlEndHand")
(setf (get dynamic_biped::controlEndHand :request) dynamic_biped::controlEndHandRequest)
(setf (get dynamic_biped::controlEndHand :response) dynamic_biped::controlEndHandResponse)

(defmethod dynamic_biped::controlEndHandRequest
  (:response () (instance dynamic_biped::controlEndHandResponse :init)))

(setf (get dynamic_biped::controlEndHandRequest :md5sum-) "741989817a59889e258aa9e94c7ada8a")
(setf (get dynamic_biped::controlEndHandRequest :datatype-) "dynamic_biped/controlEndHandRequest")
(setf (get dynamic_biped::controlEndHandRequest :definition-)
      "uint8[] left_hand_position
uint8[] right_hand_position
---
bool result

")

(setf (get dynamic_biped::controlEndHandResponse :md5sum-) "741989817a59889e258aa9e94c7ada8a")
(setf (get dynamic_biped::controlEndHandResponse :datatype-) "dynamic_biped/controlEndHandResponse")
(setf (get dynamic_biped::controlEndHandResponse :definition-)
      "uint8[] left_hand_position
uint8[] right_hand_position
---
bool result

")



(provide :dynamic_biped/controlEndHand "741989817a59889e258aa9e94c7ada8a")


