
(cl:in-package :asdf)

(defsystem "dynamic_biped-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "changeAMBACCtrlMode" :depends-on ("_package_changeAMBACCtrlMode"))
    (:file "_package_changeAMBACCtrlMode" :depends-on ("_package"))
    (:file "changeArmCtrlMode" :depends-on ("_package_changeArmCtrlMode"))
    (:file "_package_changeArmCtrlMode" :depends-on ("_package"))
    (:file "controlEndHand" :depends-on ("_package_controlEndHand"))
    (:file "_package_controlEndHand" :depends-on ("_package"))
    (:file "srvChangeJoller" :depends-on ("_package_srvChangeJoller"))
    (:file "_package_srvChangeJoller" :depends-on ("_package"))
    (:file "srvChangePhases" :depends-on ("_package_srvChangePhases"))
    (:file "_package_srvChangePhases" :depends-on ("_package"))
    (:file "srvClearPositionCMD" :depends-on ("_package_srvClearPositionCMD"))
    (:file "_package_srvClearPositionCMD" :depends-on ("_package"))
    (:file "srvManiInst" :depends-on ("_package_srvManiInst"))
    (:file "_package_srvManiInst" :depends-on ("_package"))
    (:file "srvchangeCtlMode" :depends-on ("_package_srvchangeCtlMode"))
    (:file "_package_srvchangeCtlMode" :depends-on ("_package"))
  ))