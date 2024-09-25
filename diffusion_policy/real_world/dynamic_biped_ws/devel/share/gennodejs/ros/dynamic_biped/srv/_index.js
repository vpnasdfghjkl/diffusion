
"use strict";

let srvChangeJoller = require('./srvChangeJoller.js')
let srvchangeCtlMode = require('./srvchangeCtlMode.js')
let controlEndHand = require('./controlEndHand.js')
let srvManiInst = require('./srvManiInst.js')
let changeAMBACCtrlMode = require('./changeAMBACCtrlMode.js')
let srvClearPositionCMD = require('./srvClearPositionCMD.js')
let changeArmCtrlMode = require('./changeArmCtrlMode.js')
let srvChangePhases = require('./srvChangePhases.js')

module.exports = {
  srvChangeJoller: srvChangeJoller,
  srvchangeCtlMode: srvchangeCtlMode,
  controlEndHand: controlEndHand,
  srvManiInst: srvManiInst,
  changeAMBACCtrlMode: changeAMBACCtrlMode,
  srvClearPositionCMD: srvClearPositionCMD,
  changeArmCtrlMode: changeArmCtrlMode,
  srvChangePhases: srvChangePhases,
};
