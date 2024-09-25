// Auto-generated. Do not edit!

// (in-package dynamic_biped.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class robotPhase {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.mainPhase = null;
      this.subPhase = null;
    }
    else {
      if (initObj.hasOwnProperty('mainPhase')) {
        this.mainPhase = initObj.mainPhase
      }
      else {
        this.mainPhase = 0;
      }
      if (initObj.hasOwnProperty('subPhase')) {
        this.subPhase = initObj.subPhase
      }
      else {
        this.subPhase = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type robotPhase
    // Serialize message field [mainPhase]
    bufferOffset = _serializer.uint8(obj.mainPhase, buffer, bufferOffset);
    // Serialize message field [subPhase]
    bufferOffset = _serializer.uint8(obj.subPhase, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type robotPhase
    let len;
    let data = new robotPhase(null);
    // Deserialize message field [mainPhase]
    data.mainPhase = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [subPhase]
    data.subPhase = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 2;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/robotPhase';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '26cd19545acfae4dd7729b92456b2c32';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 mainPhase
    uint8 subPhase
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new robotPhase(null);
    if (msg.mainPhase !== undefined) {
      resolved.mainPhase = msg.mainPhase;
    }
    else {
      resolved.mainPhase = 0
    }

    if (msg.subPhase !== undefined) {
      resolved.subPhase = msg.subPhase;
    }
    else {
      resolved.subPhase = 0
    }

    return resolved;
    }
};

module.exports = robotPhase;
