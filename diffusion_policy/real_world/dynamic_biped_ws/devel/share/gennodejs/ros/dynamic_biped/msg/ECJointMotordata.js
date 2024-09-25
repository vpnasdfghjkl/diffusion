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

class ECJointMotordata {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.l_pos_pitch = null;
      this.l_pos_yaw = null;
      this.l_pos_end = null;
      this.r_pos_pitch = null;
      this.r_pos_yaw = null;
      this.r_pos_end = null;
    }
    else {
      if (initObj.hasOwnProperty('l_pos_pitch')) {
        this.l_pos_pitch = initObj.l_pos_pitch
      }
      else {
        this.l_pos_pitch = 0;
      }
      if (initObj.hasOwnProperty('l_pos_yaw')) {
        this.l_pos_yaw = initObj.l_pos_yaw
      }
      else {
        this.l_pos_yaw = 0;
      }
      if (initObj.hasOwnProperty('l_pos_end')) {
        this.l_pos_end = initObj.l_pos_end
      }
      else {
        this.l_pos_end = 0;
      }
      if (initObj.hasOwnProperty('r_pos_pitch')) {
        this.r_pos_pitch = initObj.r_pos_pitch
      }
      else {
        this.r_pos_pitch = 0;
      }
      if (initObj.hasOwnProperty('r_pos_yaw')) {
        this.r_pos_yaw = initObj.r_pos_yaw
      }
      else {
        this.r_pos_yaw = 0;
      }
      if (initObj.hasOwnProperty('r_pos_end')) {
        this.r_pos_end = initObj.r_pos_end
      }
      else {
        this.r_pos_end = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ECJointMotordata
    // Serialize message field [l_pos_pitch]
    bufferOffset = _serializer.int32(obj.l_pos_pitch, buffer, bufferOffset);
    // Serialize message field [l_pos_yaw]
    bufferOffset = _serializer.int32(obj.l_pos_yaw, buffer, bufferOffset);
    // Serialize message field [l_pos_end]
    bufferOffset = _serializer.int32(obj.l_pos_end, buffer, bufferOffset);
    // Serialize message field [r_pos_pitch]
    bufferOffset = _serializer.int32(obj.r_pos_pitch, buffer, bufferOffset);
    // Serialize message field [r_pos_yaw]
    bufferOffset = _serializer.int32(obj.r_pos_yaw, buffer, bufferOffset);
    // Serialize message field [r_pos_end]
    bufferOffset = _serializer.int32(obj.r_pos_end, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ECJointMotordata
    let len;
    let data = new ECJointMotordata(null);
    // Deserialize message field [l_pos_pitch]
    data.l_pos_pitch = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [l_pos_yaw]
    data.l_pos_yaw = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [l_pos_end]
    data.l_pos_end = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [r_pos_pitch]
    data.r_pos_pitch = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [r_pos_yaw]
    data.r_pos_yaw = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [r_pos_end]
    data.r_pos_end = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/ECJointMotordata';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'da126c0fb22352a555bc24cbdebe8a85';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 l_pos_pitch 
    int32 l_pos_yaw
    int32 l_pos_end
    int32 r_pos_pitch 
    int32 r_pos_yaw
    int32 r_pos_end
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ECJointMotordata(null);
    if (msg.l_pos_pitch !== undefined) {
      resolved.l_pos_pitch = msg.l_pos_pitch;
    }
    else {
      resolved.l_pos_pitch = 0
    }

    if (msg.l_pos_yaw !== undefined) {
      resolved.l_pos_yaw = msg.l_pos_yaw;
    }
    else {
      resolved.l_pos_yaw = 0
    }

    if (msg.l_pos_end !== undefined) {
      resolved.l_pos_end = msg.l_pos_end;
    }
    else {
      resolved.l_pos_end = 0
    }

    if (msg.r_pos_pitch !== undefined) {
      resolved.r_pos_pitch = msg.r_pos_pitch;
    }
    else {
      resolved.r_pos_pitch = 0
    }

    if (msg.r_pos_yaw !== undefined) {
      resolved.r_pos_yaw = msg.r_pos_yaw;
    }
    else {
      resolved.r_pos_yaw = 0
    }

    if (msg.r_pos_end !== undefined) {
      resolved.r_pos_end = msg.r_pos_end;
    }
    else {
      resolved.r_pos_end = 0
    }

    return resolved;
    }
};

module.exports = ECJointMotordata;
