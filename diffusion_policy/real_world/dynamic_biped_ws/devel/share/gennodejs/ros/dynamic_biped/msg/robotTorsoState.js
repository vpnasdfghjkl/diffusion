// Auto-generated. Do not edit!

// (in-package dynamic_biped.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class robotTorsoState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.torsoR = null;
      this.torsoRd = null;
      this.torsoRdd = null;
      this.r = null;
      this.rd = null;
      this.rdd = null;
    }
    else {
      if (initObj.hasOwnProperty('torsoR')) {
        this.torsoR = initObj.torsoR
      }
      else {
        this.torsoR = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('torsoRd')) {
        this.torsoRd = initObj.torsoRd
      }
      else {
        this.torsoRd = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('torsoRdd')) {
        this.torsoRdd = initObj.torsoRdd
      }
      else {
        this.torsoRdd = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('r')) {
        this.r = initObj.r
      }
      else {
        this.r = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('rd')) {
        this.rd = initObj.rd
      }
      else {
        this.rd = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('rdd')) {
        this.rdd = initObj.rdd
      }
      else {
        this.rdd = new geometry_msgs.msg.Vector3();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type robotTorsoState
    // Serialize message field [torsoR]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.torsoR, buffer, bufferOffset);
    // Serialize message field [torsoRd]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.torsoRd, buffer, bufferOffset);
    // Serialize message field [torsoRdd]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.torsoRdd, buffer, bufferOffset);
    // Serialize message field [r]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.r, buffer, bufferOffset);
    // Serialize message field [rd]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rd, buffer, bufferOffset);
    // Serialize message field [rdd]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rdd, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type robotTorsoState
    let len;
    let data = new robotTorsoState(null);
    // Deserialize message field [torsoR]
    data.torsoR = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [torsoRd]
    data.torsoRd = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [torsoRdd]
    data.torsoRdd = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [r]
    data.r = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [rd]
    data.rd = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [rdd]
    data.rdd = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 144;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/robotTorsoState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0cabdc260bffa0da5323abf719a81f3e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Vector3 torsoR
    geometry_msgs/Vector3 torsoRd
    geometry_msgs/Vector3 torsoRdd
    geometry_msgs/Vector3 r
    geometry_msgs/Vector3 rd
    geometry_msgs/Vector3 rdd
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new robotTorsoState(null);
    if (msg.torsoR !== undefined) {
      resolved.torsoR = geometry_msgs.msg.Vector3.Resolve(msg.torsoR)
    }
    else {
      resolved.torsoR = new geometry_msgs.msg.Vector3()
    }

    if (msg.torsoRd !== undefined) {
      resolved.torsoRd = geometry_msgs.msg.Vector3.Resolve(msg.torsoRd)
    }
    else {
      resolved.torsoRd = new geometry_msgs.msg.Vector3()
    }

    if (msg.torsoRdd !== undefined) {
      resolved.torsoRdd = geometry_msgs.msg.Vector3.Resolve(msg.torsoRdd)
    }
    else {
      resolved.torsoRdd = new geometry_msgs.msg.Vector3()
    }

    if (msg.r !== undefined) {
      resolved.r = geometry_msgs.msg.Vector3.Resolve(msg.r)
    }
    else {
      resolved.r = new geometry_msgs.msg.Vector3()
    }

    if (msg.rd !== undefined) {
      resolved.rd = geometry_msgs.msg.Vector3.Resolve(msg.rd)
    }
    else {
      resolved.rd = new geometry_msgs.msg.Vector3()
    }

    if (msg.rdd !== undefined) {
      resolved.rdd = geometry_msgs.msg.Vector3.Resolve(msg.rdd)
    }
    else {
      resolved.rdd = new geometry_msgs.msg.Vector3()
    }

    return resolved;
    }
};

module.exports = robotTorsoState;
