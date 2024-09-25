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

class QuaternionArray {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.quaternions = null;
    }
    else {
      if (initObj.hasOwnProperty('quaternions')) {
        this.quaternions = initObj.quaternions
      }
      else {
        this.quaternions = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QuaternionArray
    // Serialize message field [quaternions]
    // Serialize the length for message field [quaternions]
    bufferOffset = _serializer.uint32(obj.quaternions.length, buffer, bufferOffset);
    obj.quaternions.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Quaternion.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QuaternionArray
    let len;
    let data = new QuaternionArray(null);
    // Deserialize message field [quaternions]
    // Deserialize array length for message field [quaternions]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.quaternions = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.quaternions[i] = geometry_msgs.msg.Quaternion.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 32 * object.quaternions.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/QuaternionArray';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c666021c5a7330bd53d9827c2e91d9f6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Quaternion[] quaternions
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QuaternionArray(null);
    if (msg.quaternions !== undefined) {
      resolved.quaternions = new Array(msg.quaternions.length);
      for (let i = 0; i < resolved.quaternions.length; ++i) {
        resolved.quaternions[i] = geometry_msgs.msg.Quaternion.Resolve(msg.quaternions[i]);
      }
    }
    else {
      resolved.quaternions = []
    }

    return resolved;
    }
};

module.exports = QuaternionArray;
