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

class handRotationEular {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.eulerAngles = null;
    }
    else {
      if (initObj.hasOwnProperty('eulerAngles')) {
        this.eulerAngles = initObj.eulerAngles
      }
      else {
        this.eulerAngles = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type handRotationEular
    // Serialize message field [eulerAngles]
    // Serialize the length for message field [eulerAngles]
    bufferOffset = _serializer.uint32(obj.eulerAngles.length, buffer, bufferOffset);
    obj.eulerAngles.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type handRotationEular
    let len;
    let data = new handRotationEular(null);
    // Deserialize message field [eulerAngles]
    // Deserialize array length for message field [eulerAngles]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.eulerAngles = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.eulerAngles[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 24 * object.eulerAngles.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/handRotationEular';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6f24fd3e9eed11c525f6da164f46e8b2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Vector3[] eulerAngles
    
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
    const resolved = new handRotationEular(null);
    if (msg.eulerAngles !== undefined) {
      resolved.eulerAngles = new Array(msg.eulerAngles.length);
      for (let i = 0; i < resolved.eulerAngles.length; ++i) {
        resolved.eulerAngles[i] = geometry_msgs.msg.Vector3.Resolve(msg.eulerAngles[i]);
      }
    }
    else {
      resolved.eulerAngles = []
    }

    return resolved;
    }
};

module.exports = handRotationEular;
