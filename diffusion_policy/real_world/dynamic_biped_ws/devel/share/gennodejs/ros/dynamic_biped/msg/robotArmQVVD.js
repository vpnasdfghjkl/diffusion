// Auto-generated. Do not edit!

// (in-package dynamic_biped.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class robotArmQVVD {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.q = null;
      this.v = null;
      this.vd = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('q')) {
        this.q = initObj.q
      }
      else {
        this.q = [];
      }
      if (initObj.hasOwnProperty('v')) {
        this.v = initObj.v
      }
      else {
        this.v = [];
      }
      if (initObj.hasOwnProperty('vd')) {
        this.vd = initObj.vd
      }
      else {
        this.vd = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type robotArmQVVD
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [q]
    bufferOffset = _arraySerializer.float64(obj.q, buffer, bufferOffset, null);
    // Serialize message field [v]
    bufferOffset = _arraySerializer.float64(obj.v, buffer, bufferOffset, null);
    // Serialize message field [vd]
    bufferOffset = _arraySerializer.float64(obj.vd, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type robotArmQVVD
    let len;
    let data = new robotArmQVVD(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [q]
    data.q = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [v]
    data.v = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [vd]
    data.vd = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 8 * object.q.length;
    length += 8 * object.v.length;
    length += 8 * object.vd.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/robotArmQVVD';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a7be9f5331e9207427b0c5c8ace7b977';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    float64[] q
    float64[] v
    float64[] vd
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new robotArmQVVD(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.q !== undefined) {
      resolved.q = msg.q;
    }
    else {
      resolved.q = []
    }

    if (msg.v !== undefined) {
      resolved.v = msg.v;
    }
    else {
      resolved.v = []
    }

    if (msg.vd !== undefined) {
      resolved.vd = msg.vd;
    }
    else {
      resolved.vd = []
    }

    return resolved;
    }
};

module.exports = robotArmQVVD;
