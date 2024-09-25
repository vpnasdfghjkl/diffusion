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

class robotHandPosition {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.left_hand_position = null;
      this.right_hand_position = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('left_hand_position')) {
        this.left_hand_position = initObj.left_hand_position
      }
      else {
        this.left_hand_position = [];
      }
      if (initObj.hasOwnProperty('right_hand_position')) {
        this.right_hand_position = initObj.right_hand_position
      }
      else {
        this.right_hand_position = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type robotHandPosition
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [left_hand_position]
    bufferOffset = _arraySerializer.uint8(obj.left_hand_position, buffer, bufferOffset, null);
    // Serialize message field [right_hand_position]
    bufferOffset = _arraySerializer.uint8(obj.right_hand_position, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type robotHandPosition
    let len;
    let data = new robotHandPosition(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [left_hand_position]
    data.left_hand_position = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    // Deserialize message field [right_hand_position]
    data.right_hand_position = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += object.left_hand_position.length;
    length += object.right_hand_position.length;
    return length + 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/robotHandPosition';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ee45ac8dfb6e0c9a4c7712025fd69d4f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    uint8[] left_hand_position
    uint8[] right_hand_position
    
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
    const resolved = new robotHandPosition(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.left_hand_position !== undefined) {
      resolved.left_hand_position = msg.left_hand_position;
    }
    else {
      resolved.left_hand_position = []
    }

    if (msg.right_hand_position !== undefined) {
      resolved.right_hand_position = msg.right_hand_position;
    }
    else {
      resolved.right_hand_position = []
    }

    return resolved;
    }
};

module.exports = robotHandPosition;
