// Auto-generated. Do not edit!

// (in-package dynamic_biped.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let armHandPose = require('./armHandPose.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class recordArmHandPose {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.left_pose = null;
      this.right_pose = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('left_pose')) {
        this.left_pose = initObj.left_pose
      }
      else {
        this.left_pose = new armHandPose();
      }
      if (initObj.hasOwnProperty('right_pose')) {
        this.right_pose = initObj.right_pose
      }
      else {
        this.right_pose = new armHandPose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type recordArmHandPose
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [left_pose]
    bufferOffset = armHandPose.serialize(obj.left_pose, buffer, bufferOffset);
    // Serialize message field [right_pose]
    bufferOffset = armHandPose.serialize(obj.right_pose, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type recordArmHandPose
    let len;
    let data = new recordArmHandPose(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [left_pose]
    data.left_pose = armHandPose.deserialize(buffer, bufferOffset);
    // Deserialize message field [right_pose]
    data.right_pose = armHandPose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 224;
  }

  static datatype() {
    // Returns string type for a message object
    return 'dynamic_biped/recordArmHandPose';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ebe76034e32233813bca8e9a0bd81c2c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    armHandPose  left_pose
    armHandPose  right_pose
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
    
    ================================================================================
    MSG: dynamic_biped/armHandPose
    float64[3] pos_xyz
    float64[4] quat_xyzw
    
    float64[7] joint_angles
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new recordArmHandPose(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.left_pose !== undefined) {
      resolved.left_pose = armHandPose.Resolve(msg.left_pose)
    }
    else {
      resolved.left_pose = new armHandPose()
    }

    if (msg.right_pose !== undefined) {
      resolved.right_pose = armHandPose.Resolve(msg.right_pose)
    }
    else {
      resolved.right_pose = new armHandPose()
    }

    return resolved;
    }
};

module.exports = recordArmHandPose;
