// Auto-generated. Do not edit!

// (in-package dynamic_biped.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class srvChangeJollerRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.l_pos = null;
      this.r_pos = null;
    }
    else {
      if (initObj.hasOwnProperty('l_pos')) {
        this.l_pos = initObj.l_pos
      }
      else {
        this.l_pos = 0;
      }
      if (initObj.hasOwnProperty('r_pos')) {
        this.r_pos = initObj.r_pos
      }
      else {
        this.r_pos = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvChangeJollerRequest
    // Serialize message field [l_pos]
    bufferOffset = _serializer.int32(obj.l_pos, buffer, bufferOffset);
    // Serialize message field [r_pos]
    bufferOffset = _serializer.int32(obj.r_pos, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvChangeJollerRequest
    let len;
    let data = new srvChangeJollerRequest(null);
    // Deserialize message field [l_pos]
    data.l_pos = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [r_pos]
    data.r_pos = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvChangeJollerRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '74a6ed9b2ff5e74bf7cf4110500f1701';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 l_pos
    int32 r_pos
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvChangeJollerRequest(null);
    if (msg.l_pos !== undefined) {
      resolved.l_pos = msg.l_pos;
    }
    else {
      resolved.l_pos = 0
    }

    if (msg.r_pos !== undefined) {
      resolved.r_pos = msg.r_pos;
    }
    else {
      resolved.r_pos = 0
    }

    return resolved;
    }
};

class srvChangeJollerResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.result = null;
    }
    else {
      if (initObj.hasOwnProperty('result')) {
        this.result = initObj.result
      }
      else {
        this.result = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvChangeJollerResponse
    // Serialize message field [result]
    bufferOffset = _serializer.bool(obj.result, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvChangeJollerResponse
    let len;
    let data = new srvChangeJollerResponse(null);
    // Deserialize message field [result]
    data.result = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvChangeJollerResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eb13ac1f1354ccecb7941ee8fa2192e8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool result
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvChangeJollerResponse(null);
    if (msg.result !== undefined) {
      resolved.result = msg.result;
    }
    else {
      resolved.result = false
    }

    return resolved;
    }
};

module.exports = {
  Request: srvChangeJollerRequest,
  Response: srvChangeJollerResponse,
  md5sum() { return '9e5320c57603a9a2673e4a9b69a7b21e'; },
  datatype() { return 'dynamic_biped/srvChangeJoller'; }
};
