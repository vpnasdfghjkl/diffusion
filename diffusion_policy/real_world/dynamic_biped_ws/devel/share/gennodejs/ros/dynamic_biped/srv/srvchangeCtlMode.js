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

class srvchangeCtlModeRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.masterID = null;
      this.control_mode = null;
    }
    else {
      if (initObj.hasOwnProperty('masterID')) {
        this.masterID = initObj.masterID
      }
      else {
        this.masterID = 0;
      }
      if (initObj.hasOwnProperty('control_mode')) {
        this.control_mode = initObj.control_mode
      }
      else {
        this.control_mode = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvchangeCtlModeRequest
    // Serialize message field [masterID]
    bufferOffset = _serializer.uint8(obj.masterID, buffer, bufferOffset);
    // Serialize message field [control_mode]
    bufferOffset = _serializer.uint8(obj.control_mode, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvchangeCtlModeRequest
    let len;
    let data = new srvchangeCtlModeRequest(null);
    // Deserialize message field [masterID]
    data.masterID = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [control_mode]
    data.control_mode = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 2;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvchangeCtlModeRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4c7eca84f2e9c0cf24bd6d4f99e07c51';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 masterID
    uint8 control_mode
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvchangeCtlModeRequest(null);
    if (msg.masterID !== undefined) {
      resolved.masterID = msg.masterID;
    }
    else {
      resolved.masterID = 0
    }

    if (msg.control_mode !== undefined) {
      resolved.control_mode = msg.control_mode;
    }
    else {
      resolved.control_mode = 0
    }

    return resolved;
    }
};

class srvchangeCtlModeResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.control_mode = null;
    }
    else {
      if (initObj.hasOwnProperty('control_mode')) {
        this.control_mode = initObj.control_mode
      }
      else {
        this.control_mode = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvchangeCtlModeResponse
    // Serialize message field [control_mode]
    bufferOffset = _serializer.uint8(obj.control_mode, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvchangeCtlModeResponse
    let len;
    let data = new srvchangeCtlModeResponse(null);
    // Deserialize message field [control_mode]
    data.control_mode = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvchangeCtlModeResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '97960bebec978676e609deac154aaa6f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 control_mode
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvchangeCtlModeResponse(null);
    if (msg.control_mode !== undefined) {
      resolved.control_mode = msg.control_mode;
    }
    else {
      resolved.control_mode = 0
    }

    return resolved;
    }
};

module.exports = {
  Request: srvchangeCtlModeRequest,
  Response: srvchangeCtlModeResponse,
  md5sum() { return '55401bc31181d236576d88caa73b57c6'; },
  datatype() { return 'dynamic_biped/srvchangeCtlMode'; }
};
