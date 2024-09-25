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

class srvChangePhasesRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.masterID = null;
      this.stateReq = null;
      this.subState = null;
    }
    else {
      if (initObj.hasOwnProperty('masterID')) {
        this.masterID = initObj.masterID
      }
      else {
        this.masterID = 0;
      }
      if (initObj.hasOwnProperty('stateReq')) {
        this.stateReq = initObj.stateReq
      }
      else {
        this.stateReq = '';
      }
      if (initObj.hasOwnProperty('subState')) {
        this.subState = initObj.subState
      }
      else {
        this.subState = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvChangePhasesRequest
    // Serialize message field [masterID]
    bufferOffset = _serializer.uint8(obj.masterID, buffer, bufferOffset);
    // Serialize message field [stateReq]
    bufferOffset = _serializer.string(obj.stateReq, buffer, bufferOffset);
    // Serialize message field [subState]
    bufferOffset = _serializer.string(obj.subState, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvChangePhasesRequest
    let len;
    let data = new srvChangePhasesRequest(null);
    // Deserialize message field [masterID]
    data.masterID = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [stateReq]
    data.stateReq = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [subState]
    data.subState = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.stateReq);
    length += _getByteLength(object.subState);
    return length + 9;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvChangePhasesRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1ca8c109297a3bc861ef8e95237d0baf';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 masterID
    string stateReq
    string subState
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvChangePhasesRequest(null);
    if (msg.masterID !== undefined) {
      resolved.masterID = msg.masterID;
    }
    else {
      resolved.masterID = 0
    }

    if (msg.stateReq !== undefined) {
      resolved.stateReq = msg.stateReq;
    }
    else {
      resolved.stateReq = ''
    }

    if (msg.subState !== undefined) {
      resolved.subState = msg.subState;
    }
    else {
      resolved.subState = ''
    }

    return resolved;
    }
};

class srvChangePhasesResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stateRes = null;
    }
    else {
      if (initObj.hasOwnProperty('stateRes')) {
        this.stateRes = initObj.stateRes
      }
      else {
        this.stateRes = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvChangePhasesResponse
    // Serialize message field [stateRes]
    bufferOffset = _serializer.int16(obj.stateRes, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvChangePhasesResponse
    let len;
    let data = new srvChangePhasesResponse(null);
    // Deserialize message field [stateRes]
    data.stateRes = _deserializer.int16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 2;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvChangePhasesResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7033fb042cc98da45f99da4d28b07c65';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int16 stateRes
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvChangePhasesResponse(null);
    if (msg.stateRes !== undefined) {
      resolved.stateRes = msg.stateRes;
    }
    else {
      resolved.stateRes = 0
    }

    return resolved;
    }
};

module.exports = {
  Request: srvChangePhasesRequest,
  Response: srvChangePhasesResponse,
  md5sum() { return 'f89a7d7f1ae7501d38174856aa23b24d'; },
  datatype() { return 'dynamic_biped/srvChangePhases'; }
};
