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

class srvManiInstRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.intruction = null;
    }
    else {
      if (initObj.hasOwnProperty('intruction')) {
        this.intruction = initObj.intruction
      }
      else {
        this.intruction = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type srvManiInstRequest
    // Serialize message field [intruction]
    bufferOffset = _serializer.string(obj.intruction, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvManiInstRequest
    let len;
    let data = new srvManiInstRequest(null);
    // Deserialize message field [intruction]
    data.intruction = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.intruction);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvManiInstRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd7df9e20c3db53dd50c5b9c0107f31a2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string intruction
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvManiInstRequest(null);
    if (msg.intruction !== undefined) {
      resolved.intruction = msg.intruction;
    }
    else {
      resolved.intruction = ''
    }

    return resolved;
    }
};

class srvManiInstResponse {
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
    // Serializes a message object of type srvManiInstResponse
    // Serialize message field [stateRes]
    bufferOffset = _serializer.int8(obj.stateRes, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type srvManiInstResponse
    let len;
    let data = new srvManiInstResponse(null);
    // Deserialize message field [stateRes]
    data.stateRes = _deserializer.int8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/srvManiInstResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '556abd838730602dd98183f793a0e26b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int8 stateRes
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new srvManiInstResponse(null);
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
  Request: srvManiInstRequest,
  Response: srvManiInstResponse,
  md5sum() { return 'b4c6768ce40d4cdb4227714a7786bd04'; },
  datatype() { return 'dynamic_biped/srvManiInst'; }
};
