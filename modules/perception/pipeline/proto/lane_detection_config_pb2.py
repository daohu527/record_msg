# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/perception/pipeline/proto/lane_detection_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/perception/pipeline/proto/lane_detection_config.proto',
  package='apollo.perception.pipeline',
  syntax='proto2',
  serialized_pb=_b('\n=modules/perception/pipeline/proto/lane_detection_config.proto\x12\x1a\x61pollo.perception.pipeline\"\x15\n\x13LaneDetectionConfig')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_LANEDETECTIONCONFIG = _descriptor.Descriptor(
  name='LaneDetectionConfig',
  full_name='apollo.perception.pipeline.LaneDetectionConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=114,
)

DESCRIPTOR.message_types_by_name['LaneDetectionConfig'] = _LANEDETECTIONCONFIG

LaneDetectionConfig = _reflection.GeneratedProtocolMessageType('LaneDetectionConfig', (_message.Message,), dict(
  DESCRIPTOR = _LANEDETECTIONCONFIG,
  __module__ = 'modules.perception.pipeline.proto.lane_detection_config_pb2'
  # @@protoc_insertion_point(class_scope:apollo.perception.pipeline.LaneDetectionConfig)
  ))
_sym_db.RegisterMessage(LaneDetectionConfig)


# @@protoc_insertion_point(module_scope)