# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/perception/pipeline/proto/stage/camera_detection_preprocessor_config.proto

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
  name='modules/perception/pipeline/proto/stage/camera_detection_preprocessor_config.proto',
  package='apollo.perception.camera',
  syntax='proto2',
  serialized_pb=_b('\nRmodules/perception/pipeline/proto/stage/camera_detection_preprocessor_config.proto\x12\x18\x61pollo.perception.camera\"#\n!CameraDetectionPreprocessorConfig')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_CAMERADETECTIONPREPROCESSORCONFIG = _descriptor.Descriptor(
  name='CameraDetectionPreprocessorConfig',
  full_name='apollo.perception.camera.CameraDetectionPreprocessorConfig',
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
  serialized_start=112,
  serialized_end=147,
)

DESCRIPTOR.message_types_by_name['CameraDetectionPreprocessorConfig'] = _CAMERADETECTIONPREPROCESSORCONFIG

CameraDetectionPreprocessorConfig = _reflection.GeneratedProtocolMessageType('CameraDetectionPreprocessorConfig', (_message.Message,), dict(
  DESCRIPTOR = _CAMERADETECTIONPREPROCESSORCONFIG,
  __module__ = 'modules.perception.pipeline.proto.stage.camera_detection_preprocessor_config_pb2'
  # @@protoc_insertion_point(class_scope:apollo.perception.camera.CameraDetectionPreprocessorConfig)
  ))
_sym_db.RegisterMessage(CameraDetectionPreprocessorConfig)


# @@protoc_insertion_point(module_scope)
