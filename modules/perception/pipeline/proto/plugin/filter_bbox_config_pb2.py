# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/perception/pipeline/proto/plugin/filter_bbox_config.proto

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
  name='modules/perception/pipeline/proto/plugin/filter_bbox_config.proto',
  package='apollo.perception.camera',
  syntax='proto2',
  serialized_pb=_b('\nAmodules/perception/pipeline/proto/plugin/filter_bbox_config.proto\x12\x18\x61pollo.perception.camera\"m\n\x10\x46ilterBboxConfig\x12\x15\n\rmin_2d_height\x18\x01 \x01(\x02\x12\x15\n\rmin_3d_height\x18\x02 \x01(\x02\x12\x15\n\rmin_3d_length\x18\x03 \x01(\x02\x12\x14\n\x0cmin_3d_width\x18\x04 \x01(\x02')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_FILTERBBOXCONFIG = _descriptor.Descriptor(
  name='FilterBboxConfig',
  full_name='apollo.perception.camera.FilterBboxConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_2d_height', full_name='apollo.perception.camera.FilterBboxConfig.min_2d_height', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='min_3d_height', full_name='apollo.perception.camera.FilterBboxConfig.min_3d_height', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='min_3d_length', full_name='apollo.perception.camera.FilterBboxConfig.min_3d_length', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='min_3d_width', full_name='apollo.perception.camera.FilterBboxConfig.min_3d_width', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
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
  serialized_start=95,
  serialized_end=204,
)

DESCRIPTOR.message_types_by_name['FilterBboxConfig'] = _FILTERBBOXCONFIG

FilterBboxConfig = _reflection.GeneratedProtocolMessageType('FilterBboxConfig', (_message.Message,), dict(
  DESCRIPTOR = _FILTERBBOXCONFIG,
  __module__ = 'modules.perception.pipeline.proto.plugin.filter_bbox_config_pb2'
  # @@protoc_insertion_point(class_scope:apollo.perception.camera.FilterBboxConfig)
  ))
_sym_db.RegisterMessage(FilterBboxConfig)


# @@protoc_insertion_point(module_scope)