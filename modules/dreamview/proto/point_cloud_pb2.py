# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/dreamview/proto/point_cloud.proto

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
  name='modules/dreamview/proto/point_cloud.proto',
  package='apollo.dreamview',
  syntax='proto2',
  serialized_pb=_b('\n)modules/dreamview/proto/point_cloud.proto\x12\x10\x61pollo.dreamview\"\x1d\n\nPointCloud\x12\x0f\n\x03num\x18\x01 \x03(\x02\x42\x02\x10\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_POINTCLOUD = _descriptor.Descriptor(
  name='PointCloud',
  full_name='apollo.dreamview.PointCloud',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num', full_name='apollo.dreamview.PointCloud.num', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
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
  serialized_start=63,
  serialized_end=92,
)

DESCRIPTOR.message_types_by_name['PointCloud'] = _POINTCLOUD

PointCloud = _reflection.GeneratedProtocolMessageType('PointCloud', (_message.Message,), dict(
  DESCRIPTOR = _POINTCLOUD,
  __module__ = 'modules.dreamview.proto.point_cloud_pb2'
  # @@protoc_insertion_point(class_scope:apollo.dreamview.PointCloud)
  ))
_sym_db.RegisterMessage(PointCloud)


_POINTCLOUD.fields_by_name['num'].has_options = True
_POINTCLOUD.fields_by_name['num']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
