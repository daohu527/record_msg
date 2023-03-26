# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/common_msgs/sensor_msgs/ultrasonic_radar.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from modules.common_msgs.basic_msgs import header_pb2 as modules_dot_common__msgs_dot_basic__msgs_dot_header__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/common_msgs/sensor_msgs/ultrasonic_radar.proto',
  package='apollo.drivers',
  syntax='proto2',
  serialized_pb=_b('\n6modules/common_msgs/sensor_msgs/ultrasonic_radar.proto\x12\x0e\x61pollo.drivers\x1a+modules/common_msgs/basic_msgs/header.proto\"C\n\nUltrasonic\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12\x0e\n\x06ranges\x18\x02 \x03(\x02')
  ,
  dependencies=[modules_dot_common__msgs_dot_basic__msgs_dot_header__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_ULTRASONIC = _descriptor.Descriptor(
  name='Ultrasonic',
  full_name='apollo.drivers.Ultrasonic',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='apollo.drivers.Ultrasonic.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ranges', full_name='apollo.drivers.Ultrasonic.ranges', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=119,
  serialized_end=186,
)

_ULTRASONIC.fields_by_name['header'].message_type = modules_dot_common__msgs_dot_basic__msgs_dot_header__pb2._HEADER
DESCRIPTOR.message_types_by_name['Ultrasonic'] = _ULTRASONIC

Ultrasonic = _reflection.GeneratedProtocolMessageType('Ultrasonic', (_message.Message,), dict(
  DESCRIPTOR = _ULTRASONIC,
  __module__ = 'modules.common_msgs.sensor_msgs.ultrasonic_radar_pb2'
  # @@protoc_insertion_point(class_scope:apollo.drivers.Ultrasonic)
  ))
_sym_db.RegisterMessage(Ultrasonic)


# @@protoc_insertion_point(module_scope)