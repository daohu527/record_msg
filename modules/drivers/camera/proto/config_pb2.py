# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/drivers/camera/proto/config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/drivers/camera/proto/config.proto',
  package='apollo.drivers.camera.config',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n)modules/drivers/camera/proto/config.proto\x12\x1c\x61pollo.drivers.camera.config\"\xd3\x06\n\x06\x43onfig\x12\x12\n\ncamera_dev\x18\x01 \x01(\t\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\t\x12\x1a\n\x0cpixel_format\x18\x03 \x01(\t:\x04yuyv\x12\x39\n\tio_method\x18\x04 \x01(\x0e\x32&.apollo.drivers.camera.config.IOMethod\x12\r\n\x05width\x18\x05 \x01(\r\x12\x0e\n\x06height\x18\x06 \x01(\r\x12\x12\n\nframe_rate\x18\x07 \x01(\r\x12\x19\n\nmonochrome\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x16\n\nbrightness\x18\t \x01(\x05:\x02-1\x12\x14\n\x08\x63ontrast\x18\n \x01(\x05:\x02-1\x12\x16\n\nsaturation\x18\x0b \x01(\x05:\x02-1\x12\x15\n\tsharpness\x18\x0c \x01(\x05:\x02-1\x12\x10\n\x04gain\x18\r \x01(\x05:\x02-1\x12\x19\n\nauto_focus\x18\x0e \x01(\x08:\x05\x66\x61lse\x12\x11\n\x05\x66ocus\x18\x0f \x01(\x05:\x02-1\x12\x1b\n\rauto_exposure\x18\x10 \x01(\x08:\x04true\x12\x15\n\x08\x65xposure\x18\x11 \x01(\x05:\x03\x31\x30\x30\x12 \n\x12\x61uto_white_balance\x18\x12 \x01(\x08:\x04true\x12\x1b\n\rwhite_balance\x18\x13 \x01(\x05:\x04\x34\x30\x30\x30\x12\x1a\n\x0f\x62ytes_per_pixel\x18\x14 \x01(\r:\x01\x33\x12\x1b\n\x10trigger_internal\x18\x15 \x01(\r:\x01\x30\x12\x17\n\x0btrigger_fps\x18\x16 \x01(\r:\x02\x33\x30\x12\x14\n\x0c\x63hannel_name\x18\x17 \x01(\t\x12\x1c\n\x0e\x64\x65vice_wait_ms\x18\x18 \x01(\r:\x04\x32\x30\x30\x30\x12\x16\n\tspin_rate\x18\x19 \x01(\r:\x03\x32\x30\x30\x12=\n\x0boutput_type\x18\x1a \x01(\x0e\x32(.apollo.drivers.camera.config.OutputType\x12J\n\rcompress_conf\x18\x1b \x01(\x0b\x32\x33.apollo.drivers.camera.config.Config.CompressConfig\x1a\x45\n\x0e\x43ompressConfig\x12\x16\n\x0eoutput_channel\x18\x01 \x01(\t\x12\x1b\n\x0fimage_pool_size\x18\x02 \x01(\r:\x02\x32\x30*`\n\x08IOMethod\x12\x15\n\x11IO_METHOD_UNKNOWN\x10\x00\x12\x12\n\x0eIO_METHOD_READ\x10\x01\x12\x12\n\x0eIO_METHOD_MMAP\x10\x02\x12\x15\n\x11IO_METHOD_USERPTR\x10\x03*\x1f\n\nOutputType\x12\x08\n\x04YUYV\x10\x00\x12\x07\n\x03RGB\x10\x01'
)

_IOMETHOD = _descriptor.EnumDescriptor(
  name='IOMethod',
  full_name='apollo.drivers.camera.config.IOMethod',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='IO_METHOD_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='IO_METHOD_READ', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='IO_METHOD_MMAP', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='IO_METHOD_USERPTR', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=929,
  serialized_end=1025,
)
_sym_db.RegisterEnumDescriptor(_IOMETHOD)

IOMethod = enum_type_wrapper.EnumTypeWrapper(_IOMETHOD)
_OUTPUTTYPE = _descriptor.EnumDescriptor(
  name='OutputType',
  full_name='apollo.drivers.camera.config.OutputType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='YUYV', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RGB', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1027,
  serialized_end=1058,
)
_sym_db.RegisterEnumDescriptor(_OUTPUTTYPE)

OutputType = enum_type_wrapper.EnumTypeWrapper(_OUTPUTTYPE)
IO_METHOD_UNKNOWN = 0
IO_METHOD_READ = 1
IO_METHOD_MMAP = 2
IO_METHOD_USERPTR = 3
YUYV = 0
RGB = 1



_CONFIG_COMPRESSCONFIG = _descriptor.Descriptor(
  name='CompressConfig',
  full_name='apollo.drivers.camera.config.Config.CompressConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_channel', full_name='apollo.drivers.camera.config.Config.CompressConfig.output_channel', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_pool_size', full_name='apollo.drivers.camera.config.Config.CompressConfig.image_pool_size', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=858,
  serialized_end=927,
)

_CONFIG = _descriptor.Descriptor(
  name='Config',
  full_name='apollo.drivers.camera.config.Config',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='camera_dev', full_name='apollo.drivers.camera.config.Config.camera_dev', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='apollo.drivers.camera.config.Config.frame_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pixel_format', full_name='apollo.drivers.camera.config.Config.pixel_format', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"yuyv".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='io_method', full_name='apollo.drivers.camera.config.Config.io_method', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='apollo.drivers.camera.config.Config.width', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='apollo.drivers.camera.config.Config.height', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='frame_rate', full_name='apollo.drivers.camera.config.Config.frame_rate', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='monochrome', full_name='apollo.drivers.camera.config.Config.monochrome', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='brightness', full_name='apollo.drivers.camera.config.Config.brightness', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='contrast', full_name='apollo.drivers.camera.config.Config.contrast', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='saturation', full_name='apollo.drivers.camera.config.Config.saturation', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sharpness', full_name='apollo.drivers.camera.config.Config.sharpness', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gain', full_name='apollo.drivers.camera.config.Config.gain', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='auto_focus', full_name='apollo.drivers.camera.config.Config.auto_focus', index=13,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='focus', full_name='apollo.drivers.camera.config.Config.focus', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='auto_exposure', full_name='apollo.drivers.camera.config.Config.auto_exposure', index=15,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='exposure', full_name='apollo.drivers.camera.config.Config.exposure', index=16,
      number=17, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='auto_white_balance', full_name='apollo.drivers.camera.config.Config.auto_white_balance', index=17,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='white_balance', full_name='apollo.drivers.camera.config.Config.white_balance', index=18,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=4000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bytes_per_pixel', full_name='apollo.drivers.camera.config.Config.bytes_per_pixel', index=19,
      number=20, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trigger_internal', full_name='apollo.drivers.camera.config.Config.trigger_internal', index=20,
      number=21, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trigger_fps', full_name='apollo.drivers.camera.config.Config.trigger_fps', index=21,
      number=22, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=30,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='channel_name', full_name='apollo.drivers.camera.config.Config.channel_name', index=22,
      number=23, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='device_wait_ms', full_name='apollo.drivers.camera.config.Config.device_wait_ms', index=23,
      number=24, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=2000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='spin_rate', full_name='apollo.drivers.camera.config.Config.spin_rate', index=24,
      number=25, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=200,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output_type', full_name='apollo.drivers.camera.config.Config.output_type', index=25,
      number=26, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='compress_conf', full_name='apollo.drivers.camera.config.Config.compress_conf', index=26,
      number=27, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_CONFIG_COMPRESSCONFIG, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=76,
  serialized_end=927,
)

_CONFIG_COMPRESSCONFIG.containing_type = _CONFIG
_CONFIG.fields_by_name['io_method'].enum_type = _IOMETHOD
_CONFIG.fields_by_name['output_type'].enum_type = _OUTPUTTYPE
_CONFIG.fields_by_name['compress_conf'].message_type = _CONFIG_COMPRESSCONFIG
DESCRIPTOR.message_types_by_name['Config'] = _CONFIG
DESCRIPTOR.enum_types_by_name['IOMethod'] = _IOMETHOD
DESCRIPTOR.enum_types_by_name['OutputType'] = _OUTPUTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Config = _reflection.GeneratedProtocolMessageType('Config', (_message.Message,), {

  'CompressConfig' : _reflection.GeneratedProtocolMessageType('CompressConfig', (_message.Message,), {
    'DESCRIPTOR' : _CONFIG_COMPRESSCONFIG,
    '__module__' : 'modules.drivers.camera.proto.config_pb2'
    # @@protoc_insertion_point(class_scope:apollo.drivers.camera.config.Config.CompressConfig)
    })
  ,
  'DESCRIPTOR' : _CONFIG,
  '__module__' : 'modules.drivers.camera.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:apollo.drivers.camera.config.Config)
  })
_sym_db.RegisterMessage(Config)
_sym_db.RegisterMessage(Config.CompressConfig)


# @@protoc_insertion_point(module_scope)