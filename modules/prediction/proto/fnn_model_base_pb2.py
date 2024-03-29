# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/prediction/proto/fnn_model_base.proto

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
  name='modules/prediction/proto/fnn_model_base.proto',
  package='apollo.prediction',
  syntax='proto2',
  serialized_pb=_b('\n-modules/prediction/proto/fnn_model_base.proto\x12\x11\x61pollo.prediction\"\x19\n\x06Vector\x12\x0f\n\x07\x63olumns\x18\x01 \x03(\x01\"1\n\x06Matrix\x12\'\n\x04rows\x18\x01 \x03(\x0b\x32\x19.apollo.prediction.Vector\"\xa8\x02\n\x05Layer\x12\x17\n\x0flayer_input_dim\x18\x01 \x01(\x05\x12\x18\n\x10layer_output_dim\x18\x02 \x01(\x05\x12\x35\n\x12layer_input_weight\x18\x03 \x01(\x0b\x32\x19.apollo.prediction.Matrix\x12-\n\nlayer_bias\x18\x04 \x01(\x0b\x32\x19.apollo.prediction.Vector\x12\x46\n\x15layer_activation_func\x18\x05 \x01(\x0e\x32\'.apollo.prediction.Layer.ActivationFunc\">\n\x0e\x41\x63tivationFunc\x12\x08\n\x04RELU\x10\x00\x12\x08\n\x04TANH\x10\x01\x12\x0b\n\x07SIGMOID\x10\x02\x12\x0b\n\x07SOFTMAX\x10\x03')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_LAYER_ACTIVATIONFUNC = _descriptor.EnumDescriptor(
  name='ActivationFunc',
  full_name='apollo.prediction.Layer.ActivationFunc',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RELU', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TANH', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SOFTMAX', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=381,
  serialized_end=443,
)
_sym_db.RegisterEnumDescriptor(_LAYER_ACTIVATIONFUNC)


_VECTOR = _descriptor.Descriptor(
  name='Vector',
  full_name='apollo.prediction.Vector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='columns', full_name='apollo.prediction.Vector.columns', index=0,
      number=1, type=1, cpp_type=5, label=3,
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
  serialized_start=68,
  serialized_end=93,
)


_MATRIX = _descriptor.Descriptor(
  name='Matrix',
  full_name='apollo.prediction.Matrix',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rows', full_name='apollo.prediction.Matrix.rows', index=0,
      number=1, type=11, cpp_type=10, label=3,
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
  serialized_start=95,
  serialized_end=144,
)


_LAYER = _descriptor.Descriptor(
  name='Layer',
  full_name='apollo.prediction.Layer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='layer_input_dim', full_name='apollo.prediction.Layer.layer_input_dim', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='layer_output_dim', full_name='apollo.prediction.Layer.layer_output_dim', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='layer_input_weight', full_name='apollo.prediction.Layer.layer_input_weight', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='layer_bias', full_name='apollo.prediction.Layer.layer_bias', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='layer_activation_func', full_name='apollo.prediction.Layer.layer_activation_func', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LAYER_ACTIVATIONFUNC,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=147,
  serialized_end=443,
)

_MATRIX.fields_by_name['rows'].message_type = _VECTOR
_LAYER.fields_by_name['layer_input_weight'].message_type = _MATRIX
_LAYER.fields_by_name['layer_bias'].message_type = _VECTOR
_LAYER.fields_by_name['layer_activation_func'].enum_type = _LAYER_ACTIVATIONFUNC
_LAYER_ACTIVATIONFUNC.containing_type = _LAYER
DESCRIPTOR.message_types_by_name['Vector'] = _VECTOR
DESCRIPTOR.message_types_by_name['Matrix'] = _MATRIX
DESCRIPTOR.message_types_by_name['Layer'] = _LAYER

Vector = _reflection.GeneratedProtocolMessageType('Vector', (_message.Message,), dict(
  DESCRIPTOR = _VECTOR,
  __module__ = 'modules.prediction.proto.fnn_model_base_pb2'
  # @@protoc_insertion_point(class_scope:apollo.prediction.Vector)
  ))
_sym_db.RegisterMessage(Vector)

Matrix = _reflection.GeneratedProtocolMessageType('Matrix', (_message.Message,), dict(
  DESCRIPTOR = _MATRIX,
  __module__ = 'modules.prediction.proto.fnn_model_base_pb2'
  # @@protoc_insertion_point(class_scope:apollo.prediction.Matrix)
  ))
_sym_db.RegisterMessage(Matrix)

Layer = _reflection.GeneratedProtocolMessageType('Layer', (_message.Message,), dict(
  DESCRIPTOR = _LAYER,
  __module__ = 'modules.prediction.proto.fnn_model_base_pb2'
  # @@protoc_insertion_point(class_scope:apollo.prediction.Layer)
  ))
_sym_db.RegisterMessage(Layer)


# @@protoc_insertion_point(module_scope)
