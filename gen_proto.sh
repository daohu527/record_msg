#!/usr/bin/env bash

function build_py_proto() {
  if [ -d "./py_proto" ];then
    rm -rf py_proto
  fi
  mkdir py_proto
  find modules/ cyber/ -name "*.proto" \
      | grep -v -e node_modules -e canbus_vehicle \
      | xargs protoc --python_out=py_proto
  find py_proto/* -type d -exec touch "{}/__init__.py" \;
}

build_py_proto
