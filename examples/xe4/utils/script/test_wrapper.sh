#!/bin/bash

EXECUTABLE=$(basename "$1")

if [[ "$EXECUTABLE" == *"cluster"* ]]; then
  export L0SIM_GRITS_AUBLOAD_OPTS="$L0SIM_GRITS_AUBLOAD_OPTS_CLUSTER"
fi

exec "$@"
