#!/bin/bash

exe="$1"
np=$2
args="$3"
TYPEART_PATH_RT="$4"
TYPEART_PATH_INTERCEPT="$5"

log_file="$(basename $exe)_out.log"

if [ -f "$log_file" ]; then
    rm "$log_file"
fi

type_file="$(pwd)/types.yaml"

echo "Executing integration test: mpiexec -n $np $exe $args with typeart runtime=$TYPEART_PATH_RT and typeart intercept=$TYPEART_PATH_INTERCEPT inside folder: $(pwd)"

LD_PRELOAD="$TYPEART_PATH_RT/libtypeart-rt.so $TYPEART_PATH_INTERCEPT/libinterceptor-rt.so" TA_EXE_TARGET=$exe TA_TYPE_FILE=${type_file} mpiexec --oversubscribe -n $np $exe $args &> "$log_file"

app_result=$?

if [ $app_result -ne 0 ]; then
    echo "Application terminated with error code $app_result"
    exit 1
fi

if [ ! -f "$log_file" ]; then
    echo "Integration test broken - no log generated"
    exit 1
fi

if grep -q "R\[[0-9]*\]\[Error\]" "$log_file"; then
  echo "Integration test failed - for details, view $log_file"
  exit 1
fi
