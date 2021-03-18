#!/usr/bin/env bash
tests_noerr=(01_noerr 02_noerr 03_noerr)
tests_err=(10_err 11_err 12_err)

echo "This script assumes bash"

failures=0

for i in ${tests_noerr[@]}; do
	echo "Running $i"
	mpirun -np 2 ./$i &> /dev/null
	if [ $? -ne 0 ]; then
		failures=$(($failures+1))
	fi
done

for i in ${tests_err[@]}; do
	echo "Running $i"
	mpirun -np 2 ./$i &>> log.err
	if [ $? -eq 0 ]; then
		failures=$(($failures+1))
	fi
	fatals=$(cat log.err | grep -i "Fatal" | wc -l)
done

echo "Test fatals: " $fatals
echo "Test fails: " $(($fatals+$failures))
echo "|------------------------"
echo "Test FAILURES: " $failures
echo "|------------------------"

exit $failures
