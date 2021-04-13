# mini-cpr

A basic Checkpoint/Restart Library written in C

## Building
`cd` to the source folder:
- ```mkdir build```
- ```cd build```
- ```cmake ..```
- ```make```

## Run the Tests
After building is complete the following tests can be run.
### Unit Tests: 
```./test-mini-cpr```
### Test App:
```./test-app```

Execution can be aborted by pressing `CTRL + C`. Start the app again to see that execution is restarted from a checkpoint.