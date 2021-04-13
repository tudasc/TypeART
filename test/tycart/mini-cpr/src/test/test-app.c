//
// Created by mority on 11/22/19.
//

#include "stdlib.h"
#include "unistd.h"

#include "include/unity.h"
#include "../mini-cpr.h"

#define CFG_FILE_PATH "./assets/test.config"
#define DATA_SIZE 1024
#define MAX_ITERATIONS 10

// run before and after each test, unity requires these to be defined
void setUp(void){}
void tearDown(void){}

void integration_test(void) {
    // initialize mini-cpr
    TEST_ASSERT_EQUAL_INT(0,mini_cpr_init(CFG_FILE_PATH));

    // initialize app data
    int * data[DATA_SIZE];
    for(int i = 0; i < DATA_SIZE; ++i) {

        // allocate space for app data
        data[i] = malloc((DATA_SIZE - i) * sizeof(int));

        // register allocated memory regions with mini-cpr
        TEST_ASSERT_EQUAL_INT(0, mini_cpr_register(i, data[i], DATA_SIZE - i, sizeof(int)));
    }

    // is this a restart?
    int version = mini_cpr_restart_check("integration_test", MAX_ITERATIONS);

    if(version == -1) {
        // no checkpoint found, start loop in iteration = 0
        printf("No Checkpoints found, starting loop in iteration = 0\n");
        version = 0;
    } else {
        printf("Restarting from Checkpoint %d\n", version);
        // load checkpoint version found
        TEST_ASSERT_EQUAL(0, mini_cpr_restart("integration_test", version));

        // compute check data for checkpoint version
        int * check[DATA_SIZE];
        for(int i = 0; i < DATA_SIZE; ++i) {
            check[i] = malloc((DATA_SIZE - i) * sizeof(int));
        }
        for(int check_version = 0; check_version <= version; ++check_version) {
            if(check_version == 0) {
                for(int i = 0; i < DATA_SIZE; ++i) {
                    for(int j = 0; j < DATA_SIZE - i; ++j) {
                        check[i][j] = i*j;
                    }
                }
            } else {
                for(int i = 0; i < DATA_SIZE; ++i) {
                    for(int j = 0; j < DATA_SIZE - i; ++j) {
                        check[i][j] = check[i][j]*check_version;
                    }
                }
            }
            if(check_version == version) {
                // compare check data to data loaded from checkpoint
                for(int i = 0; i < DATA_SIZE; ++i) {
                    for(int j = 0; j < DATA_SIZE - i; ++j) {
                        TEST_ASSERT_EQUAL_INT(check[i][j], data[i][j]);
                    }
                }
            }
        }

        // clean up check
        for(int i = 0; i < DATA_SIZE; ++i) {
            free(check[i]);
        }

        // start loop in following iteration
        ++version;
    }

    for(; version < MAX_ITERATIONS; ++version) {
        printf("Begin iteration %d\n", version);

        // loop manipulates data somehow
        if(version == 0) {
            for(int i = 0; i < DATA_SIZE; ++i) {
                for(int j = 0; j < DATA_SIZE - i; ++j) {
                    data[i][j] = i*j;
                }
            }
        } else {
            for(int i = 0; i < DATA_SIZE; ++i) {
                for(int j = 0; j < DATA_SIZE - i; ++j) {
                    data[i][j] = data[i][j]*version;
                }
            }
        }

        // checkpoint at end of iteration
        TEST_ASSERT_EQUAL_INT(0, mini_cpr_checkpoint("integration_test", version));

        // wait for kill after checkpoint
        printf("Reached checkpoint %d, awaiting kill", version);
        fflush(stdout);
        sleep(1);
        printf(".");
        fflush(stdout);
        sleep(1);
        printf(".");
        fflush(stdout);
        sleep(1);
        printf(".\n");
        fflush(stdout);
        sleep(1);
    }

    //final data check
    for(int i = 0; i < DATA_SIZE; ++i) {
        for(int j = 0; j < DATA_SIZE - i; ++j) {
            TEST_ASSERT_EQUAL_INT(i*j*362880, data[i][j]);
        }
    }

    // clean up
    for(int i = 0; i < DATA_SIZE; ++i) {
        free(data[i]);
    }
    mini_cpr_fin();
    char filename[37] = "assets/checkpoints/integration_test0";
    for(char c = '0'; c <= '9'; ++c){
        filename[35] = c;
        remove(filename);
    }
}


int main(void) {
    UNITY_BEGIN();
    RUN_TEST(integration_test);
    return UNITY_END();
}