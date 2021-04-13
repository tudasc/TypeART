//
// Created by mority on 11/19/19.
//

#include "stdlib.h"
#include "string.h"
#include "include/unity.h"
#include "../mini-cpr.h"
#include "../mini-cpr_p.h"

#define CFG_FILE_PATH "./assets/test.config"


// run before and after each test, unity requires these to be defined
void setUp(void){mini_cpr_init(CFG_FILE_PATH);}
void tearDown(void){mini_cpr_fin();}

void test_mini_cpr_read_config(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_read_config(CFG_FILE_PATH));
    TEST_ASSERT_EQUAL_STRING("./assets/checkpoints", mini_cpr_checkpoint_dir());
}

void test_mini_cpr_rt_init(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
}

void test_mini_cpr_init(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_init(CFG_FILE_PATH));
}

void test_mini_cpr_rt_aux_hash(void) {
    TEST_ASSERT_EQUAL_INT(42, mini_cpr_rt_aux_hash(42));
    TEST_ASSERT_EQUAL_INT(1, mini_cpr_rt_aux_hash(129));
}

void test_mini_cpr_rt_hash(void) {
    TEST_ASSERT_EQUAL_INT(1, mini_cpr_rt_hash(42, 87));
}

void test_mini_cpr_rt_insert(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    for(int i = 0; i < 64; ++i) {
        TEST_ASSERT_EQUAL_INT(i,mini_cpr_rt_insert(i, NULL, i, i));
    }
    TEST_ASSERT_EQUAL_FLOAT(0.5, mini_cpr_rt_loadfactor());
    for(int j = 128; j < 192; ++j) {
        TEST_ASSERT_EQUAL_INT(j - 64, mini_cpr_rt_insert(j, NULL, j, j));
    }
    TEST_ASSERT_EQUAL_FLOAT(1.0, mini_cpr_rt_loadfactor());
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_insert(192, NULL, 192, 192));
}

void test_mini_cpr_rt_search(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_search(0));
    for(int i = 0; i < 128; i += 2) {
        TEST_ASSERT_EQUAL_INT(i,mini_cpr_rt_insert(i, NULL, i, i));
    }
    TEST_ASSERT_EQUAL_FLOAT(0.5, mini_cpr_rt_loadfactor());
    for(int j = 126; j >= 0; j -= 2) {
        TEST_ASSERT_EQUAL_INT(j,mini_cpr_rt_search(j));
    }
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_search(128));
}

void test_mini_cpr_rt_resize(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    for(int i = 0; i < 64; i += 2) {
        TEST_ASSERT_EQUAL_INT(i,mini_cpr_rt_insert(i, NULL, i, i));
    }
    TEST_ASSERT_EQUAL_FLOAT(0.25, mini_cpr_rt_loadfactor());
    for(int j = 192; j < 256; ++j) {
        TEST_ASSERT_EQUAL_INT(j - 128, mini_cpr_rt_insert(j, NULL, j, j));
    }
    TEST_ASSERT_EQUAL_FLOAT(0.75, mini_cpr_rt_loadfactor());
    TEST_ASSERT_EQUAL_INT(0,mini_cpr_rt_resize());
    TEST_ASSERT_EQUAL_FLOAT(0.375, mini_cpr_rt_loadfactor());
    for(int k = 0; k < 64; k += 2) {
        TEST_ASSERT_EQUAL_INT(k,mini_cpr_rt_search(k));
    }
    for(int l = 192; l < 256; ++l) {
        TEST_ASSERT_EQUAL_INT(l,mini_cpr_rt_search(l));
    }
}

void test_mini_cpr_makeMemregion(void) {
    Mem_region_ptr mrp = mini_cpr_makeMemregion(42, NULL, 23, 666);
    TEST_ASSERT_EQUAL_INT(42,mrp->id);
    TEST_ASSERT_EQUAL(NULL, mrp->start_addr);
    TEST_ASSERT_EQUAL_size_t(23, mrp->count);
    TEST_ASSERT_EQUAL_size_t(666, mrp->size);
}

void test_mini_cpr_rt_update(void) {
    Mem_region_ptr mrp = NULL;
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_update(mrp, 43, NULL, 24, 667));
    mrp = mini_cpr_makeMemregion(42, NULL, 23, 666);
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_update(mrp, 43, NULL, 24, 667));
    TEST_ASSERT_EQUAL_INT(43,mrp->id);
    TEST_ASSERT_EQUAL(NULL, mrp->start_addr);
    TEST_ASSERT_EQUAL_size_t(24, mrp->count);
    TEST_ASSERT_EQUAL_size_t(667, mrp->size);
    free(mrp);
}

void test_mini_cpr_unregister(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_unregister(42));
    TEST_ASSERT_EQUAL_INT(42, mini_cpr_rt_insert(42, NULL, 42, 42));
    TEST_ASSERT_EQUAL_INT(43, mini_cpr_rt_insert(170, NULL, 170, 170));
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_unregister(42));
    TEST_ASSERT_EQUAL_INT(42, mini_cpr_rt_insert(298, NULL, 298, 298));
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_unregister(170));
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_unregister(298));
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_search(42));
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_search(170));
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_rt_search(298));
    TEST_ASSERT_EQUAL_INT(42, mini_cpr_rt_insert(170, NULL, 170, 170));
}

void test_mini_cpr_register(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    for(int i = 0; i < 1048576; ++i) {
        TEST_ASSERT_EQUAL_INT(0, mini_cpr_register(i, NULL, i, i));
    }
    TEST_ASSERT_EQUAL_FLOAT(0.5,mini_cpr_rt_loadfactor());
}

void test_mini_cpr_open_cp_file(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    FILE * fp = mini_cpr_open_cp_file("test_mini_cpr_open_cp_file", 0, "r");
    TEST_ASSERT_EQUAL(NULL, fp);
    fp = mini_cpr_open_cp_file("test_mini_cpr_open_cp_file", 0, "w");
    TEST_ASSERT(fp != NULL);
    fclose(fp);
    TEST_ASSERT_EQUAL_INT(0, remove("assets/checkpoints/test_mini_cpr_open_cp_file0"));
}

void test_mini_cpr_write_mem_region(void) {
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_rt_init());
    char * str = "Hello World";
    Mem_region_ptr mrp = mini_cpr_makeMemregion(42, str, strlen(str) + 1, sizeof(char));
    FILE * fp = mini_cpr_open_cp_file("test_mini_cpr_write_mem_region", 0, "wb");
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_write_mem_region(fp, mrp));
    fclose(fp);
    fp = fopen("assets/checkpoints/test_mini_cpr_write_mem_region0", "rb");
    int read_id;
    TEST_ASSERT_EQUAL_INT(1, fread(&read_id, sizeof(int), 1, fp));
    TEST_ASSERT_EQUAL_INT(42, read_id);
    size_t read_count;
    TEST_ASSERT_EQUAL_INT(1, fread(&read_count, sizeof(size_t), 1, fp));
    TEST_ASSERT_EQUAL_size_t(strlen(str)+ 1, read_count);
    size_t read_size;
    TEST_ASSERT_EQUAL_INT(1, fread(&read_size, sizeof(size_t), 1, fp));
    TEST_ASSERT_EQUAL_size_t(sizeof(char), read_size);
    char * read_str = calloc(read_count, read_size);
    TEST_ASSERT_EQUAL_INT(read_count, fread((void *)read_str, read_size, read_count, fp));
    TEST_ASSERT_EQUAL_STRING(str, read_str);
    fclose(fp);
    free(read_str);
    free(mrp);
    TEST_ASSERT_EQUAL_INT(0, remove("assets/checkpoints/test_mini_cpr_write_mem_region0"));
}

void test_mini_cpr_checkpoint() {
    // on heap
    char * data1 = (char*)malloc(sizeof(char)*6);
    memcpy(data1, "data1", 6);
    // on stack
    char * data2 = "data2";

    TEST_ASSERT_EQUAL_INT(0, mini_cpr_register(1, data1, strlen(data1)+1, sizeof(char)));
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_register(2, data2, strlen(data2)+1, sizeof(char)));
    TEST_ASSERT_EQUAL_INT(1, mini_cpr_rt_search(1));
    Mem_region_ptr mrp1 = mini_cpr_rt_getMemregion(1);
    TEST_ASSERT_EQUAL_size_t(strlen(data1)+1,mrp1->count);
    TEST_ASSERT_EQUAL_size_t(sizeof(char),mrp1->size);
    TEST_ASSERT_EQUAL_STRING(data1, (char*)mrp1->start_addr);
    TEST_ASSERT_EQUAL_INT(2, mini_cpr_rt_search(2));
    TEST_ASSERT_EQUAL_INT(0,mini_cpr_checkpoint("test_mini_cpr_checkpoint", 0));
    FILE * fp = fopen("assets/checkpoints/test_mini_cpr_checkpoint0", "r");
    setbuf(fp, NULL);
    Mem_region mr1;
    mr1.start_addr = NULL;
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_read_mem_region(fp, &mr1));
    TEST_ASSERT_EQUAL_INT(1, mr1.id);
    TEST_ASSERT_EQUAL_size_t(strlen(data1)+1, mr1.count);
    TEST_ASSERT_EQUAL_size_t(sizeof(char), mr1.size);
    TEST_ASSERT_EQUAL_STRING(data1, (char*)mr1.start_addr);
    Mem_region mr2;
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_read_mem_region(fp, &mr2));
    TEST_ASSERT_EQUAL_INT(2, mr2.id);
    TEST_ASSERT_EQUAL_size_t(strlen(data2)+1, mr2.count);
    TEST_ASSERT_EQUAL_size_t(sizeof(char), mr2.size);
    TEST_ASSERT_EQUAL_STRING(data2, (char*)mr2.start_addr);
    fclose(fp);
    free(data1);
    TEST_ASSERT_EQUAL_INT(0, remove("assets/checkpoints/test_mini_cpr_checkpoint0"));
}

void test_mini_cpr_restart_check(void) {
    char * valid_name = "test_mini_cpr_restart_check";
    char * false_name = "false_name";
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_restart_check(valid_name, 0));
    TEST_ASSERT_EQUAL_INT(1, mini_cpr_restart_check(valid_name, 1));
    TEST_ASSERT_EQUAL_INT(2, mini_cpr_restart_check(valid_name, 2));
    TEST_ASSERT_EQUAL_INT(3, mini_cpr_restart_check(valid_name, 3));
    TEST_ASSERT_EQUAL_INT(3, mini_cpr_restart_check(valid_name, 42));
    TEST_ASSERT_EQUAL_INT(-1, mini_cpr_restart_check(false_name, 23));
}

void test_mini_cpr_restart(void) {
    int * data[128];
    for(int i = 0; i < 128; ++i) {
        data[i] = malloc((128 - i) * sizeof(int));
        TEST_ASSERT_EQUAL_INT(0, mini_cpr_register(i, data[i], 128 - i, sizeof(int)));
        for(int j = 0; j < 128 - i; ++j) {
            data[i][j] = i*j;
        }
    }
    TEST_ASSERT_EQUAL_FLOAT(0.5, mini_cpr_rt_loadfactor());
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_checkpoint("test_mini_cpr_restart", 0));
    for(int version = 1; version < 10; ++version) {
        for(int i = 0; i < 128; ++i) {
            for(int j = 0; j < 128 - i; ++j) {
                data[i][j] = data[i][j]*version;
            }
        }
        TEST_ASSERT_EQUAL_INT(0, mini_cpr_checkpoint("test_mini_cpr_restart", version));
    }
    TEST_ASSERT_EQUAL_INT(7, mini_cpr_restart_check("test_mini_cpr_restart", 7));
    TEST_ASSERT_EQUAL_INT(0, mini_cpr_restart("test_mini_cpr_restart", 7));
    for(int i = 0; i < 128; ++i) {
        for(int j = 0; j < 128 - i; ++j) {
            TEST_ASSERT_EQUAL_INT(i*j*5040, data[i][j]);
        }
    }

    for(int i = 0; i < 128; ++i) {
        free(data[i]);
    }
    char filename[42] = "assets/checkpoints/test_mini_cpr_restart0";
    for(char c = '0'; c <= '9'; ++c){
        filename[40] = c;
        TEST_ASSERT_EQUAL_INT(0, remove(filename));
    }
}

/*
 * Test Runner
 */
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mini_cpr_read_config);
    RUN_TEST(test_mini_cpr_rt_init);
    RUN_TEST(test_mini_cpr_init);
    RUN_TEST(test_mini_cpr_rt_aux_hash);
    RUN_TEST(test_mini_cpr_rt_hash);
    RUN_TEST(test_mini_cpr_rt_insert);
    RUN_TEST(test_mini_cpr_rt_search);
    RUN_TEST(test_mini_cpr_rt_resize);
    RUN_TEST(test_mini_cpr_makeMemregion);
    RUN_TEST(test_mini_cpr_rt_update);
    RUN_TEST(test_mini_cpr_unregister);
    RUN_TEST(test_mini_cpr_register);
    RUN_TEST(test_mini_cpr_open_cp_file);
    RUN_TEST(test_mini_cpr_write_mem_region);
    RUN_TEST(test_mini_cpr_checkpoint);
    RUN_TEST(test_mini_cpr_restart_check);
    RUN_TEST(test_mini_cpr_restart);
    return UNITY_END();
}