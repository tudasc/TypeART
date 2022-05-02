// This file tests for an specific endless recursion in the filter implementations w.r.t. following store targets
// RUN: %c-to-llvm -fno-discard-value-names %s | %opt -O3 -S \
// RUN: | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 \
// RUN: | %filecheck %s --check-prefix=CHECK-exp-default-opt

// CHECK-exp-default-opt: TypeArtPass [Heap & Stack]
// CHECK-exp-default-opt-next: Malloc :   0
// CHECK-exp-default-opt-next: Free   :   0
// CHECK-exp-default-opt-next: Alloca :   0
// CHECK-exp-default-opt-next: Global :   3

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define XUP   0
#define YUP   1
#define ZUP   2
#define TUP   3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7

#define NODIR -1

#define OPP_DIR(dir) (7 - (dir))
#define NDIRS        8

#define NREPS      1
#define NLOOP      3
#define MAX_LENGTH 6
#define MAX_NUM    16

extern char gauge_action_description[128];
int gauge_action_nloops = NLOOP;
int gauge_action_nreps  = NREPS;
int loop_length[NLOOP];
int loop_num[NLOOP];  // TypeART: kept due to printf

int loop_ind[NLOOP][MAX_LENGTH];
int loop_table[NLOOP][MAX_NUM][MAX_LENGTH];
float loop_coeff[NLOOP][NREPS];  // TypeART: kept due to printf
int loop_char[MAX_NUM];
double loop_expect[NLOOP][NREPS][MAX_NUM];

float beta, mass, u0;  // TypeART: u0 kept, used by log function

extern char* strcpy(char* destination, const char* source);

/* Make table of loops in action */
void make_loop_table() {
  int perm[8], pp[8], ir[4];
  int length, iloop, i, j, chr;
  int vec[MAX_LENGTH];
  int count, flag;
  void char_num(int* dig, int* chr, int length);

  /* defines all loops and their coefficients */
  static int loop_ind[NLOOP][MAX_LENGTH] = {
      {XUP, YUP, XDOWN, YDOWN, NODIR, NODIR},
      {XUP, XUP, YUP, XDOWN, XDOWN, YDOWN},
      {XUP, YUP, ZUP, XDOWN, YDOWN, ZDOWN},
  };
  static int loop_length_in[NLOOP] = {4, 6, 6};

  for (j = 0; j < NLOOP; j++) {
    loop_num[j]    = 0;
    loop_length[j] = loop_length_in[j];
    for (i = 0; i < NREPS; i++) {
      loop_coeff[j][i] = 0.0;
    }
  }

  /* Loop coefficients from Urs */
  loop_coeff[0][0] = 1.0;
  loop_coeff[1][0] = -1.00 / (20.0 * u0 * u0) * (1.00 - 0.6264 * log(u0));
  loop_coeff[2][0] = 1.00 / (u0 * u0) * 0.04335 * log(u0);
  // TypeART: this is transformed to intrinsic IR memcpy!:
  strcpy(gauge_action_description, "\"Symanzik 1x1 + 1x2 + 1x1x1 action\"");
  printf("Symanzik 1x1 + 1x2 + 1x1x1 action\n");

  for (iloop = 0; iloop < NLOOP; iloop++) {
    length = loop_length[iloop];
    count  = 0;
    /* permutations */
    for (perm[0] = 0; perm[0] < 4; perm[0]++)
      for (perm[1] = 0; perm[1] < 4; perm[1]++)
        for (perm[2] = 0; perm[2] < 4; perm[2]++)
          for (perm[3] = 0; perm[3] < 4; perm[3]++) {
            if (perm[0] != perm[1] && perm[0] != perm[2] && perm[0] != perm[3] && perm[1] != perm[2] &&
                perm[1] != perm[3] && perm[2] != perm[3]) {
              /* reflections*/
              for (ir[0] = 0; ir[0] < 2; ir[0]++)
                for (ir[1] = 0; ir[1] < 2; ir[1]++)
                  for (ir[2] = 0; ir[2] < 2; ir[2]++)
                    for (ir[3] = 0; ir[3] < 2; ir[3]++) {
                      for (j = 0; j < 4; j++) {
                        pp[j] = perm[j];

                        if (ir[j] == 1)
                          pp[j] = 7 - pp[j];
                        pp[7 - j] = 7 - pp[j];
                      }
                      /* create new vector*/
                      for (j = 0; j < length; j++)
                        vec[j] = pp[loop_ind[iloop][j]];

                      char_num(vec, &chr, length);
                      flag = 0;
                      /* check if it's a new set: */
                      for (j = 0; j < count; j++)
                        if (chr == loop_char[j])
                          flag = 1;
                      if (flag == 0) {
                        loop_char[count] = chr;
                        for (j = 0; j < length; j++)
                          loop_table[iloop][count][j] = vec[j];
                        count++;
                        /**node0_printf("ADD LOOP: "); printpath( vec, length );**/
                      }
                      if (count > MAX_NUM) {
                        printf("OOPS: MAX_NUM too small\n");
                        exit(0);
                      }
                      loop_num[iloop] = count;

                    } /* end reflection*/
            }         /* end permutation if block */
          }           /* end permutation */
  }                   /* end iloop */

  /* print out the loop coefficients */
  printf("loop coefficients: nloop rep loop_coeff  multiplicity\n");
  for (i = 0; i < NREPS; i++)
    for (j = 0; j < NLOOP; j++) {
      printf("                    %d %d      %e     %d\n", j, i, loop_coeff[j][i], loop_num[j]);
    }

} /* make_loop_table */

/* find a number uniquely identifying the cyclic permutation of a path,
   or the starting point on the path.  Backwards paths are considered
   equivalent here, so scan those too. */
void char_num(int* dig, int* chr, int length) {
  int j;
  int bdig[MAX_LENGTH], tenl, newv, old;
  /* "dig" is array of directions.  "bdig" is array of directions for
backwards path. */

  tenl = 1;
  for (j = 0; j < length - 1; j++)
    tenl = tenl * 10;

  *chr = dig[length - 1];
  for (j = length - 2; j >= 0; j--)
    *chr = *chr * 10 + dig[j];

  /* forward*/
  old = *chr;
  for (j = length - 1; j >= 1; j--) {
    newv = old - tenl * dig[j];
    newv = newv * 10 + dig[j];
    if (newv < *chr)
      *chr = newv;
    old = newv;
  }

  /* backward*/
  for (j = 0; j < length; j++)
    bdig[j] = 7 - dig[length - j - 1];
  old = bdig[length - 1];
  for (j = length - 2; j >= 0; j--)
    old = old * 10 + bdig[j];
  if (old < *chr)
    *chr = old;
  for (j = length - 1; j >= 1; j--) {
    newv = old - tenl * bdig[j];
    newv = newv * 10 + bdig[j];
    if (newv < *chr)
      *chr = newv;
    old = newv;
  }

} /* char_num */
