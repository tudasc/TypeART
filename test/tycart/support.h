/*
 * support.h
 *
 *  Created on: Mar 16, 2020
 *      Author: ahueck
 */

#ifndef TEST_TYCART_RT_SUPPORT_H_
#define TEST_TYCART_RT_SUPPORT_H_

#include "tycart.h"
#include <stdio.h>

#define make_assert(id, ptr, num, type) \
  TY_protect_mem(id, ptr, num, type);   \
  printf("No assert %i.\n", id);

#endif /* TEST_TYCART_RT_SUPPORT_H_ */
