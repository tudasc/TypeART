//run: FileCheck %s
//XFAIL: *
#include "stdlib.h"

int main() {

	double *pd = calloc(10, sizeof(double));

	pd = realloc(pd, 20 * sizeof(double));

	return 0;
}
