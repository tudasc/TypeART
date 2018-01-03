#include <stdlib.h>
void test(){
	int *p = (int *) malloc(42 * sizeof(int));
	free(p);
}
