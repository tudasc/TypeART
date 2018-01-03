#include <stdlib.h>
typedef struct ms {
	int a;
	double b;
} mystruct;

void test() {
	mystruct *m = (mystruct *) malloc(sizeof(mystruct));
	free(m);
}
