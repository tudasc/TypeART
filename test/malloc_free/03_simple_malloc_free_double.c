
void test() {
	double *p = (double *) malloc(42 * sizeof(double));
	free(p);
}
