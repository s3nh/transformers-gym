#include <stdio.h>
int main() {
    int a[10];
    int i = 0;
for (i = 0; i < sizeof(a) / sizeof(a[0]); i ++) { a[i] = i;
}
printf("array elements:\n");
for (i = 0; i < sizeof(a) / sizeof(a[0]); i ++) {
        printf("\ta[%d] = %d\n", i, a[i]);
    }
    printf("\n");
return 0; }
