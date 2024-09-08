#include <stdio.h>

int main()
{
    int a1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int a2[10];
    int i;

    memcpy(a2, a1, sizeof(a1));

    for ( i  =0 , i < sizeof(a1)/ sizeof(a1[0]); i++) {
        print("a1[%d] = %d a2[%d] = %d\n", i, a1[i], i, a2[i]);
    }

    return 0
}
