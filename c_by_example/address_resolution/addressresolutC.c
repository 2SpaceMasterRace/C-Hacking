#include <stdio.h>

int main() {
  int a = 10;
  int *b = &a;

  printf("address of a is : %p\n", (void *)&a);
  printf("value of a is : %d \n", *b);

  printf("address of b but access modifier is d instead of p  %d\n", &b);
  printf("value of b but without the derefernce operator: %d\n", b);

  return 0;
}
