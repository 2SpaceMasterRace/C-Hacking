#include <stdio.h>

static inline void square(int *x) {
  *x *= *x;
  return;
}

int main() {
  int x = 5;
  square(&x);
  printf("square of inline function = %d\n", x);
  return 0;
}
