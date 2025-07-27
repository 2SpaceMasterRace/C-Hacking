#include <stdio.h>

int double_a(int x) { return x * 2; }

int double_b(int *y) { return *y *= 2; }

int main() {
  int a = 3, b = 6;
  printf("Original Values: %d %d \n", a, b);
  printf("Pass by value: %d \n", double_a(a));
  printf("Pass by reference: %d \n", double_b(&b));
  printf("Modified values: %d %d \n", a, b);
  return 0;
}
