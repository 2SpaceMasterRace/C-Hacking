#include <stdbool.h>
#include <stdio.h>

int main() {
  int a;
  printf("size of int: %d \n", sizeof(a));
  unsigned int b;
  printf("size of unsigned: %d \n", sizeof(b));
  char c;
  printf("size of char: %d \n", sizeof(c));
  long e;
  printf("size of long: %d \n", sizeof(e));
  short f;
  printf("size of short: %d \n", sizeof(f));
  float g;
  printf("size of float: %f \n", sizeof(g));
  double h;
  printf("size of double: %f \n", sizeof(h));
  bool i;
  printf("size of bool: %d \n", sizeof(i));
  return 0;
}
