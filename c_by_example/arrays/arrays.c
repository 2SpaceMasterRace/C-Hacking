#include <stdio.h>

int main() {
  int array[5];
  int array_2[] = {0, 1, 2, 3, 4, 5};

  size_t len = sizeof(array) / sizeof(array[0]);

  for (size_t i = 0; i < len; i++) {
    array[i] = i;
  }

  printf("size of array: %d\n", sizeof(array));
  printf("size of array[0]: %d\n", sizeof(array[0]));
  printf("value of len: %d\n", len);
  printf("size of len: %d\n", sizeof(len));

  printf("element 2 in array: %d\n", array[2]);
  return 0;
}
