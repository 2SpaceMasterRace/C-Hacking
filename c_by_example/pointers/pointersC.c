#include <stdio.h>

int main() {
  int array[5];
  int *pointer = &array[0];

  printf("value of pointer variable : %p \n", (void *)pointer);
  printf("value of pointer+1 variable: %p\n", (void *)(pointer + 1));

  char *array_2 = "Hello";
  char *pointer_2 = &array_2;

  int len = sizeof(array_2) / sizeof(array_2[0]);

  for (size_t i = 0; i < sizeof(len); i++) {
    printf("value of pointer variable + %d : %p \n", i,
           (void *)(pointer_2 + i));
  }
}
