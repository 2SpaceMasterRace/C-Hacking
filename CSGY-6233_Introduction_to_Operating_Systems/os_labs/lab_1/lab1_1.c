#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  printf("Hello world! This is Introduction to Operating Systems, CS 6233, "
         "Fall 2025 \n");
  printf("Hari Varsha V \n");
  srand(time(NULL));
  // rand() returns a pseudo-random integer value between 0 and RAND_MAX (0 and
  // RAND_MAX included)
  printf("Random number between 0 - 149 : %d \n", rand() % 149);
  return 0;
}
