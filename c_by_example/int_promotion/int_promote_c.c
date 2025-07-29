#include <stdio.h>

int main() {
  char x = 'a'; // a = 95
  char y = 'A'; // A = 65

  if (x < y) {
    printf("a is lesser than A: a: %d < A:%d \n", x, y);
  } else {
    printf("a is greater than A: a:%d > A:%d \n", x, y);
  }
}
