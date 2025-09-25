#include <stdio.h>

int main() {
  int number = 5;

  if (!number)
    printf("%d is 0", number);
  else if (number < 0)
    printf("%d is negative \n", number);
  else {
    printf("%d is positive \n", number);
  }

  int number_2 = 10;
  number_2 < 0 ? printf("%d is negative \n", number_2)
               : printf("%d is positive \n", number_2);
  return 0;
}
