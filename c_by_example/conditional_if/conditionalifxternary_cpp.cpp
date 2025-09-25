#include <iostream>

int main() {

  if (int x = 4; !x)
    std::cout << "x is 0" << std::endl;
  else if (x < 0)
    std::cout << "x is less than 0" << std::endl;
  else {
    std::cout << "x is greater than 0" << std::endl;
  }

  int y = 10;

  y < 0 ? std::cout << "y is less than 0" << std::endl
        : std::cout << "y is greater than 0" << std::endl;

  return 0;
}
