#include <iostream>

int main() {
  auto x = 1;

  while (x <= 5) {
    std::cout << "While loop cpp edition: x = " << x++ << std::endl;
  }
  return 0;
}
