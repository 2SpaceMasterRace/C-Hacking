#include <iostream>

int main() {
  auto x = 'a'; // a = 95
  auto y = 'A'; // A = 65

  if (x < y) {
    std::cout << "a is less than A : x < y " << std::endl;
  } else {
    std::cout << "a is greater than A: x > y" << std::endl;
  }
  return 0;
}
