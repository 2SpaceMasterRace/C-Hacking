#include <iostream>

int double_a(int a) { return 2 * a; }

int double_b(int *b) { return *b *= 2; }

int main() {
  int a = 3, b = 6;
  std::cout << "Original Values:" << a << " " << b << std::endl;
  std::cout << "Pass by Value: " << double_a(a) << std::endl;
  std::cout << "Pass by Reference: " << double_b(&b) << std::endl;
  std::cout << "Modified values: " << a << " " << b << std::endl;
  return 0;
}
