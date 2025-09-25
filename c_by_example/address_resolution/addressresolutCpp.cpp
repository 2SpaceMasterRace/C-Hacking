#include <iostream>

int main() {
  int a = 10;
  int *b = &a;

  std::cout << "value of a : " << a << std::endl;
  std::cout << "address of a : " << &a << std::endl;

  std::cout << "value of b :" << *b << std::endl;
  std::cout << "address of b:" << b << std::endl;

  return 0;
}
