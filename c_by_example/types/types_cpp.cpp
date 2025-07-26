#include <climits>
#include <cstdint>
#include <iostream>

int main() {
  int a;
  std::cout << "Size of int : " << sizeof(a) * CHAR_BIT << std::endl;

  unsigned int b;
  std::cout << "Size of unsinged int: " << sizeof(b) * CHAR_BIT << std::endl;

  char c;
  std::cout << "Size of char :" << sizeof(c) * CHAR_BIT << std::endl;

  short d;
  std::cout << "Size of short :" << sizeof(d) * CHAR_BIT << std::endl;

  long e;
  std::cout << "Size of long: " << sizeof(e) * CHAR_BIT << std::endl;

  float f;
  std::cout << "Size of float: " << sizeof(f) * CHAR_BIT << std::endl;

  double g;
  std::cout << "Size of double: " << sizeof(g) * CHAR_BIT << std::endl;

  bool h;
  std::cout << "Size of bool : " << sizeof(h) * CHAR_BIT << std::endl;

  auto i = 10;
  std::cout << "Size of auto : " << sizeof(i) * CHAR_BIT << std::endl;

  return 0;
}
