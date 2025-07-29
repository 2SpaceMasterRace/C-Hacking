#include <iostream>

inline void square(int &x) {
  x *= x;
  return;
}

int main() {
  int x = 10;
  square(x);
  std::cout << "square of x :" << x << std::endl;
  return 0;
}
