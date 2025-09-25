#include <atomic>

int main() {
  const int a = 1;    // "a" once defined cannot be modified
  std::atomic<int> b; // "b" can only be modified one thread at a time
  volatile int c; // "c" can be modified externally. the program will check for
                  // x's value even if it hasn't been modified locally
  return 0;
}
