int main() {
  restrict int *a; // should only access from this pointer
  const int b;     // once defined, value stays constant and cannot be modified
  atomic int c;    // only can be acccessed by one thread at a time
  volatile int d;  // can be modoified externally ; program checks value of c
  return 0;        // even if it hasn't changed
}
