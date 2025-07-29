int main() {
  extern int a;   // defined elsewhere
  static int b;   // hold value between invocations
  register int c; // automatic storage duration. hints the compiler to place the
                  // object in the processor's register
  thread_local int d; // thread storage duration
  return 0;
}
