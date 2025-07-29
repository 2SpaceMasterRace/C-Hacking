int main() {
  extern int a;   // define elsewhere
  static int b;   // hold value between invocations
  register int c; // store in CPU register for fast access
  auto int d; // automatic duration - scope lifetime. Implict if not defined.
  _Thread_local int e; // thread storage duration
  return 0;
}
