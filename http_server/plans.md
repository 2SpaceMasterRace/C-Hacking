I've spent the last 2 weeks learning about Nginx and Assembly in order to make a very small HTTP server.

It's currently around 5.3 KiB.

• no dependencies/libc/bloat
• multiple processes managed via epoll
• single arena allocation

and it is very, very, very fast.

I had to write my own ELF headers, glibc bindings, URL decoding algorithm etc.

This could probably be even faster too. I didn't even use any SIMD instructions (only SWAR) and all the events are level triggered. Not to mention it's very prone to DoS attacks.

Regardless, I learned a ton along the way about Assembly, syscalls and Linux as a whole.

benchmarks
https://github.com/peachey2k2/relayouter
https://github.com/gd-arnold/tiny-nginx
