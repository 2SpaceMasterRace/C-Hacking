#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/module.h>
#include <linux/time.h>
#include <linux/timekeeping.h>

static unsigned long init_jiffies;
static ktime_t init_ktime;

static int __init hello_init(void) {
  int hours, minutes, seconds;
  unsigned int tick_time_ms;
  s64 total_seconds;
  struct timespec64 ts;

  tick_time_ms = 1000 / HZ;
  init_jiffies = jiffies;
  ktime_get_real_ts64(&ts);
  init_ktime = ktime_get_real();
  total_seconds = ts.tv_sec;
  seconds = total_seconds % 60;
  minutes = (total_seconds / 60) % 60;
  hours = (total_seconds / 3600) % 24;

  printk(KERN_INFO "Hello, tick time: %u ms\n", tick_time_ms);
  printk(KERN_INFO "Current time is: %02d:%02d:%02d\n", hours, minutes,
         seconds);
  return 0;
}

static void __exit hello_exit(void) {

  int hours, minutes, seconds;
  unsigned long elapsed_jiffies;
  unsigned long elapsed_ms;
  struct timespec64 ts;
  s64 total_seconds;

  elapsed_jiffies = jiffies - init_jiffies;
  elapsed_ms = jiffies_to_msecs(elapsed_jiffies);
  ktime_get_real_ts64(&ts);
  total_seconds = ts.tv_sec;
  seconds = total_seconds % 60;
  minutes = (total_seconds / 60) % 60;
  hours = (total_seconds / 3600) % 24;

  printk(KERN_INFO "Goodbye, elapsed time: %lu ms\n", elapsed_ms);
  printk(KERN_INFO "Current time is: %02d:%02d:%02d\n", hours, minutes,
         seconds);
}

module_init(hello_init);
module_exit(hello_exit);
