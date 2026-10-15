/* ---------------------------------------------------------------------------
 * Runs a TestU01 battery on a stream of random numbers read from stdin.
 *
 *   testu01 [options] <battery>
 *
 *   <battery>   smallcrush | crush | bigcrush
 *   -n NAME     the generator's name in the report
 *   -r          reverse the bits of every 32-bit word (tests the low bits as
 *               strongly as the high ones; TestU01 weighs the high bits most)
 *   -d          read 64-bit doubles in [0, 1) instead of 32-bit words
 *   -p FILE     also write every statistic's p-value as TSV to FILE
 *   -v          TestU01's detailed output for every test
 *
 * The stream is read as raw native-endian 32-bit unsigned words (or doubles
 * with -d), e.g. from `Rscript gen.R --seed 42 | ./testu01 smallcrush`. The
 * exit status is 0 if no p-value lies outside [1e-10, 1 - 1e-10], 1 if one
 * does, 2 on a usage error and 3 if the stream ended before the battery did.
 * ------------------------------------------------------------------------- */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bbattery.h"
#include "swrite.h"
#include "unif01.h"

#define BUF_BYTES (1u << 22)

static unsigned char buf[BUF_BYTES];
static size_t buf_pos = 0, buf_len = 0;
static unsigned long long n_read = 0;
static int reverse = 0;

static void refill(void) {
  buf_len = fread(buf, 1, BUF_BYTES, stdin);
  buf_pos = 0;
  if (buf_len == 0) {
    fprintf(stderr, "testu01: the stream ended after %llu numbers\n", n_read);
    exit(3);
  }
}

static void next_bytes(void *dst, size_t size) {
  unsigned char *out = dst;
  while (size > 0) {
    if (buf_pos == buf_len) refill();
    size_t take = buf_len - buf_pos < size ? buf_len - buf_pos : size;
    memcpy(out, buf + buf_pos, take);
    buf_pos += take;
    out += take;
    size -= take;
  }
  n_read++;
}

static uint32_t reverse_bits(uint32_t x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
  x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
  return (x >> 16) | (x << 16);
}

static unsigned int next_word(void) {
  uint32_t x;
  next_bytes(&x, sizeof x);
  return reverse ? reverse_bits(x) : x;
}

static double next_double(void) {
  double x;
  next_bytes(&x, sizeof x);
  return x;
}

static void usage(void) {
  fprintf(stderr, "usage: testu01 [-n name] [-r] [-d] [-p file] [-v] smallcrush|crush|bigcrush\n");
  exit(2);
}

int main(int argc, char **argv) {
  char *name = "stdin";
  char *pfile = NULL;
  int doubles = 0, verbose = 0, opt;
  while ((opt = getopt(argc, argv, "n:rdp:v")) != -1) {
    switch (opt) {
    case 'n': name = optarg; break;
    case 'r': reverse = 1; break;
    case 'd': doubles = 1; break;
    case 'p': pfile = optarg; break;
    case 'v': verbose = 1; break;
    default: usage();
    }
  }
  if (optind != argc - 1) usage();
  const char *battery = argv[optind];
  void (*run)(unif01_Gen *);
  if (strcmp(battery, "smallcrush") == 0) run = bbattery_SmallCrush;
  else if (strcmp(battery, "crush") == 0) run = bbattery_Crush;
  else if (strcmp(battery, "bigcrush") == 0) run = bbattery_BigCrush;
  else usage();
  if (reverse && doubles) {
    fprintf(stderr, "testu01: -r applies to 32-bit words, not to -d\n");
    exit(2);
  }

  swrite_Basic = verbose;
  unif01_Gen *gen = doubles ? unif01_CreateExternGen01(name, next_double)
                            : unif01_CreateExternGenBits(name, next_word);
  run(gen);

  fprintf(stderr, "testu01: %s consumed %llu %s\n", battery, n_read,
          doubles ? "doubles" : "32-bit words");

  int failed = 0;
  FILE *pf = pfile ? fopen(pfile, "w") : NULL;
  if (pfile && !pf) {
    perror(pfile);
    exit(2);
  }
  if (pf) fprintf(pf, "statistic\ttest\tp_value\n");
  for (int j = 0; j < bbattery_NTests; j++) {
    double p = bbattery_pVal[j];
    if (p < 0.0) continue; /* not computed */
    if (p < 1e-10 || p > 1.0 - 1e-10) failed = 1;
    if (pf) fprintf(pf, "%d\t%s\t%.6g\n", j + 1, bbattery_TestNames[j], p);
  }
  if (pf) fclose(pf);

  if (doubles) unif01_DeleteExternGen01(gen);
  else unif01_DeleteExternGenBits(gen);
  return failed;
}
