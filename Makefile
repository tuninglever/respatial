CC     = cc
CFLAGS = -g -O2 -std=gnu99 -Wall -Wextra -fPIC -Isrc

SRCS = src/fft.c src/stft.c src/nmf.c src/spatial.c src/render.c src/wav.c
OBJS = $(SRCS:.c=.o)

librespatial.dylib: $(OBJS)
	$(CC) -dynamiclib -o $@ $(OBJS) -lm

%.o: %.c src/respatial.h
	$(CC) $(CFLAGS) -c $< -o $@

test: librespatial.dylib
	python3 test_pipeline.py

clean:
	rm -f librespatial.dylib $(OBJS)

.PHONY: test clean
