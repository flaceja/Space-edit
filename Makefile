CFLAGS ?= -O3 -ffast-math -fopenmp -march=native
all: src/render
src/render: src/render.c
	gcc $(CFLAGS) -o $@ $< -lm
clean:
	rm -f src/render
.PHONY: all clean
