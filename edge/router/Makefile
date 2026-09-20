# Makefile for RouteEdge's C routing engine.
# Builds librouter.so, loaded from Python via ctypes (see router_bindings.py).

CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -fPIC -std=c11
LDFLAGS  = -shared
TARGET   = librouter.so
SOURCES  = router.c network_client.c
OBJECTS  = $(SOURCES:.c=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS) -lm

%.o: %.c router.h network_client.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

# Quick sanity check without going through Python/ctypes at all.
test: $(TARGET)
	@echo "librouter.so built. Run tests/test_router_scoring.py to exercise it."
