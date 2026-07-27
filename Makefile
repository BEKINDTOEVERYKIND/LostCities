CC      ?= gcc
CFLAGS  ?= -O3 -march=native -ffast-math -funroll-loops -Wall -Wextra -std=c11
LDFLAGS ?= -lm -pthread

BIN     := bin
SRC     := src

HDRS    := $(wildcard $(SRC)/*.h)
CORE    := $(SRC)/lc.c $(SRC)/features.c $(SRC)/net.c $(SRC)/heuristic.c \
           $(SRC)/search.c $(SRC)/rollout.c $(SRC)/agent.c $(SRC)/match.c $(SRC)/spec.c

all: $(BIN)/test_engine $(BIN)/arena $(BIN)/train $(BIN)/bench $(BIN)/probe $(BIN)/rl $(BIN)/ladder $(BIN)/play $(BIN)/showgame

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_engine: tests/test_engine.c $(SRC)/lc.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/arena: tools/arena.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/train: tools/train.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/bench: tools/bench.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: $(BIN)/test_engine
	./$(BIN)/test_engine

clean:
	rm -rf $(BIN)

.PHONY: all test clean

$(BIN)/probe: tools/probe.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/rl: tools/rl.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/ladder: tools/ladder.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/play: tools/play.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/showgame: tools/showgame.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/dumpfeat: tools/dumpfeat.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN)/analyze: tools/analyze.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
