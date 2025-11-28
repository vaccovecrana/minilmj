# ---- Config ----
CC      := clang
BUILD   := build
TARGET  := $(BUILD)/example

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

INCLUDES := -I src/main/c -I src/main/c/tokenizer

ifeq ($(JAVA_HOME),)
  JAVA_HOME := $(shell /usr/libexec/java_home 2>/dev/null)
  ifeq ($(JAVA_HOME),)
    JAVA_HOME := $(shell java -XshowSettings:properties -version 2>&1 | grep 'java.home' | sed 's/.*java.home = //' | head -1)
  endif
  ifeq ($(JAVA_HOME),)
    ifeq ($(UNAME_S),Darwin)
      JAVA_HOME := $(shell find /Library/Java/JavaVirtualMachines -name "Home" -type d 2>/dev/null | head -1)
    else
      JAVA_HOME := $(shell readlink -f /usr/bin/java 2>/dev/null | sed "s:bin/java::")
    endif
  endif
endif

ifeq ($(JAVA_HOME),)
  $(error JAVA_HOME not found. Please set JAVA_HOME environment variable or ensure Java is installed.)
endif

ifeq ($(UNAME_S),Darwin)
  INCLUDES += -I $(JAVA_HOME)/include -I $(JAVA_HOME)/include/darwin
  LIB_EXT := dylib
  LIB_NAME := libminilm.dylib
else ifeq ($(UNAME_S),Linux)
  INCLUDES += -I $(JAVA_HOME)/include -I $(JAVA_HOME)/include/linux
  LIB_EXT := so
  LIB_NAME := libminilm.so
else
  $(error Unsupported platform: $(UNAME_S))
endif

# CFLAGS for tests (with sanitizer)
CFLAGS_TEST := -std=c11 -g -O3 -ffast-math -march=native -mtune=native -ffp-contract=fast -fsanitize=address $(INCLUDES)

# CFLAGS for library (no sanitizer, position independent code)
CFLAGS_LIB := -std=c11 -g -O3 -ffast-math -march=native -mtune=native -ffp-contract=fast -fPIC $(INCLUDES)

# Default to test flags for backward compatibility
CFLAGS := $(CFLAGS_TEST)
LDFLAGS := -fsanitize=address
LDLIBS  :=
SRCS := $(filter-out src/main/c/%_test.c src/main/c/tokenizer/%_test.c,$(wildcard src/main/c/*.c) $(wildcard src/main/c/tokenizer/*.c)) src/test/c/example.c

OBJS := $(patsubst %.c,$(BUILD)/%.o,$(SRCS))

LIB_SRCS := src/main/c/minilm.c \
            src/main/c/nn.c \
            src/main/c/tensor.c \
            src/main/c/tbf.c \
            src/main/c/tokenizer/tokenizer.c \
            src/main/c/tokenizer/trie.c \
            src/main/c/tokenizer/str.c \
            src/main/c/tokenizer/s8.c \
            src/main/c/jni/minilm_jni.c

LIB_OBJS := $(patsubst src/main/c/%.c,$(BUILD)/lib/%.o,$(LIB_SRCS))

# ---- Rules ----
.PHONY: all help run clean libminilm.dylib libminilm.so libminilm test-tokenizer test-minilm

all: libminilm

help:
	@echo "Available targets:"
	@echo "  make libminilm     - Build the native library (default)"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make all           - Same as 'make libminilm'"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Test targets:"
	@echo "  make test-tokenizer"
	@echo "  make test-minilm"
	@echo ""
	@echo "Platform: $(UNAME_S) ($(UNAME_M))"
	@echo "JAVA_HOME: $(JAVA_HOME)"
	@echo "Library: $(LIB_NAME)"

# Example executable target (optional - only build if example.c exists)
ifneq ($(wildcard src/test/c/example.c),)
$(TARGET): $(OBJS) | $(BUILD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)
endif

# ---- Library Build ----
# Build shared library for macOS
libminilm.dylib: $(LIB_OBJS) | $(BUILD)/lib
	$(CC) -shared -install_name @rpath/libminilm.dylib -o $(BUILD)/lib/libminilm.dylib $(LIB_OBJS) $(LDLIBS)

# Build shared library for Linux
libminilm.so: $(LIB_OBJS) | $(BUILD)/lib
	$(CC) -shared -fPIC -o $(BUILD)/lib/libminilm.so $(LIB_OBJS) $(LDLIBS)

libminilm: $(LIB_NAME)
	@echo ""
	@echo "✓ Successfully built $(LIB_NAME)"
	@echo "  Location: $(BUILD)/lib/$(LIB_NAME)"
	@echo ""

$(BUILD)/lib/%.o: src/main/c/%.c | $(BUILD)/lib
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_LIB) -c $< -o $@

TOKENIZER_TEST_SRCS := src/main/c/tokenizer/tokenizer_test.c src/main/c/tokenizer/tokenizer.c src/main/c/tokenizer/trie.c src/main/c/tokenizer/str.c src/main/c/tokenizer/s8.c
TOKENIZER_TEST_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(TOKENIZER_TEST_SRCS))
$(BUILD)/tokenizer_test: $(TOKENIZER_TEST_OBJS) | $(BUILD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

MINILM_TEST_SRCS := src/main/c/minilm_test.c src/main/c/minilm.c src/main/c/nn.c src/main/c/tensor.c src/main/c/tbf.c src/main/c/tokenizer/tokenizer.c src/main/c/tokenizer/trie.c src/main/c/tokenizer/str.c src/main/c/tokenizer/s8.c
MINILM_TEST_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(MINILM_TEST_SRCS))
$(BUILD)/minilm_test: $(MINILM_TEST_OBJS) | $(BUILD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

test-tokenizer: $(BUILD)/tokenizer_test
	cd src/main/c/tokenizer && ../../../$(BUILD)/tokenizer_test

test-minilm: $(BUILD)/minilm_test
	cd src/main/c && ../../$(BUILD)/minilm_test

$(BUILD)/%.o: %.c | $(BUILD)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_TEST) -c $< -o $@

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/lib:
	@mkdir -p $(BUILD)/lib

run: $(TARGET)
	time ./$(TARGET)

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD)
	rm -f *_output.txt *_test_output.txt
	@echo "✓ Clean complete"
