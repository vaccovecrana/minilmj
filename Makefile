# Simple Makefile for MiniLM library
CC := zig cc
BUILD := build

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Find JAVA_HOME
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

# Platform-specific settings
ifeq ($(UNAME_S),Darwin)
  JAVA_INCLUDES := -I $(JAVA_HOME)/include -I $(JAVA_HOME)/include/darwin
  LIB_EXT := dylib
  OS_NAME := macos
else ifeq ($(UNAME_S),Linux)
  JAVA_INCLUDES := -I $(JAVA_HOME)/include -I $(JAVA_HOME)/include/linux
  LIB_EXT := so
  OS_NAME := linux
else
  $(error Unsupported platform: $(UNAME_S))
endif

# Detect architecture
ifeq ($(filter aarch64 arm64,$(UNAME_M)),$(UNAME_M))
  ARCH_NAME := arm64
else ifeq ($(filter x86_64 amd64,$(UNAME_M)),$(UNAME_M))
  ARCH_NAME := amd64
else
  $(error Unsupported architecture: $(UNAME_M))
endif

# Java resource directory for native libraries
RESOURCE_DIR := src/main/resources/native/$(OS_NAME)-$(ARCH_NAME)

# Use native platform optimizations - will be built on target server for minimum supported platforms
ARCH_FLAGS := -march=native

# Compiler flags
# Experimenting with aggressive optimizations for ~300ms target
CFLAGS := -std=c11 -g0 -O3 $(ARCH_FLAGS) -ffast-math -ffp-contract=fast -fPIC \
          -funroll-loops -fno-math-errno -fno-trapping-math -fomit-frame-pointer \
          -fno-stack-protector \
          -I src/main/c -I src/main/c/tokenizer $(JAVA_INCLUDES)
LDFLAGS := -O3 -flto -Wl,--gc-sections

# Source files
LIB_SRCS := src/main/c/minilm.c \
            src/main/c/nn.c \
            src/main/c/tensor.c \
            src/main/c/tbf.c \
            src/main/c/tokenizer/tokenizer.c \
            src/main/c/tokenizer/trie.c \
            src/main/c/tokenizer/str.c \
            src/main/c/tokenizer/s8.c \
            src/main/c/jni/minilm_jni.c

TEST_SRCS := src/test/c/embedding_timing_test.c \
             src/test/c/tokenizer_test.c \
             src/test/c/minilm_test.c \
             src/test/c/tensor_test.c \
             src/test/c/trie_test.c \
             src/test/c/gradual_token_test.c
# Debug test (optional):
#             src/test/c/debug_attention_mask.c

# Object files
LIB_OBJS := $(patsubst src/main/c/%.c,$(BUILD)/lib/%.o,$(LIB_SRCS))
TEST_OBJS := $(patsubst src/test/c/%.c,$(BUILD)/test/%.o,$(TEST_SRCS))

.PHONY: all libminilm copyResources test run-tests clean

all: libminilm copyResources test run-tests
	@echo "✓ Build complete (run 'make test' to build tests)"

# Build library
libminilm: $(BUILD)/lib/libminilm.$(LIB_EXT)
	@echo "✓ Built libminilm.$(LIB_EXT)"

# Copy library to Java resources directory
copyResources: libminilm
	@mkdir -p $(RESOURCE_DIR)
	@cp $(BUILD)/lib/libminilm.$(LIB_EXT) $(RESOURCE_DIR)/libminilm.$(LIB_EXT)
	@echo "✓ Copied libminilm.$(LIB_EXT) to $(RESOURCE_DIR)/"

$(BUILD)/lib/%.o: src/main/c/%.c | $(BUILD)/lib
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/lib/libminilm.$(LIB_EXT): $(LIB_OBJS) | $(BUILD)/lib
	$(CC) -shared $(LDFLAGS) -fPIC -o $@ $^
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libminilm.dylib $@
endif

# Test executables
TEST_EXES := $(patsubst src/test/c/%.c,$(BUILD)/test/%,$(TEST_SRCS))

test: $(TEST_EXES)
	@echo "✓ All tests built"

# Run tests
run-tests: test
	@echo "Running all tests..."
	@echo ""
	@echo "=== Tokenizer Test ==="
	@$(BUILD)/test/tokenizer_test || true
	@echo ""
	@echo "=== Trie Test ==="
	@$(BUILD)/test/trie_test || true
	@echo ""
	@echo "=== Tensor Test ==="
	@$(BUILD)/test/tensor_test || true
	@echo ""
	@echo "=== Gradual Token Test ==="
	@$(BUILD)/test/gradual_token_test || true
	@echo ""
	@echo "=== MiniLM Semantic Test ==="
	@$(BUILD)/test/minilm_test || true
	@echo ""
	@echo "=== Embedding Timing Test ==="
	@$(BUILD)/test/embedding_timing_test || true
	@echo ""
	@echo "✓ All tests completed"

$(BUILD)/test/%.o: src/test/c/%.c | $(BUILD)/test
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# All tests use shared library for performance
$(BUILD)/test/%: $(BUILD)/test/%.o $(BUILD)/lib/libminilm.$(LIB_EXT) | $(BUILD)/test
	$(CC) -L$(BUILD)/lib -Wl,-rpath,$(BUILD)/lib $(LDFLAGS) $< -lminilm -lm -o $@

# Build directories
$(BUILD)/lib:
	@mkdir -p $(BUILD)/lib

$(BUILD)/test:
	@mkdir -p $(BUILD)/test

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD)
	@echo "✓ Clean complete"
