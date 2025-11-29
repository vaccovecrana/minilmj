# Simple Makefile for MiniLM library
CC := zig cc
BUILD := build

# Require JAVA_HOME to be set - fail immediately if not defined
ifeq ($(JAVA_HOME),)
  $(error JAVA_HOME is not defined. Please set JAVA_HOME environment variable to your JDK installation path.)
endif

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Validate JAVA_HOME is not /usr/ (which would cause /usr//include issues)
ifeq ($(JAVA_HOME),/usr)
  $(error Invalid JAVA_HOME detected: /usr/. Please set JAVA_HOME to a valid JDK installation path (e.g., /usr/lib/jvm/java-11-openjdk-amd64).)
endif

# Validate JAVA_HOME contains include directory
JAVA_INCLUDE_DIR := $(JAVA_HOME)/include
ifeq ($(wildcard $(JAVA_INCLUDE_DIR)),)
  $(error JAVA_HOME does not contain an 'include' directory: $(JAVA_INCLUDE_DIR). Please verify JAVA_HOME points to a valid JDK installation.)
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

ARCH_FLAGS := -march=native

# Java resource directory for native libraries
RESOURCE_DIR := src/main/resources/native/$(OS_NAME)-$(ARCH_NAME)

# Platform-specific target flags for zig cc
# On Linux, use native-native-gnu to use zig's bundled libc (avoids system header issues)
# On macOS, don't specify target (zig will use system libc which works fine on macOS)
ifeq ($(UNAME_S),Linux)
  ZIG_TARGET := -target native-native-gnu
else
  ZIG_TARGET :=
endif

# Compiler flags
# Experimenting with aggressive optimizations for ~300ms target
# On Linux: Use zig's bundled libc instead of system headers to avoid header dependency issues
# On macOS: Use system libc (which works fine)
CFLAGS := -std=c11 -g0 -O3 $(ARCH_FLAGS) -ffast-math -ffp-contract=fast -fPIC \
          -funroll-loops -fno-math-errno -fno-trapping-math -fomit-frame-pointer \
          -fno-stack-protector \
          $(ZIG_TARGET) \
          -I src/main/c -I src/main/c/tokenizer $(JAVA_INCLUDES)

# Platform-specific linker flags
# On Linux: Enable LTO and garbage collection of unused sections
# On macOS: Disable LTO (requires LLD which may not be available) but keep other optimizations
ifeq ($(UNAME_S),Linux)
  LDFLAGS := -O3 -flto -Wl,--gc-sections
else
  LDFLAGS := -O3
endif

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
