# minilmj

Java JNI wrapper for MiniLM embeddings. Self-contained JAR with native libraries for macOS x64 and Linux.

## Model & Formats

* **Weights**: expected in `.tbf` format named `bert_weights.tbf`. See `scripts/dump_tbf1.py` for an example.
* **Vocab**: `vocab.txt` (one token per line, BERT-style).

## Usage

```java
import io.vacco.minilm.MiniLM;

try (MiniLM model = new MiniLM("path/to/bert_weights.tbf", "path/to/vocab.txt")) {
    float[] embedding = model.embed("Hello, world!");
    // Returns float[384] embedding vector
}
```

## Building

```bash
# Build native library (requires clang)
make libminilm

# Build JAR
./gradlew build
```
