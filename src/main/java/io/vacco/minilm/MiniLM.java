package io.vacco.minilm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * MiniLM embeddings model - JNI wrapper around C implementation.
 * Thread-safe for concurrent inference using a single shared session.
 */
public final class MiniLM implements AutoCloseable {

  private static boolean libraryLoaded = false;
  private final long sessionHandle;

  /**
   * Create a new MiniLM session.
   *
   * @param tbfPath   Path to the BERT weights file (.tbf)
   * @param vocabPath Path to the vocabulary file (vocab.txt)
   * @throws RuntimeException if session creation fails
   */
  public MiniLM(String tbfPath, String vocabPath) {
    loadLibrary();
    long handle = nCreate(tbfPath, vocabPath);
    if (handle == 0) {
      throw new RuntimeException("Failed to create MiniLM session");
    }
    this.sessionHandle = handle;
  }

  // Native method declarations
  private static native long nCreate(String tbfPath, String vocabPath);

  private static native float[] nEmbed(long sessionHandle, String text);

  private static native void nDestroy(long sessionHandle);

  private static synchronized void loadLibrary() {
    if (libraryLoaded) {
      return;
    }

    // Detect platform
    var osName = System.getProperty("os.name").toLowerCase();
    var osArch = System.getProperty("os.arch").toLowerCase();
    String libraryName;

    if (osName.contains("mac") || osName.contains("darwin")) {
      libraryName = "libminilm.dylib";
    } else if (osName.contains("linux")) {
      libraryName = "libminilm.so";
    } else {
      throw new UnsupportedOperationException(
        "Unsupported platform: " + osName + " (" + osArch + ")"
      );
    }

    // Try to load from JAR resources
    var resourcePath = "/native/" + libraryName;
    var libraryStream = MiniLM.class.getResourceAsStream(resourcePath);

    try (libraryStream) {
      try {
        if (libraryStream == null) {
          throw new UnsatisfiedLinkError(
            "Native library not found in JAR: " + resourcePath +
              ". Make sure the library is packaged in the JAR resources.");
        }
        // Create temporary file
        var prefix = "libminilm";
        var suffix = libraryName.contains(".dylib") ? ".dylib" : ".so";
        var tempLibraryFile = File.createTempFile(prefix, suffix);
        tempLibraryFile.deleteOnExit();

        Files.copy(libraryStream, tempLibraryFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        System.load(tempLibraryFile.getAbsolutePath());
        libraryLoaded = true;
      } catch (IOException e) {
        throw new UnsatisfiedLinkError(
          "Failed to extract native library: " + e.getMessage());
      }
    } catch (IOException e) {
      // Ignore
    }
  }

  /**
   * Generate an embedding for the given text.
   *
   * @param text Input text to embed
   * @return Embedding vector as float[384]
   * @throws RuntimeException if embedding generation fails
   */
  public float[] embed(String text) {
    if (text == null) {
      throw new IllegalArgumentException("Text cannot be null");
    }

    float[] result = nEmbed(sessionHandle, text);
    if (result == null) {
      throw new RuntimeException("Failed to generate embedding");
    }

    return result;
  }

  @Override public void close() {
    if (sessionHandle != 0) {
      nDestroy(sessionHandle);
    }
  }

}
