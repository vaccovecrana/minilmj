package io.vacco.minilm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * MiniLM embeddings model - JNI wrapper around C implementation.
 * Thread-safe for concurrent inference using a single shared session.
 */
public final class MiniLM implements AutoCloseable {

  private static boolean libraryLoaded = false;
  private final long sessionHandle;
  private ExecutorService executorService = null;

  /**
   * Create a new MiniLM session.
   *
   * @param tbfPath   Path to the BERT weights file (.tbf)
   * @param vocabPath Path to the vocabulary file (vocab.txt)
   * @throws IllegalArgumentException if model or vocabulary files cannot be found
   * @throws RuntimeException if session creation fails
   */
  public MiniLM(String tbfPath, String vocabPath) {
    loadLibrary();

    var tbfFile = new File(tbfPath);
    if (!tbfFile.exists() || !tbfFile.isFile()) {
      throw new IllegalArgumentException(
        "Model file not found: " + tbfPath + " (absolute path: " + tbfFile.getAbsolutePath() + ")");
    }

    var vocabFile = new File(vocabPath);
    if (!vocabFile.exists() || !vocabFile.isFile()) {
      throw new IllegalArgumentException(
        "Vocabulary file not found: " + vocabPath + " (absolute path: " + vocabFile.getAbsolutePath() + ")");
    }

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

    // Detect platform and architecture
    var osName = System.getProperty("os.name").toLowerCase();
    var osArch = System.getProperty("os.arch").toLowerCase();
    
    // Normalize architecture names
    boolean isArm64 = osArch.equals("aarch64") || osArch.equals("arm64");
    boolean isAmd64 = osArch.equals("x86_64") || osArch.equals("amd64");
    
    // Determine OS and architecture directory
    String osDir;
    String libraryName;
    
    if (osName.contains("mac") || osName.contains("darwin")) {
      if (isArm64) {
        osDir = "macos-arm64";
      } else if (isAmd64) {
        osDir = "macos-amd64";
      } else {
        throw new UnsupportedOperationException(
          "Unsupported architecture: " + osArch + " on " + osName);
      }
      libraryName = "libminilm.dylib";
    } else if (osName.contains("linux")) {
      if (isArm64) {
        osDir = "linux-arm64";
      } else if (isAmd64) {
        osDir = "linux-amd64";
      } else {
        throw new UnsupportedOperationException(
          "Unsupported architecture: " + osArch + " on " + osName);
      }
      libraryName = "libminilm.so";
    } else {
      throw new UnsupportedOperationException(
        "Unsupported platform: " + osName + " (" + osArch + ")");
    }

    // Try to load from JAR resources
    var resourcePath = "/native/" + osDir + "/" + libraryName;
    loadLibraryFromResource(resourcePath, libraryName);
  }

  private static void loadLibraryFromResource(String resourcePath, String libraryName) {
    var libraryStream = MiniLM.class.getResourceAsStream(resourcePath);

    try (libraryStream) {
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

  /**
   * Generate embeddings for a list of texts. This method is thread-safe and can be used concurrently.
   *
   * @param texts      List of input texts to embed
   * @param concurrent If true, process embeddings concurrently using a thread pool.
   *                   If false, process sequentially.
   * @return List of embedding vectors, each as float[384]
   * @throws RuntimeException if embedding generation fails
   */
  public List<float[]> embedBatch(List<String> texts, boolean concurrent) {
    if (texts == null) {
      throw new IllegalArgumentException("Texts list cannot be null");
    }
    if (texts.isEmpty()) {
      return new ArrayList<>();
    }

    List<float[]> results = new ArrayList<>(texts.size());

    if (concurrent) {
      // Use concurrent processing with instance executor service
      synchronized (this) {
        if (executorService == null || executorService.isShutdown()) {
          int numThreads = Runtime.getRuntime().availableProcessors();
          executorService = Executors.newFixedThreadPool(numThreads);
        }
      }

      List<Future<float[]>> futures = new ArrayList<>(texts.size());
      for (String text : texts) {
        if (text == null) {
          throw new IllegalArgumentException("Text in list cannot be null");
        }
        Future<float[]> future = executorService.submit(() -> {
          float[] result = nEmbed(sessionHandle, text);
          if (result == null) {
            throw new RuntimeException("Failed to generate embedding for: " + text);
          }
          return result;
        });
        futures.add(future);
      }

      // Collect results
      for (Future<float[]> future : futures) {
        try {
          results.add(future.get());
        } catch (Exception e) {
          throw new RuntimeException("Failed to get embedding result", e);
        }
      }
    } else {
      // Sequential processing
      for (String text : texts) {
        if (text == null) {
          throw new IllegalArgumentException("Text in list cannot be null");
        }
        results.add(embed(text));
      }
    }

    return results;
  }

  @Override public void close() {
    synchronized (this) {
      if (executorService != null && !executorService.isShutdown()) {
        executorService.shutdown();
        try {
          if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
            executorService.shutdownNow();
          }
        } catch (InterruptedException e) {
          executorService.shutdownNow();
          Thread.currentThread().interrupt();
        }
      }
    }
    if (sessionHandle != 0) {
      nDestroy(sessionHandle);
    }
  }

}
