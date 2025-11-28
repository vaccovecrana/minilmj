#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include "minilm.h"
#include "tensor.h"

// JNI function: Create a MiniLM session
// Returns: jlong session handle (pointer to minilm_t)
JNIEXPORT jlong JNICALL
Java_io_vacco_minilm_MiniLM_nCreate(JNIEnv *env, jclass clazz, jstring tbfPath, jstring vocabPath)
{
    // Convert Java strings to C strings
    const char *tbf_path = (*env)->GetStringUTFChars(env, tbfPath, NULL);
    if (tbf_path == NULL)
    {
        return 0; // Exception already thrown
    }

    const char *vocab_path = (*env)->GetStringUTFChars(env, vocabPath, NULL);
    if (vocab_path == NULL)
    {
        (*env)->ReleaseStringUTFChars(env, tbfPath, tbf_path);
        return 0; // Exception already thrown
    }

    // Allocate minilm_t on heap
    minilm_t *m = (minilm_t *)malloc(sizeof(minilm_t));
    if (m == NULL)
    {
        (*env)->ReleaseStringUTFChars(env, tbfPath, tbf_path);
        (*env)->ReleaseStringUTFChars(env, vocabPath, vocab_path);
        (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/OutOfMemoryError"),
                         "Failed to allocate memory for MiniLM session");
        return 0;
    }

    // Initialize the session
    int result = minilm_create(m, tbf_path, vocab_path);

    // Release Java string references
    (*env)->ReleaseStringUTFChars(env, tbfPath, tbf_path);
    (*env)->ReleaseStringUTFChars(env, vocabPath, vocab_path);

    if (result != 0)
    {
        free(m);
        (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/RuntimeException"),
                         "Failed to create MiniLM session");
        return 0;
    }

    // Return pointer as jlong
    return (jlong)m;
}

// JNI function: Embed a string
// Returns: jfloatArray with 384 floats (shape [1, 384] flattened)
JNIEXPORT jfloatArray JNICALL
Java_io_vacco_minilm_MiniLM_nEmbed(JNIEnv *env, jclass clazz, jlong sessionHandle, jstring text)
{
    // Convert session handle to pointer
    minilm_t *m = (minilm_t *)sessionHandle;
    if (m == NULL)
    {
        (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/IllegalArgumentException"),
                         "Invalid session handle");
        return NULL;
    }

    // Convert Java string to C string
    const char *text_str = (*env)->GetStringUTFChars(env, text, NULL);
    if (text_str == NULL)
    {
        return NULL; // Exception already thrown
    }

    size_t text_len = strlen(text_str);

    // Call minilm_embed
    tensor_t out;
    t_status status = minilm_embed(*m, (char *)text_str, text_len, &out);

    // Release Java string reference
    (*env)->ReleaseStringUTFChars(env, text, text_str);

    if (status != T_OK)
    {
        (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/RuntimeException"),
                         "Failed to generate embedding");
        return NULL;
    }

    // Verify output shape is [1, 384]
    size_t numel = tensor_numel(out);
    if (numel != 384)
    {
        tensor_destroy(&out);
        (*env)->ThrowNew(env, (*env)->FindClass(env, "java/lang/RuntimeException"),
                         "Unexpected embedding size");
        return NULL;
    }

    // Create Java float array
    jfloatArray result = (*env)->NewFloatArray(env, 384);
    if (result == NULL)
    {
        tensor_destroy(&out);
        return NULL; // Exception already thrown (OutOfMemoryError)
    }

    // Copy tensor data to Java array
    (*env)->SetFloatArrayRegion(env, result, 0, 384, out.data);

    // Clean up tensor
    tensor_destroy(&out);

    return result;
}

// JNI function: Destroy a MiniLM session
JNIEXPORT void JNICALL
Java_io_vacco_minilm_MiniLM_nDestroy(JNIEnv *env, jclass clazz, jlong sessionHandle)
{
    minilm_t *m = (minilm_t *)sessionHandle;
    if (m == NULL)
    {
        return; // Invalid handle, but don't throw exception on cleanup
    }

    // Destroy the session
    minilm_destroy(m);

    // Free the allocated memory
    free(m);
}

