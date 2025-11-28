plugins { id("io.vacco.oss.gitflow") version "1.8.3" }

group = "io.vacco.minlm"
version = "0.5.0"

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addJ8Spec()
  sharedLibrary(true, false)
}

val osName = System.getProperty("os.name").lowercase()
val osArch = System.getProperty("os.arch").lowercase()

val libraryName = when {
    osName.contains("mac") || osName.contains("darwin") -> "libminilm.dylib"
    osName.contains("linux") -> "libminilm.so"
    else -> throw UnsupportedOperationException("Unsupported platform: $osName ($osArch)")
}

val nativeLibDir = file("src/main/resources/native")
val buildLibPath = file("build/lib/$libraryName")

tasks.register("copyNativeLibrary", Copy::class) {
    group = "build"
    description = "Copy native library to resources directory"
    from(buildLibPath)
    into(nativeLibDir)
    rename { libraryName }
}

tasks.named("processResources") {
    dependsOn("copyNativeLibrary")
}

tasks.named("sourcesJar") {
    dependsOn("copyNativeLibrary")
}
