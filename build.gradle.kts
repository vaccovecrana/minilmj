plugins { id("io.vacco.oss.gitflow") version "1.8.3" }

group = "io.vacco.minlm"
version = "0.5.0"

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addJ8Spec()
  sharedLibrary(true, false)
}

val nativeLibDir = file("src/main/resources/native")

tasks.register("copyNativeLibraries", Copy::class) {
    group = "build"
    description = "Copy cross-compiled native libraries to resources directory"
    from("build/lib/linux-amd64/libminilm.so") {
        into("linux-amd64")
    }
    from("build/lib/macos-amd64/libminilm.dylib") {
        into("macos-amd64")
    }
    from("build/lib/linux-arm64/libminilm.so") {
        into("linux-arm64")
    }
    from("build/lib/macos-arm64/libminilm.dylib") {
        into("macos-arm64")
    }
    into(nativeLibDir)
}

tasks.named("processResources") {
    dependsOn("copyNativeLibraries")
}

tasks.named("sourcesJar") {
    dependsOn("copyNativeLibraries")
}
