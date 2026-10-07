package main

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
	"github.com/cwbudde/go-pocket-tts/internal/doctor"
	"github.com/cwbudde/go-pocket-tts/internal/modelcfg"
)

func TestProbePocketTTSVersion_MissingExecutable(t *testing.T) {
	_, err := probePocketTTSVersion("/nonexistent/pocket-tts-binary")
	if err == nil {
		t.Fatal("expected error for missing executable")
	}
}

func TestProbePocketTTSVersion_RealExecutable(t *testing.T) {
	// Create a tiny script that exits 0 and prints a fixed string, simulating
	// a pocket-tts binary that honours --version.
	tmp := t.TempDir()

	script := filepath.Join(tmp, "fake-tts")

	writeErr := os.WriteFile(script, []byte("#!/bin/sh\necho 'fake-tts 1.2.3'\n"), 0o755)
	if writeErr != nil {
		t.Fatalf("WriteFile: %v", writeErr)
	}

	got, err := probePocketTTSVersion(script)
	if err != nil {
		t.Fatalf("probePocketTTSVersion: %v", err)
	}

	if got != "fake-tts 1.2.3" {
		t.Errorf("unexpected version output: %q", got)
	}
}

func TestProbePocketTTSVersion_NoVersionFlag(t *testing.T) {
	// pocket-tts 2.x and 3.x have no --version: typer exits 2 on the unknown
	// option, while --help works.
	tmp := t.TempDir()
	script := filepath.Join(tmp, "pocket-tts")

	body := "#!/bin/sh\nif [ \"$1\" = --help ]; then echo 'Usage: pocket-tts'; exit 0; fi\nexit 2\n"

	writeErr := os.WriteFile(script, []byte(body), 0o755)
	if writeErr != nil {
		t.Fatalf("WriteFile: %v", writeErr)
	}

	got, err := probePocketTTSVersion(script)
	if err != nil {
		t.Fatalf("probePocketTTSVersion: %v", err)
	}

	if !strings.Contains(got, "version unknown") {
		t.Errorf("version output %q, want it to say the version is unknown", got)
	}
}

func TestProbePocketTTSVersion_BrokenExecutable(t *testing.T) {
	tmp := t.TempDir()
	script := filepath.Join(tmp, "pocket-tts")

	writeErr := os.WriteFile(script, []byte("#!/bin/sh\nexit 1\n"), 0o755)
	if writeErr != nil {
		t.Fatalf("WriteFile: %v", writeErr)
	}

	_, err := probePocketTTSVersion(script)
	if err == nil {
		t.Fatal("expected an error when neither --version nor --help works")
	}
}

func TestProbePythonVersion_ReturnsVersion(t *testing.T) {
	// python3 or python must be present in CI / dev environments.
	ver, err := probePythonVersion()
	if err != nil {
		t.Skipf("python not available: %v", err)
	}

	if ver == "" {
		t.Error("expected non-empty version string")
	}
}

func TestCollectVoiceFiles_NoManifest(t *testing.T) {
	// Change to a temp dir that has no voices/manifest.json.
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	t.Cleanup(func() {
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore failed: %v", err)
		}
	})

	err = os.Chdir(t.TempDir())
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	files := collectVoiceFiles("voices/manifest.json")
	// With no manifest, should return nil/empty (not panic).
	if len(files) != 0 {
		t.Errorf("expected nil/empty slice without manifest, got %v", files)
	}
}

func TestCollectVoiceFiles_WithManifest(t *testing.T) {
	tmp := t.TempDir()

	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	t.Cleanup(func() {
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore failed: %v", err)
		}
	})

	err = os.Chdir(tmp)
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	voiceDir := filepath.Join(tmp, "voices")

	err = os.MkdirAll(voiceDir, 0o755)
	if err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	// Voice file is stored as "test.bin" relative to the manifest directory.
	voiceFile := filepath.Join(voiceDir, "test.bin")

	err = os.WriteFile(voiceFile, []byte("dummy"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile voice: %v", err)
	}

	manifest := `{"voices":[{"id":"test","path":"test.bin","license":"MIT"}]}`

	err = os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	files := collectVoiceFiles("voices/manifest.json")
	if len(files) != 1 {
		t.Errorf("expected 1 voice file, got %d: %v", len(files), files)
	}
	// Path must be absolute so the doctor check works regardless of CWD.
	if !filepath.IsAbs(files[0]) {
		t.Errorf("expected absolute path, got %q", files[0])
	}
}

// TestCollectVoiceFiles_PathResolvedRelativeToManifest verifies that
// collectVoiceFiles resolves paths relative to the manifest directory, not to
// the working directory. This is the regression test for the bug where
// voice files like "mimi.safetensors" were checked against CWD instead of
// the "voices/" directory next to the manifest.
func TestCollectVoiceFiles_PathResolvedRelativeToManifest(t *testing.T) {
	// Layout inside the fake project root (<tmp>):
	//   <tmp>/
	//     voices/
	//       manifest.json   (references "mimi.safetensors")
	//       mimi.safetensors   (voice file lives HERE, next to the manifest)
	//
	// CWD is set to <tmp> so "voices/manifest.json" resolves, but
	// "mimi.safetensors" does NOT exist at the CWD level — only inside voices/.
	tmp := t.TempDir()

	voiceDir := filepath.Join(tmp, "voices")

	err := os.MkdirAll(voiceDir, 0o755)
	if err != nil {
		t.Fatalf("MkdirAll voiceDir: %v", err)
	}

	// Voice file lives next to the manifest, not at the project root.
	err = os.WriteFile(filepath.Join(voiceDir, "mimi.safetensors"), []byte("dummy"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile voice: %v", err)
	}

	manifest := `{"voices":[{"id":"mimi","path":"mimi.safetensors","license":"CC-BY-4.0"}]}`

	err = os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("WriteFile manifest: %v", err)
	}

	// CWD = project root (<tmp>). "mimi.safetensors" does NOT exist here,
	// only inside "voices/". The old buggy code returned the raw path
	// "mimi.safetensors" which would be stat'd against CWD and not found.
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	t.Cleanup(func() {
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore: %v", err)
		}
	})

	err = os.Chdir(tmp)
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	files := collectVoiceFiles("voices/manifest.json")
	if len(files) != 1 {
		t.Fatalf("expected 1 resolved path, got %d: %v", len(files), files)
	}

	// The returned path must be absolute and point to the actual file.
	if !filepath.IsAbs(files[0]) {
		t.Errorf("path is not absolute: %q", files[0])
	}

	_, err = os.Stat(files[0])
	if err != nil {
		t.Errorf("returned path does not exist: %q (%v)", files[0], err)
	}
}

// TestCollectVoiceFiles_MissingFileResolvedRelativeToManifest verifies that a
// missing voice file is also reported next to the manifest, not relative to
// the working directory, where a file of the same name would hide it.
func TestCollectVoiceFiles_MissingFileResolvedRelativeToManifest(t *testing.T) {
	tmp := t.TempDir()

	voiceDir := filepath.Join(tmp, "voices")

	err := os.MkdirAll(voiceDir, 0o755)
	if err != nil {
		t.Fatalf("MkdirAll voiceDir: %v", err)
	}

	manifest := `{"voices":[{"id":"gone","path":"gone.safetensors","license":"MIT"}]}`

	err = os.WriteFile(filepath.Join(voiceDir, "manifest.json"), []byte(manifest), 0o644)
	if err != nil {
		t.Fatalf("WriteFile manifest: %v", err)
	}

	// A decoy at the working directory that must not satisfy the check.
	err = os.WriteFile(filepath.Join(tmp, "gone.safetensors"), []byte("dummy"), 0o644)
	if err != nil {
		t.Fatalf("WriteFile decoy: %v", err)
	}

	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	t.Cleanup(func() {
		err := os.Chdir(orig)
		if err != nil {
			t.Logf("chdir restore: %v", err)
		}
	})

	err = os.Chdir(tmp)
	if err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	files := collectVoiceFiles("voices/manifest.json")
	want := filepath.Join(cwd, "voices", "gone.safetensors")

	if len(files) != 1 || files[0] != want {
		t.Fatalf("collectVoiceFiles = %v; want [%s]", files, want)
	}

	_, err = os.Stat(files[0])
	if !os.IsNotExist(err) {
		t.Errorf("expected %q to be missing, got err=%v", files[0], err)
	}
}

// germanDoctorConfig is the config doctor gets for --language german, with the
// repo-relative paths seen from this package directory.
func germanDoctorConfig(t *testing.T) config.Config {
	t.Helper()

	paths := config.PathsForLanguage("german")
	root := filepath.Join("..", "..")

	cfg := config.DefaultConfig()
	cfg.TTS.Language = "german"
	cfg.TTS.Backend = config.BackendNative
	cfg.Paths.ModelPath = filepath.Join(root, filepath.FromSlash(paths.ModelPath))
	cfg.Paths.TokenizerModel = filepath.Join(root, filepath.FromSlash(paths.TokenizerModel))
	cfg.Paths.VoiceManifest = filepath.Join(root, filepath.FromSlash(paths.VoiceManifest))

	for _, p := range []string{cfg.Paths.ModelPath, cfg.Paths.TokenizerModel, cfg.Paths.VoiceManifest} {
		_, err := os.Stat(p)
		if err != nil {
			t.Skipf("german assets not downloaded (%v); run pockettts model download --language german", err)
		}
	}

	mc, err := modelcfg.Lookup("german")
	if err != nil {
		t.Fatal(err)
	}

	cfg.Model = mc

	return cfg
}

func TestNewDoctorConfig_GermanPasses(t *testing.T) {
	cfg := germanDoctorConfig(t)

	var out strings.Builder

	result := doctor.Run(newDoctorConfig(cfg, config.BackendNative), &out)
	if result.Failed() {
		t.Fatalf("doctor --language german failed: %v\n%s", result.Failures(), out.String())
	}

	// The downloaded german checkpoint is the ungated one: cloning is reported
	// as unavailable without failing doctor.
	for _, want := range []string{
		"language: german", "default voice juergen:", "tokenizer load: ok",
		doctor.WarnMark + " voice cloning: unavailable",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

func TestNewDoctorConfig_ChecksLanguageFiles(t *testing.T) {
	cfg := germanDoctorConfig(t)
	cfg.Model.FlowLM.LookupTable.NBins--

	result := doctor.Run(newDoctorConfig(cfg, config.BackendNative), io.Discard)
	if !slices.ContainsFunc(result.Failures(), func(f string) bool { return strings.Contains(f, "n_bins") }) {
		t.Errorf("n_bins mismatch: failures = %v; want a tokenizer vocab size failure", result.Failures())
	}

	cfg = germanDoctorConfig(t)
	cfg.Model.DefaultVoice = "nobody"

	result = doctor.Run(newDoctorConfig(cfg, config.BackendNative), io.Discard)
	if !slices.ContainsFunc(result.Failures(), func(f string) bool { return strings.Contains(f, `default voice "nobody"`) }) {
		t.Errorf("unknown default voice: failures = %v; want a default voice failure", result.Failures())
	}

	cfg = germanDoctorConfig(t)
	cfg.Paths.VoiceManifest = filepath.Join(t.TempDir(), "manifest.json")

	result = doctor.Run(newDoctorConfig(cfg, config.BackendNative), io.Discard)
	if !slices.ContainsFunc(result.Failures(), func(f string) bool { return strings.Contains(f, "voice manifest") }) {
		t.Errorf("missing voice manifest: failures = %v; want a voice manifest failure", result.Failures())
	}
}

// native-onnx synthesizes without the default voice, so doctor does not
// require it there.
func TestNewDoctorConfig_ONNXSkipsDefaultVoice(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Model = &modelcfg.ModelConfig{DefaultVoice: "juergen"}

	if got := newDoctorConfig(cfg, config.BackendNativeONNX).DefaultVoice; got != "" {
		t.Errorf("native-onnx DefaultVoice = %q; want empty", got)
	}

	if got := newDoctorConfig(cfg, config.BackendNative).DefaultVoice; got != "juergen" {
		t.Errorf("native DefaultVoice = %q; want juergen", got)
	}
}

// Only native-safetensors checks the checkpoint for the voice-cloning encoder;
// it is the backend that sets NativeModelPath.
func TestNewDoctorConfig_VoiceCloningCheckOnNative(t *testing.T) {
	cfg := config.DefaultConfig()

	if newDoctorConfig(cfg, config.BackendNative).CheckVoiceCloning == nil {
		t.Error("native: CheckVoiceCloning is nil; want the Mimi encoder check")
	}

	if newDoctorConfig(cfg, config.BackendNativeONNX).CheckVoiceCloning != nil {
		t.Error("native-onnx: CheckVoiceCloning is set; want nil")
	}
}
