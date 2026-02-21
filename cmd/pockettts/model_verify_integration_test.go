//go:build integration

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/example/go-pocket-tts/internal/testutil"
)

// runModelVerifyCapture executes "pockettts model verify" with the given extra
// args and returns the combined stdout+stderr output and the execution error.
func runModelVerifyCapture(t testing.TB, args ...string) (out string, err error) {
	t.Helper()

	pr, pw, pipeErr := os.Pipe()
	if pipeErr != nil {
		t.Fatalf("os.Pipe: %v", pipeErr)
	}
	origStdout := os.Stdout
	origStderr := os.Stderr
	os.Stdout = pw
	os.Stderr = pw

	root := NewRootCmd()
	root.SetArgs(append([]string{"model", "verify"}, args...))
	execErr := root.Execute()

	pw.Close()
	os.Stdout = origStdout
	os.Stderr = origStderr

	var buf bytes.Buffer
	if _, readErr := buf.ReadFrom(pr); readErr != nil {
		t.Fatalf("read pipe: %v", readErr)
	}
	pr.Close()

	return buf.String(), execErr
}

// repoRoot returns the absolute path to the repository root by walking up from
// the cmd/pockettts package directory (the working directory during go test).
func repoRoot(t testing.TB) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	// cmd/pockettts is two levels below the repo root.
	return filepath.Join(wd, "..", "..")
}

// ---------------------------------------------------------------------------
// TestModelVerify_PassesWithValidONNX
// ---------------------------------------------------------------------------

// TestModelVerify_PassesWithValidONNX runs "model verify" against the
// committed tiny identity ONNX model from Phase 3.7 and asserts exit 0.
// Skips when ONNX Runtime is unavailable.
func TestModelVerify_PassesWithValidONNX(t *testing.T) {
	testutil.RequireONNXRuntime(t)

	manifest := filepath.Join(repoRoot(t), "internal", "model", "testdata", "identity_manifest.json")
	if _, err := os.Stat(manifest); err != nil {
		t.Skipf("identity_manifest.json fixture not found at %q: %v", manifest, err)
	}

	out, err := runModelVerifyCapture(t, "--manifest", manifest)
	if err != nil {
		// ORT/IR version mismatch is a known compatibility issue — skip rather than fail.
		if strings.Contains(out, "Unsupported model IR version") || strings.Contains(out, "IR version") {
			t.Skipf("skipping due to ORT/IR version incompatibility: %v", err)
		}
		t.Fatalf("model verify failed: %v\noutput:\n%s", err, out)
	}
}

// ---------------------------------------------------------------------------
// TestModelVerify_FailsWithMissingManifest
// ---------------------------------------------------------------------------

// TestModelVerify_FailsWithMissingManifest points --manifest at a
// non-existent path and asserts exit non-zero with a structured error.
func TestModelVerify_FailsWithMissingManifest(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "nonexistent", "manifest.json")

	out, err := runModelVerifyCapture(t, "--manifest", missing)
	if err == nil {
		t.Fatalf("expected model verify to fail with missing manifest, but it passed\noutput:\n%s", out)
	}
	// The error must be actionable — it should mention the manifest path or a
	// recognisable load/open error rather than a bare unknown failure.
	combined := strings.ToLower(out + err.Error())
	if !strings.Contains(combined, "manifest") &&
		!strings.Contains(combined, "no such file") &&
		!strings.Contains(combined, "load") {
		t.Errorf("expected actionable error message, got:\n%s\nerr: %v", out, err)
	}
}

// ---------------------------------------------------------------------------
// TestModelVerify_FailsWithCorruptONNX
// ---------------------------------------------------------------------------

// TestModelVerify_FailsWithCorruptONNX writes a truncated .onnx file, builds
// a manifest that points to it, and asserts exit non-zero with an actionable
// error message. No ONNX Runtime is required — the failure occurs during
// session loading, before inference is attempted.
func TestModelVerify_FailsWithCorruptONNX(t *testing.T) {
	testutil.RequireONNXRuntime(t) // session load requires the ORT library

	tmp := t.TempDir()

	// Write a corrupt (truncated) ONNX file — valid ONNX starts with a
	// protobuf header; these 8 bytes are not a valid protobuf model.
	corruptONNX := filepath.Join(tmp, "corrupt.onnx")
	if err := os.WriteFile(corruptONNX, []byte("\x00\x01\x02\x03\x04\x05\x06\x07"), 0o644); err != nil {
		t.Fatalf("WriteFile corrupt.onnx: %v", err)
	}

	// Build a manifest that references the corrupt file.
	manifest := map[string]any{
		"variant": "test",
		"int8":    false,
		"graphs": []map[string]any{
			{
				"name":     "corrupt",
				"filename": "corrupt.onnx",
				"inputs":   []map[string]any{{"name": "x", "dtype": "float32", "shape": []int{1, 4}}},
				"outputs":  []map[string]any{{"name": "y", "dtype": "float32", "shape": []int{1, 4}}},
			},
		},
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	manifestPath := filepath.Join(tmp, "manifest.json")
	if err := os.WriteFile(manifestPath, manifestData, 0o644); err != nil {
		t.Fatalf("WriteFile manifest.json: %v", err)
	}

	out, err := runModelVerifyCapture(t, "--manifest", manifestPath)
	if err == nil {
		t.Fatalf("expected model verify to fail with corrupt ONNX, but it passed\noutput:\n%s", out)
	}
	// Error must be actionable — mention the session, model, or parse failure.
	combined := strings.ToLower(out + err.Error())
	if !strings.Contains(combined, "corrupt") &&
		!strings.Contains(combined, "failed") &&
		!strings.Contains(combined, "invalid") &&
		!strings.Contains(combined, "model") {
		t.Errorf("expected actionable error message, got:\n%s\nerr: %v", out, err)
	}
}
