package model

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-pocket-tts/internal/config"
)

func TestResolveVoiceTarget(t *testing.T) {
	tests := []struct {
		name          string
		language      string
		voiceManifest string // paths.voice_manifest; "" keeps the language default
		outDir        string
		outDirSet     bool
		want          VoiceTarget
	}{
		{
			name: "default language keeps the tracked flat manifest", language: config.DefaultLanguage, outDir: "voices",
			want: VoiceTarget{Language: config.DefaultLanguage, Dir: "voices", Manifest: filepath.Join("voices", "manifest.json")},
		},
		{
			name: "other language gets voices/<lang>", language: "german", outDir: "voices",
			want: VoiceTarget{
				Language: "german", Dir: filepath.Join("voices", "german"),
				Manifest: filepath.Join("voices", "german", "manifest.json"), WriteIndex: true,
			},
		},
		{
			name: "configured voice manifest is the destination", language: "german", voiceManifest: "/data/de/stimmen.json", outDir: "voices",
			want: VoiceTarget{Language: "german", Dir: "/data/de", Manifest: "/data/de/stimmen.json", WriteIndex: true},
		},
		{
			name: "configured manifest for the flat language gets an index too", language: config.DefaultLanguage,
			voiceManifest: "/data/en/manifest.json", outDir: "voices",
			want: VoiceTarget{Language: config.DefaultLanguage, Dir: "/data/en", Manifest: "/data/en/manifest.json", WriteIndex: true},
		},
		{
			name: "explicit out-dir wins", language: "german", voiceManifest: "/data/de/stimmen.json", outDir: "/tmp/v", outDirSet: true,
			want: VoiceTarget{Language: "german", Dir: "/tmp/v", Manifest: filepath.Join("/tmp/v", "manifest.json"), WriteIndex: true},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			cfg.TTS.Language = tc.language

			cfg.Paths.VoiceManifest = config.PathsForLanguage(tc.language).VoiceManifest
			if tc.voiceManifest != "" {
				cfg.Paths.VoiceManifest = tc.voiceManifest
			}

			got, err := ResolveVoiceTarget(cfg, tc.outDir, tc.outDirSet)
			if err != nil {
				t.Fatal(err)
			}

			if got != tc.want {
				t.Errorf("ResolveVoiceTarget = %+v; want %+v", got, tc.want)
			}
		})
	}
}

func TestResolveVoiceTarget_CustomModelConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TTS.ModelConfigPath = "custom.yaml"

	_, err := ResolveVoiceTarget(cfg, "voices", false)
	if err == nil {
		t.Error("ResolveVoiceTarget with --model-config = nil error; want error")
	}
}

// A voice that is already present with the pinned checksum is not fetched
// again, so this runs offline: it checks that DownloadVoices records the voice
// in the lock file and in the voice manifest the runtime reads.
func TestDownloadVoices_RecordsLockAndIndex(t *testing.T) {
	src := filepath.Join("..", "..", "voices", "german", "juergen.safetensors")

	data, err := os.ReadFile(src)
	if err != nil {
		t.Skipf("german voice not downloaded (%v); run pockettts model download --language german", err)
	}

	dir := t.TempDir()

	err = os.WriteFile(filepath.Join(dir, "juergen.safetensors"), data, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	target := VoiceTarget{Language: "german", Dir: dir, Manifest: filepath.Join(dir, "manifest.json"), WriteIndex: true}

	err = DownloadVoices(target, []string{"juergen"}, io.Discard)
	if err != nil {
		t.Fatalf("DownloadVoices: %v", err)
	}

	var index voiceIndex
	readJSON(t, target.Manifest, &index)

	if len(index.Voices) != 1 || index.Voices[0].ID != "juergen" || index.Voices[0].Path != "juergen.safetensors" {
		t.Errorf("voice manifest = %+v; want only juergen -> juergen.safetensors", index.Voices)
	}

	var lock lockManifest
	readJSON(t, filepath.Join(dir, "download-manifest.lock.json"), &lock)

	rec, ok := lock.Files["languages/german/embeddings/juergen.safetensors"]
	if !ok || rec.Repo != VoiceRepo || rec.SHA256 == "" {
		t.Errorf("lock files = %+v; want a %s record for languages/german/embeddings/juergen.safetensors", lock.Files, VoiceRepo)
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	err = json.Unmarshal(data, v)
	if err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}
