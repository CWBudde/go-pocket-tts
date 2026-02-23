package tts

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/example/go-pocket-tts/internal/config"
	textpkg "github.com/example/go-pocket-tts/internal/text"
)

var ErrBackendNotImplemented = errors.New("tts backend is not implemented")

const (
	ParityStatusOK      = "ok"
	ParityStatusSkipped = "skipped"
	ParityStatusError   = "error"
)

type ParitySnapshot struct {
	Backend       string  `json:"backend"`
	Seed          int64   `json:"seed"`
	TokenCount    int     `json:"token_count"`
	ChunkCount    int     `json:"chunk_count"`
	SampleCount   int     `json:"sample_count"`
	PeakAbs       float64 `json:"peak_abs"`
	RMS           float64 `json:"rms"`
	PCMHashSHA256 string  `json:"pcm_hash_sha256"`
	Status        string  `json:"status"`
	Reason        string  `json:"reason,omitempty"`
}

type ServiceFactory func(backend string) (*Service, error)

// NewServiceFactory builds services for parity runs from config.
func NewServiceFactory(cfg config.Config) ServiceFactory {
	return func(backend string) (*Service, error) {
		normalized, err := config.NormalizeBackend(backend)
		if err != nil {
			return nil, err
		}

		switch normalized {
		case config.BackendNative, config.BackendNativeSafetensors:
			next := cfg
			next.TTS.Backend = normalized
			return NewService(next)
		default:
			return nil, fmt.Errorf("unsupported backend %q for parity harness", normalized)
		}
	}
}

// RunParityCase executes the same text/voice input on each backend and returns
// comparable snapshots. Backends that are not yet implemented are marked as
// skipped.
func RunParityCase(factory ServiceFactory, backends []string, input, voicePath string, seed int64) ([]ParitySnapshot, error) {
	if factory == nil {
		return nil, fmt.Errorf("parity service factory is required")
	}
	if len(backends) == 0 {
		return nil, fmt.Errorf("at least one backend is required")
	}

	results := make([]ParitySnapshot, 0, len(backends))
	for _, backend := range backends {
		snap := ParitySnapshot{
			Backend: backend,
			Seed:    seed,
		}

		svc, err := factory(backend)
		if err != nil {
			if errors.Is(err, ErrBackendNotImplemented) {
				snap.Status = ParityStatusSkipped
				snap.Reason = err.Error()
				results = append(results, snap)
				continue
			}
			snap.Status = ParityStatusError
			snap.Reason = fmt.Sprintf("create service: %v", err)
			results = append(results, snap)
			continue
		}

		chunks, err := textpkg.PrepareChunks(input, svc.tokenizer, maxTokensPerChunk)
		if err != nil {
			svc.Close()
			snap.Status = ParityStatusError
			snap.Reason = fmt.Sprintf("prepare chunks: %v", err)
			results = append(results, snap)
			continue
		}
		snap.ChunkCount = len(chunks)
		for _, c := range chunks {
			snap.TokenCount += len(c.TokenIDs)
		}

		pcm, err := svc.Synthesize(input, voicePath)
		svc.Close()
		if err != nil {
			snap.Status = ParityStatusError
			snap.Reason = fmt.Sprintf("synthesize: %v", err)
			results = append(results, snap)
			continue
		}

		snap.SampleCount = len(pcm)
		snap.PeakAbs = peakAbs(pcm)
		snap.RMS = rms(pcm)
		snap.PCMHashSHA256 = hashPCM(pcm)
		snap.Status = ParityStatusOK
		results = append(results, snap)
	}

	return results, nil
}

func SaveParitySnapshots(path string, snapshots []ParitySnapshot) error {
	data, err := json.MarshalIndent(snapshots, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal parity snapshots: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write parity snapshots: %w", err)
	}
	return nil
}

func LoadParitySnapshots(path string) ([]ParitySnapshot, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read parity snapshots: %w", err)
	}
	var snapshots []ParitySnapshot
	if err := json.Unmarshal(data, &snapshots); err != nil {
		return nil, fmt.Errorf("decode parity snapshots: %w", err)
	}
	return snapshots, nil
}

func peakAbs(samples []float32) float64 {
	var peak float64
	for _, s := range samples {
		v := math.Abs(float64(s))
		if v > peak {
			peak = v
		}
	}
	return peak
}

func rms(samples []float32) float64 {
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, s := range samples {
		v := float64(s)
		sum += v * v
	}
	return math.Sqrt(sum / float64(len(samples)))
}

func hashPCM(samples []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, s := range samples {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(s))
		_, _ = h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
