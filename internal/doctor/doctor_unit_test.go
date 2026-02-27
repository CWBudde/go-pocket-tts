package doctor

import (
	"testing"
)

func TestParseMajorMinor(t *testing.T) {
	tests := []struct {
		name      string
		ver       string
		wantMajor int
		wantMinor int
		wantErr   bool
	}{
		{"simple", "3.11", 3, 11, false},
		{"with patch", "3.11.4", 3, 11, false},
		{"python2", "2.7.18", 2, 7, false},
		{"single number", "3", 0, 0, true},
		{"empty", "", 0, 0, true},
		{"bad major", "abc.11", 0, 0, true},
		{"bad minor", "3.xyz", 0, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			major, minor, err := parseMajorMinor(tt.ver)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseMajorMinor(%q) = (%d,%d,nil); want error", tt.ver, major, minor)
				}

				return
			}

			if err != nil {
				t.Fatalf("parseMajorMinor(%q) error: %v", tt.ver, err)
			}

			if major != tt.wantMajor || minor != tt.wantMinor {
				t.Fatalf("parseMajorMinor(%q) = (%d,%d); want (%d,%d)",
					tt.ver, major, minor, tt.wantMajor, tt.wantMinor)
			}
		})
	}
}

func TestCheckPythonVersion(t *testing.T) {
	tests := []struct {
		name    string
		ver     string
		wantErr bool
	}{
		{"3.10 ok", "3.10.0", false},
		{"3.11 ok", "3.11.4", false},
		{"3.14 ok", "3.14.0", false},
		{"too old", "3.9.1", true},
		{"too new", "3.15.0", true},
		{"python2", "2.7.18", true},
		{"not python", "abc", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkPythonVersion(tt.ver)
			if (err != nil) != tt.wantErr {
				t.Fatalf("checkPythonVersion(%q) = %v; wantErr=%v", tt.ver, err, tt.wantErr)
			}
		})
	}
}
