# Integration von Pocket TTS in Go

## Executive Summary

Pocket TTS (Kyutai) ist ein CPU‑optimiertes Text‑to‑Speech‑System mit Voice Cloning, das als Python‑Paket (PyTorch) verfügbar ist und von Haus aus sowohl eine CLI als auch einen lokalen HTTP‑Server bereitstellt. Offiziell liegt der Fokus auf CPU‑Echtzeitfähigkeit (u. a. ~200 ms bis zum ersten Audioblock und ~6× schneller als Echtzeit auf einem MacBook Air M4; Nutzung von nur 2 CPU‑Kernen). citeturn13view0

Für Go gibt es mehrere realistische Integrationspfade – von „einfach aus Go starten“ bis hin zu „native Inference ohne Python“. Die wichtigsten Rahmenbedingungen, die die Integrationsarchitektur stark beeinflussen:

- **Lizenz & Zugang zu Modellartefakten:** Das **Python‑Repo ist MIT‑lizenziert**. citeturn6view0 Die **offiziellen Modellgewichte auf Hugging Face sind CC‑BY‑4.0** und **„gated“** (Zustimmung zur Weitergabe von Kontaktdaten/Prohibited‑Use‑Bedingungen erforderlich). citeturn13view0
- **Schnittstellen:** Offizielle CLI (`generate`, `serve`, `export-voice`) inkl. Streaming und Voice‑State‑Export in `.safetensors`. citeturn3view0turn1view1turn6view2
- **Voice‑Lizenzierung:** Die „tts‑voices“‑Sammlung enthält Voices aus verschiedenen Quellen mit teils **nicht‑kommerziellen** Lizenzen (z. B. Expresso/EARS: CC‑BY‑NC‑4.0). citeturn14view0

**Empfehlungen (zwei konkrete Wege):**

- **(A) Referenz-/„Just run it“-Methode:** Aus Go per `os/exec` die offizielle CLI `pocket-tts generate` starten, Text via **stdin** übergeben (`--text -`) und WAV via **stdout** abgreifen (`--output-path -`). Das ist die schnellste, stabilste, am besten dokumentierte Integration mit minimalem Go‑Code und maximaler Nähe zum Upstream. citeturn6view2turn3view0turn19view0
- **(B) Native‑Pfad ohne Python:** `gudrob/pocket-tts.cpp` (C++‑Port) per **C‑API + cgo** in Go einbinden. Diese Implementierung nutzt **ONNX Runtime**, bietet **INT8‑Inferenz** und **Streaming** und ist explizit für FFI gedacht. citeturn12view0  
  (Langfristig kann man daraus „mehr Go“ machen, indem man direkt die ONNX‑Modelle in Go mit ONNX Runtime nutzt – siehe Optionenanalyse.)

## Technische Ausgangslage: Pocket TTS Eigenschaften und Schnittstellen

Pocket TTS ist laut offizieller Model‑Card ein **100M‑Parameter** TTS‑Modell, ausgelegt auf CPU‑Inference inklusive **Audio‑Streaming** und **Voice Cloning**. citeturn13view0turn8search2 Relevante, integrationsnahe Eigenschaften:

- **Performance‑Claims (offiziell, Modellkarte):** „Low latency“ mit **~200 ms bis zum ersten Audio‑Chunk**, und **~6× Real‑Time** auf einem MacBook Air M4; außerdem **„uses only 2 CPU cores“**. citeturn13view0
- **Python‑Runtime‑Anforderungen:** `pocket-tts` verlangt **Python 3.10–<3.15** und **PyTorch ≥ 2.5.0**, plus u. a. `sentencepiece`, `safetensors`, `fastapi`, `uvicorn`. citeturn19view0turn13view0
- **CLI‑Verhalten:**
  - `generate` erzeugt standardmäßig **WAV 24 kHz, mono, 16‑bit PCM**. citeturn3view0
  - Der CLI‑Code unterstützt **Text über stdin**, wenn `--text -` gesetzt ist. citeturn6view2
  - Die Ausgabe kann auf **stdout** geschrieben werden (implizit, wenn `--output-path -` genutzt wird – im Code wird `output_path != "-"` geprüft). citeturn6view2
- **HTTP‑Server (`serve`)**: Implementiert FastAPI/uvicorn, liefert u. a. `GET /health` und `POST /tts` als **StreamingResponse** (`audio/wav`, chunked). citeturn6view2

Voice‑Cloning‑/Voice‑State‑Mechanik ist für Integration entscheidend:

- Das offizielle CLI `export-voice` konvertiert eine Referenz‑Audio‑Datei in eine **Voice‑Embedding/Model‑State‑Datei in `.safetensors`** (im Doc: „it’s actually the kvcache“), was das spätere Generieren deutlich beschleunigen kann. citeturn1view1turn13view0

Architektur-Skizze der offiziellen „Python‑Seite“ (vereinfacht, aus den öffentlich sichtbaren APIs abgeleitet):

```mermaid
flowchart LR
  A[Go-Anwendung] -->|os/exec| B[pocket-tts CLI]
  B -->|stdin: --text -| C[Python + PyTorch Runtime]
  C --> D[TTSModel / Streaming Generation]
  D -->|stdout: --output-path -| A

  A2[Go-Anwendung] -->|HTTP POST| S[pocket-tts serve (FastAPI)]
  S -->|/tts StreamingResponse audio/wav| A2
```

## Integrationsoptionen für Go und Trade-offs

### Kurzvergleich über Kernattribute

| Option                                                     | Ease of Use (PoC) | „Nativeness“ (ohne Python) |                Performance-Potenzial | Engineering‑Aufwand |
| ---------------------------------------------------------- | ----------------: | -------------------------: | -----------------------------------: | ------------------: |
| Go → **CLI subprocess** (`pocket-tts generate`)            |         Sehr hoch |                    Niedrig | Mittel (Cold‑Start), gut pro Request |             Niedrig |
| Go → **lokaler HTTP‑Server** (`pocket-tts serve` + `/tts`) |              Hoch |                    Niedrig |         Hoch (Warm‑Model, Streaming) |      Niedrig–Mittel |
| **Embedded CPython** (Go + libpython)                      |           Niedrig |                    Niedrig |                          Mittel–hoch |                Hoch |
| Go → **C++ ONNX Port** (`pocket-tts.cpp` via C‑API)        |            Mittel |                       Hoch |              Hoch (INT8 + Streaming) |              Mittel |
| Go → **direkt ONNX Runtime** (KevinAHM ONNX Export)        |            Mittel |                       Hoch |            Hoch (INT8/Thread‑Tuning) |                Hoch |
| Go → **Rust/Candle Port** (Sidecar oder FFI/WASM)          |            Mittel |                       Hoch |         Hoch (Quantization/Features) |         Mittel–Hoch |

Die folgenden Optionen decken die von dir gewünschten Integrationsrichtungen ab (Python‑Aufrufvarianten, FFI/cgo, ONNX/TFLite, C/C++‑Runtimes, WASM/WASI, Third‑Party‑Ports). Für jede Option sind die verlangten Dimensionen zusammengefasst.

### Option: Go ruft die offizielle CLI als Subprocess auf

**Implementationsschritte:** CLI installieren (z. B. `pip install pocket-tts` oder mit `uvx`/`uv`), dann aus Go `pocket-tts generate` starten. Text via `--text -` in stdin, Audio über `--output-path -` aus stdout lesen, und WAV weiterverarbeiten/speichern. citeturn13view0turn6view2turn3view0  
**Tooling/Sprachen:** Go + Python; PyTorch; keine Compiler/FFI nötig. Abhängigkeiten sind in `pyproject.toml` sichtbar (u. a. `torch`, `sentencepiece`, `safetensors`, `fastapi`). citeturn19view0  
**Performance‑Erwartung:** Sehr gut pro Inference _nach_ Model‑Load; aber **Cold‑Start teuer**, weil `generate` den Model‑Load pro Invocation macht (implizit aus CLI‑Design; `serve` wird explizit als schneller beschrieben, weil Modell im Speicher bleibt). citeturn13view0turn6view2  
**Cross‑Platform:** Solange Python + PyTorch Wheels verfügbar sind (macOS/Linux/Windows typischerweise ok). citeturn13view0turn19view0  
**Build/Deployment:** Einfach im Dev; in Prod braucht man reproduzierbare Python‑Umgebung (venv/uv/Docker). Das Projekt selbst definiert das CLI‑Entry‑Point `pocket-tts = pocket_tts.main:cli_app`. citeturn19view0turn6view2  
**Maintenance:** Niedrig, da Upstream‑CLI stabiler Vertrag; Updates folgen PyPI‑Releases (z. B. Version `1.1.1` in `pyproject.toml`). citeturn19view0  
**Lizenz:** Code MIT. citeturn6view0 Modellgewichte CC‑BY‑4.0 und „gated“ (Kontaktinfo/Prohibited‑Use akzeptieren). citeturn13view0  
**Security/Sandboxing:** Sehr gut isolierbar (separater Prozess). Risiko: wenn ihr untrusted Inputs in `--voice`/`hf://`/Remote‑URLs erlaubt, besteht Supply‑Chain/SSRF‑ähnliche Risikofläche (je nach Nutzung). citeturn3view0turn13view0  
**Feasibility‑Risiken:** Gering. Größtes Risiko ist Packaging/Model‑Download in Zielumgebungen (gated HF, Offline‑Betrieb, Cache‑Pflege).

### Option: Go nutzt den offiziellen lokalen HTTP‑Server (`serve`) über `/tts`

**Implementationsschritte:** `pocket-tts serve` als Sidecar/Daemon starten; Go sendet HTTP POST an `/tts`. Der Server streamt WAV. citeturn6view2turn13view0  
**Tooling/Sprachen:** Go + Python/uvicorn/FastAPI. citeturn19view0turn6view2  
**Performance:** Sehr gut für wiederholte Requests, weil Modell resident im Speicher bleibt (explizit auf Model‑Card). citeturn13view0turn6view2  
**Cross‑Platform:** Gut; Netzwerk/Ports müssen erlaubt sein. citeturn6view2  
**Build/Deployment:** Sidecar‑Pattern, Docker‑Pattern oder Systemd‑Service; mehr Ops‑Komplexität als CLI‑Einmalaufruf.  
**Maintenance:** Mittel (ihr betreibt eine Service‑Komponente).  
**Lizenz:** wie oben (MIT Code, CC‑BY‑Model). citeturn6view0turn13view0  
**Security/Sandboxing:** Größeres Angriffsprofil als CLI, weil HTTP‑Exposition. Außerdem erlaubt der Server in `/tts` einen `voice_url`, der u. a. `http://`, `https://`, `hf://` akzeptiert (und Predefined Voices). Das ist funktional hilfreich, aber – bei untrusted Nutzern – ein **SSRF‑ähnliches** Risiko. citeturn6view2  
**Feasibility‑Risiken:** Niedrig bis mittel; typische Risiken sind Port‑Konflikte, Lifecycle, Log‑/Health‑Checks, Netzwerkpolicies.

### Option: Go integriert über eine OpenAI‑kompatible API (Third‑Party Server)

**Implementationsschritte:** Statt `/tts` direkt nutzt man einen Community‑Server mit OpenAI‑kompatiblem Endpoint (z. B. „OpenAI‑compatible streaming server, dockerized“). citeturn13view0turn9search18  
**Tooling/Sprachen:** Meist Python‑Server; Go spricht standardisierte HTTP‑API, ggf. vorhandene OpenAI‑Client‑Lib in Go. citeturn9search18turn9search31  
**Performance:** Gut, ähnliche Vorteile wie `serve` (Warm‑Model, Streaming).  
**Cross‑Platform:** Gut, solange Container/Runtime läuft.  
**Build/Deployment:** Kann einfacher sein, wenn ihr ohnehin OpenAI‑TTS‑Semantik standardisieren wollt; aber zusätzlicher Layer.  
**Maintenance:** Mittel–hoch (Third‑Party‑Projekt, Abweichung von Upstream).  
**Lizenz:** Abhängig vom Server‑Repo; Modell bleibt CC‑BY‑4.0. citeturn13view0turn9search18  
**Security:** Positiv: Einige Third‑Party Server adressieren konkret SSRF‑Gefahren (z. B. Commit/Notiz „block HTTP/HTTPS URLs to prevent SSRF attacks“). citeturn9search18  
**Feasibility‑Risiko:** Mittel (Projektaktivität, Semantik‑Drift zu Upstream).

### Option: Python‑Integration „in‑process“ durch Embedded CPython

**Implementationsschritte:** Go bindet `libpython` ein (cgo) und ruft `pocket_tts` APIs direkt. Das reduziert Prozess‑Overhead, bleibt aber Python/PyTorch‑gebunden.  
**Tooling/Sprachen:** Go + Cgo + Python/C‑API; Python‑Header/ABI‑Management. Hinweis: Pocket‑TTS nutzt `uv`‑Managed Pythons u. a. wegen möglicher fehlender Header in System‑Python („python-preference = only-managed“). citeturn19view0  
**Performance:** Potenziell besser als Subprocess (kein Prozessstart), aber PyTorch/Python‑Overhead bleibt.  
**Cross‑Platform:** Schwierig, weil `libpython`‑ABI und Wheel‑Struktur pro OS/Arch; Windows besonders unangenehm.  
**Build/Deployment:** Hochkomplex (reproduzierbare Python‑Distribution + native Linking).  
**Maintenance:** Hoch (Debugging GIL/Interop, Python‑Upgrades).  
**Lizenz:** wie oben (Model CC‑BY, „gated“). citeturn13view0  
**Security:** Weniger isoliert (gleicher Prozess). Sandboxing schwerer.  
**Feasibility‑Risiko:** Hoch – meist nur sinnvoll, wenn „single‑binary“ keine harte Vorgabe ist, aber Latenz extrem kritisch und Sidecar unerwünscht.

### Option: C++‑Port via ONNX Runtime + C‑API (FFI/cgo)

Hier ist `gudrob/pocket-tts.cpp` besonders relevant, weil es bereits ein **C‑API** für FFI anbietet.

**Implementationsschritte:** Repo klonen, Modelle herunterladen (`download_models.sh`), via CMake bauen, dann aus Go über cgo gegen die `pocket_tts_c.h` API linken. citeturn12view0  
**Tooling/Sprachen:** Go + cgo + C/C++; zusätzliche native Dependencies: ONNX Runtime, SentencePiece, libsndfile, libsamplerate, CMake (je nach OS via Homebrew/apt/vcpkg). citeturn12view0  
**Performance:** Laut Repo: **~3× Real‑Time auf Apple Silicon (INT8)**, Modellgröße **~200 MB**, und Streaming‑Support. citeturn12view0  
**Cross‑Platform:** Explizit beschrieben für macOS/Linux/Windows; Windows Setup via vcpkg. citeturn12view0  
**Build/Deployment:** Mittel–hoch (native Toolchains, Shared Libraries). Auf der positiven Seite: keine Python/PyTorch‑Distribution mehr.  
**Maintenance:** Mittel (Upstream‑Drift zu Pocket‑TTS‑Python möglich; ORT Updates).  
**Lizenz:** laut Repo: Code MIT, Modelle CC‑BY‑4.0 (über kyutai/pocket‑tts). citeturn12view0turn13view0  
**Security/Sandboxing:** Besser als Python‑Server im gleichen Prozess, weil kein dynamischer Python‑Import; aber ihr linkt native Libs. Sandboxing weiterhin möglich (Container, seccomp), aber weniger „outsourced“.  
**Feasibility‑Risiko:** Mittel (Abhängigkeit von Community‑Port; Build‑Stabilität; ABI‑Kompatibilitäten).

### Option: Direkte ONNX‑Integration in Go (ohne C++‑Port)

Diese Option ist „Go‑port‑artig“, weil ihr den TTS‑Loop selbst in Go implementiert, aber die Tensor‑Compute‑Engine (ONNX Runtime) bleibt native.

Grundlage ist z. B. **KevinAHM/pocket-tts-onnx**, das die Architektur bereits in mehrere ONNX‑Submodelle splittet und Performance‑/Thread‑Tuning dokumentiert: Flow LM aufgeteilt in `flow_lm_main` (Transformer) und `flow_lm_flow` (Flow‑Net), plus `mimi_decoder`, `mimi_encoder`, `text_conditioner`, inklusive INT8‑Varianten und RTF‑Benchmarks (~4× Real‑Time für INT8 auf einem 16‑Core‑CPU, Modellgröße ~200 MB). citeturn16view0  
**Go‑Binding‑Routen:**

- Klassisch mit cgo: z. B. `onnxruntime_go` verlangt cgo und die passende ONNX Runtime Shared Library. citeturn15search16
- „Ohne cgo“ via purego: `onnxruntime-purego` lädt ONNX Runtime per Dynamic Loading, ist aber „unstable“/API‑Änderungen möglich. citeturn15search1  
  **Tokenisierung:** Pocket‑TTS nutzt SentencePiece (`sentencepiece>=0.2.1`), also braucht ihr in Go entweder eine kompatible SentencePiece‑Implementierung oder Bindings. citeturn19view0 Es gibt Go‑Bibliotheken, die `tokenizer.model` lesen können (z. B. `github.com/lwch/sentencepiece`), aber ihr müsst Kompatibilität gegen das Pocket‑TTS‑Tokenisierungsergebnis verifizieren. citeturn15search30  
  **Feasibility‑Risiko:** Mittel–hoch: technisch machbar (Modelle/Architektur liegen bereits in ONNX‑Splits vor), aber Sampling‑Loop, Zustandsmanagement und Streaming müssen sauber in Go nachgebaut werden. citeturn16view0

### Option: Rust/Candle‑Port (CLI/HTTP/WASM/FFI)

Es gibt mindestens zwei relevante Rust‑Ports:

- `try-coil/pocket-tts-rust` (Candle‑Port) bewirbt u. a. **int8 Quantization**, **Streaming**, **HTTP API (OpenAI‑kompatibel)** und **WebAssembly**. citeturn12view3
- `jamesfebin/pocket-tts-candle` als weiterer Candle‑Port mit CLI und Features (z. B. Metal), der explizit die Kompatibilität zur Original‑Architektur betont. citeturn12view2

Integration aus Go wäre möglich als:

- **Sidecar‑Prozess** (am einfachsten; ähnlich Option HTTP‑Server).
- **FFI** (wenn Port eine stabile C‑ABI anbietet / `cdylib`), oder
- **WASM** (z. B. über wasmtime/wazero) – starkes Sandboxing, aber potenziell Performance‑Nachteile / große Memory‑Footprints.

**Feasibility‑Risiko:** Mittel: Rust‑Build/Tooling kommt hinzu; dafür kein Python/PyTorch.

### Option: libtorch/TorchScript, TensorFlow Lite, tinygrad‑Ports

- **libtorch/TorchScript:** theoretisch möglich (PyTorch‑Modelle exportieren), aber Pocket‑TTS ist streaming/stateful; Export/Tracing kann schwierig werden. Zudem ist libtorch distribution‑seitig oft sehr schwergewichtig. (Kein hoher „Realismus“ gegenüber ONNX‑Pfad, der bereits Community‑Exports hat.)
- **TFLite:** In der Praxis hohes Konvertierungsrisiko (Operator‑Abdeckung, dynamische Shapes, Control‑Flow) und wenig direkte Anknüpfungspunkte in den verfügbaren Quellen.
- **tinygrad‑Ports:** derzeit eher Forschungs-/Hobby‑Pfad ohne belastbare, spezifische Pocket‑TTS‑Quellenbasis.

## Empfehlung A: Referenzintegration über CLI‑Subprocess

### Warum diese Empfehlung

Diese Methode maximiert **Stabilität** und **Upstream‑Nähe**: Ihr nutzt genau die offizielle Ausführungspipeline (`pocket_tts.main:generate`) und vermeidet Re‑Implementierungsfehler bei Tokenisierung/State/Decoder. Außerdem unterstützt die CLI explizit eine sehr integrationsfreundliche Pipe‑Semantik: `--text -` (stdin) und `--output-path -` (stdout). citeturn6view2turn19view0

### Schritt‑für‑Schritt Plan

**Installation & Voraussetzungen**

1. Python‑Voraussetzungen erfüllen (3.10–<3.15) und `pocket-tts` installieren (`pip install pocket-tts` oder per `uv`). citeturn19view0turn13view0
2. Falls ihr Voice‑Cloning‑Gewichte benötigt, müsst ihr die HF‑Bedingungen akzeptieren (gated) und ggf. authentifiziert sein. citeturn13view0turn11view0

**Go‑Integration (PoC)** 3. In Go: `exec.Command("pocket-tts", "generate", "--text", "-", "--output-path", "-", "--voice", "<voice>")` verwenden.  
4. Text in `cmd.StdinPipe()` schreiben; WAV‑Bytes aus `cmd.StdoutPipe()` lesen und entweder:

- direkt als Datei schreiben, oder
- WAV parsen (Header) und PCM weiterverarbeiten (Audio‑Playback, Streaming an Client).

5. Für Voice‑Wiederverwendung: einmalig `pocket-tts export-voice <audio> <voice.safetensors>` ausführen und danach `--voice voice.safetensors` nutzen (schneller als jedes Mal Audio prompt zu verarbeiten). citeturn1view1turn3view0turn6view2

**Betrieb & Skalierung** 6. Für wiederholte Requests:

- entweder Prozess‑Pool / Worker‑Pool (mehrere CLI‑Instanzen parallel), oder
- Wechsel zu `pocket-tts serve` (Option „Warm‑Server“), wenn niedrige Latenz bei vielen Requests entscheidend ist (Modell bleibt in Memory). citeturn13view0turn6view2

7. Caching/Offline: Lokalen Cache persistieren (Volume), da Modell/Voices typischerweise beim ersten Mal heruntergeladen werden und der offizielle Betrieb stark auf HF‑Artefakte verweist. citeturn13view0turn13view1

### Aufwandsschätzung

- **PoC (einfacher „Text rein → WAV raus“‑Aufruf):** ca. **4–8 Stunden**.
- **Produktionsreif (Caching, Worker‑Pool, Error‑Handling, observability, HF‑Auth‑Handling/Offline):** ca. **2–5 Tage**.

### Minimaler Proof‑of‑Concept‑Checkliste

- [ ] `pip install pocket-tts` erfolgreich; `pocket-tts generate` läuft lokal. citeturn13view0turn19view0
- [ ] Go startet `pocket-tts generate --text - --output-path -` und speichert WAV. citeturn6view2
- [ ] WAV‑Eigenschaften plausibel (24 kHz, mono, 16‑bit PCM). citeturn3view0
- [ ] Optional: `export-voice` generiert `.safetensors` und wird als `--voice` akzeptiert. citeturn1view1turn3view0
- [ ] Fehlerfälle: leere Eingabe (`--text -` ohne Inhalt) wird sauber behandelt (CLI bricht ab). citeturn6view2

## Empfehlung B: Native Integration via C++/ONNX C‑API

### Warum diese Empfehlung

Wenn das Ziel „**kein Python in Prod**“ ist, ist der pragmatischste und aktuell am besten „produktisierbare“ Pfad ein Port, der bereits:

- ohne Python läuft,
- eine stabile FFI‑Oberfläche bietet und
- Performance‑Optimierungen (INT8, Streaming) enthält.

`gudrob/pocket-tts.cpp` erfüllt genau diese Kriterien: **C++‑Inference ohne Python**, **C‑API für FFI**, **Audio‑Streaming**, und **INT8**‑Optionen mit ~3× Real‑Time auf Apple Silicon plus ~200 MB Modellgröße. citeturn12view0

### Zielarchitektur

```mermaid
flowchart LR
  G[Go Service] -->|cgo| C[C-ABI Wrapper]
  C --> L[libpocket_tts (C++ Port)]
  L --> ORT[ONNX Runtime]
  L --> SP[SentencePiece]
  L --> AUD[Audio I/O/Resampling libs]
  ORT --> CPU[CPU Inference]
  CPU --> L --> C --> G
```

### Schritt‑für‑Schritt Implementierungsplan

**Build & Abhängigkeiten**

1. `pocket-tts.cpp` klonen, Modelle herunterladen (`download_models.sh`), Build via CMake. citeturn12view0
2. Abhängigkeiten je OS installieren (Repo nennt explizit: onnxruntime, sentencepiece, libsndfile, libsamplerate, cmake; Windows via vcpkg). citeturn12view0
3. Shared Library builden (`-DBUILD_SHARED=ON`), damit Go sie linken kann. citeturn12view0

**Go‑Binding (cgo)** 4. cgo‑Wrapper erstellen, der die `pocket_tts_c.h` API nutzt. Die Readme zeigt Kernfunktionen:

- `pocket_tts_create`,
- `pocket_tts_encode_voice`,
- `pocket_tts_generate` (liefert `float[]` Samples, sample_rate = 24000),
- sowie `pocket_tts_generate_streaming` mit Callback. citeturn12view0

5. Speicherverwaltung sauber kapseln: Result‑Buffer aus C in Go kopieren und mit WAV‑Writer serialisieren; danach `pocket_tts_free_audio`, `pocket_tts_free_voice`, `pocket_tts_destroy`. citeturn12view0

**Streaming (optional im PoC, aber wichtig in Echtzeit‑Usecases)** 6. Für Streaming: Callback‑Brücke von C nach Go (cgo `//export`‑Pattern) oder erst mal „offline generate“ im PoC. (Streaming ist technisch machbar, aber cgo‑Callbacks erfordern saubere Thread‑/CGO‑Regeln.)

**Deployment** 7. Deployment‑Packaging:

- Shared Libs (libpocket_tts + ORT + SentencePiece etc.) in euer Artefakt/Container bundle.
- Version‑Pinning für ORT und Modellartefakte (um Reproduzierbarkeit zu sichern).

### Aufwandsschätzung

- **PoC (offline generate, ohne Streaming‑Callback, lokale Builds):** ca. **2–4 Tage**.
- **Produktionsreif (Streaming‑Callback, Cross‑Platform CI, sauberes Bundling der nativen Libs, Security Hardening):** ca. **2–4 Wochen**.

### Minimaler Proof‑of‑Concept‑Checkliste

- [ ] `pocket-tts.cpp` lokal gebaut, Demo‑CLI erzeugt WAV. citeturn12view0
- [ ] Go ruft `pocket_tts_create`/`pocket_tts_generate` auf und schreibt WAV mit 24 kHz. citeturn12view0
- [ ] Voice‑Encoding funktioniert (Reference Audio rein → VoiceHandle). citeturn12view0
- [ ] Modellpfade konfigurierbar (z. B. `models_dir`, `tokenizer`). citeturn12view0
- [ ] Lizenzhinweise dokumentiert (MIT Code, CC‑BY‑Modelle) in eurem Repo/Artefakt. citeturn12view0turn13view0turn6view0

## Lizenz, Security und Betriebsrisiken

### Lizenzierung und erlaubte Nutzung

- **Code‑Lizenz (Upstream):** Pocket‑TTS‑Repository ist MIT‑lizenziert. citeturn6view0
- **Modellgewichte:** Offizielle Gewichte auf Hugging Face sind **CC‑BY‑4.0**. citeturn13view0 Das bedeutet i. d. R. Attribution‑Pflicht (und ggf. weitere CC‑BY‑Anforderungen in eurer Distribution/Docs).
- **Gated Access & Prohibited Use:** Das offizielle Modell verlangt die Zustimmung zu Bedingungen, u. a. Verbot von Voice‑Impersonation ohne rechtmäßige Zustimmung, Desinformation etc. citeturn13view0
- **Voices‑Repository:** Hier liegt ein häufiger Stolperstein für kommerzielle Nutzung:
  - `voice-donations/` ist CC0,
  - `vctk/` ist CC‑BY‑4.0,
  - `expresso/` und `ears/` sind **CC‑BY‑NC‑4.0 (nur nicht‑kommerziell)**,
  - weitere Subsets haben eigene Hinweise. citeturn14view0  
    Für Produkt‑Usecases müsst ihr daher sehr bewusst auswählen, welche Voice‑Prompts ihr shippt/als Defaults anbietet.

### Security/Sandboxing

- **Subprocess‑Ansatz (Empfehlung A):** Sehr gut sandboxbar (Container, seccomp, AppArmor, cgroups). Außerdem verhindert Prozessisolation, dass ein Fehler in PyTorch/Python direkt den Go‑Prozess kompromittiert (trotzdem: gleiche Host‑Ressource).
- **HTTP‑Server (`serve`):** Das FastAPI‑Endpoint `/tts` akzeptiert `voice_url`, wenn es u. a. mit `http://`, `https://` oder `hf://` anfängt. Das ist funktional, kann aber – wenn untrusted Nutzer Parameter setzen – in Richtung SSRF/unerwünschtes Fetching gehen. citeturn6view2  
  Praktische Härtung: untrusted `voice_url` komplett deaktivieren, nur Whitelist von Predefined Voices erlauben, oder Netzwerkzugriffe des Sidecars per Policy/Firewall unterbinden.
- **Third‑party OpenAI‑Server:** Einige Projekte adressieren das konkret und blockieren HTTP/HTTPS‑Voice‑URLs explizit zur SSRF‑Prävention. citeturn9search18

### Feasibility‑/Wartungsrisiken

- **Upstream‑Dynamik:** Pocket‑TTS ist 2026er Release und hat sichtbare Iteration (z. B. CLI‑stdin‑Support, „reduced memory usage“, local model loading). citeturn4view0turn6view2
- **Native Pfade sind Community‑getrieben:** ONNX/C++/Rust‑Ports können schneller sein (INT8 etc.), aber ihr tragt das Risiko von „Projekt driftet ab / wird nicht gepflegt“.
- **Tokenisierung & Pipeline‑Parity:** Für einen echten Go‑Port (direkter ORT‑Loop) ist die größte Risikoquelle **exakte Reproduktion** der Tokenisierung und State‑Semantik. Dass Kyutai selbst WebAssembly/ONNX als „nicht offiziell supported, aber community implementations exist“ gelistet hat, zeigt: machbar, aber nicht trivial. citeturn13view0turn16view0
