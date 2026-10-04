// Command gen writes the web app's language catalog (web/languages.json).
//
// Run it via `go generate ./internal/webmanifest/`.
package main

import (
	"flag"
	"log"
	"os"

	"github.com/cwbudde/go-pocket-tts/internal/webmanifest"
)

func main() {
	out := flag.String("out", "web/languages.json", "output path")

	flag.Parse()

	data, err := webmanifest.JSON()
	if err != nil {
		log.Fatal(err)
	}

	err = os.WriteFile(*out, data, 0o600)
	if err != nil {
		log.Fatal(err)
	}
}
