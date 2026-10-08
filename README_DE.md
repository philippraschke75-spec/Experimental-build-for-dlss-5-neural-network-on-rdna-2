# DLSS 5 Neural Rendering auf AMD-RX-6000-Karten (experimentell)

**English:** [README.md](README.md)

![Beispiel im Spiel](images/ingame2.jpg)

## Was ist das?

NVIDIA hat eine neue Grafik-Funktion gebaut: **DLSS Neural Rendering** („DLSS 5“). Damit sehen Spiele
realistischer aus: besseres Licht, mehr Details, schönere Farben. Normalerweise geht das **nur mit
NVIDIA-Grafikkarten**.

Diese Mod bringt es auf **AMD-Radeon-RX-6000-Karten** (RDNA2).

> **Experimentell.** Kann abstürzen, falsch aussehen oder viele FPS kosten. Nicht in Online-Spielen
> mit Anti-Cheat benutzen.

## Geht das auf meinem PC?

Du brauchst **alles** davon:

| | |
|---|---|
| ✅ Grafikkarte | AMD **RX-6000-Serie**, siehe Tabelle unten. RX 7000/9000, NVIDIA und Intel gehen **nicht**. |
| ✅ Windows | Windows 10 oder 11, 64-Bit |
| ✅ AMD-Treiber | Ein aktueller AMD-Adrenalin-Treiber ([Download](https://www.amd.com/de/support/download/drivers.html)) |
| ✅ Spiel | Ein 64-Bit-Spiel mit DirectX 11 oder 12, in dem du in den Grafikeinstellungen **DLSS, FSR oder XeSS** auswählen kannst |
| ✅ NVIDIA-Modelldatei | `nvngx_dlssnr.dll`. **Nicht dabei**, die musst du dir selbst besorgen (Schritt 2) |

| Grafikkarte | Status | Zeit pro Modell-Frame |
|---|---|---|
| RX 6800 / 6800 XT / 6900 XT / 6950 XT | ✅ getestet | ~40 ms |
| RX 6700 / 6700 XT / 6750 XT | 🧪 sollte gehen, **ungetestet** (v0.2-beta) | ~80 ms (geschätzt) |
| RX 6600 / 6600 XT / 6650 XT | 🧪 sollte gehen, **ungetestet** (v0.2-beta) | ~110 ms (geschätzt) |
| RX 6500 XT / 6400 | 🧪 sollte gehen, sehr langsam | 250 ms+ |

**Du hast eine RX 6700 oder RX 6600?** Probier bitte v0.2-beta aus und schreib in [Issues](../../issues),
ob es geht (mit deiner `dlssnr_nr.log`).

Du weißt nicht, welche Grafikkarte du hast? Drück `Strg + Umschalt + Esc`, klick auf **Leistung**, dann
auf **GPU**. Der Name steht oben rechts.

## Installation

### Schritt 1: Herunterladen

Geh zu [**Releases**](../../releases/latest) und lade die ZIP (`DLSS5-RDNA2-Experimental-...zip`) herunter. RX 6700 / 6600 / 6500: nimm **v0.2-beta**.

### Schritt 2: NVIDIA-Modelldatei besorgen

Diese Mod enthält **keine** NVIDIA-Dateien. Du brauchst die Datei **`nvngx_dlssnr.dll`** (ca. 160 MB).
Wo du sie bekommst, steht in janblades Anleitung:
[Choose the correct runtime](https://github.com/janblade/OptiScaler-DLSSNR-PreSR-Multipass/blob/main/INSTALL-DLSSNR.md#choose-the-correct-runtime).

### Schritt 3: Spielordner finden

Das ist der Ordner, in dem die `.exe` des Spiels liegt.

- **Steam:** Rechtsklick aufs Spiel → **Verwalten** → **Lokale Dateien durchsuchen**
- **Epic / GOG / andere:** Rechtsklick aufs Spiel → Installationsordner öffnen

Gibt es dort einen Ordner `bin` oder `Binaries\Win64` mit der `.exe` drin, nimm den.

### Schritt 4: Dateien kopieren

1. Öffne die ZIP aus Schritt 1.
2. Kopiere **alles** daraus in den Spielordner. Wenn Windows fragt: **Ersetzen**.
3. Kopiere `nvngx_dlssnr.dll` aus Schritt 2 in denselben Spielordner.

### Schritt 5: Setup starten

Doppelklick im Spielordner auf **`setup_windows.bat`**. Ein schwarzes Fenster stellt ein paar Fragen:

| Frage | Eingabe |
|---|---|
| „Do you want to delete these files?“ (nur wenn ein altes OptiScaler da ist) | `1` + Enter |
| „Choose a filename for OptiScaler“ | nur Enter (= `dxgi.dll`) |
| „dxgi.dll already exists … overwrite?“ (du nutzt ReShade o. Ä.) | `2` + Enter, dann `2` (`winmm.dll`) |
| „Are you using an Nvidia GPU or AMD/Intel GPU?“ | `1` + Enter (AMD) |

Wenn das Fenster fertig ist, schließ es.

### Schritt 6: Spiel starten

1. In den Grafikeinstellungen des Spiels **DLSS**, **FSR** oder **XeSS** einschalten (was das Spiel halt hat).
2. Im Spiel die Taste **`Einfg`** (Insert) drücken. Das OptiScaler-Menü geht auf.
3. **DLSS Neural Rendering** aufklappen, Haken bei **Enable Neural Rendering** prüfen.
4. Nochmal `Einfg` drücken, Menü geht zu.

![OptiScaler-Menü](images/menu.jpg)

Der erste Start dauert ein paar Sekunden länger. Das ist normal.

**DirectX-11-Spiele** (z. B. Fallout 4): Oben im Menü unter **Upscalers** einen Upscaler mit
**w/Dx12** wählen (z. B. `FSR 2.2.1 w/Dx12`) und das Spiel neu starten.

## Geht nicht. Was jetzt?

Öffne **`dlssnr_nr.log`** im Spielordner mit dem Editor und schau dir die letzten Zeilen an:

| Im Log steht | Was tun |
|---|---|
| `runtime ready` | Läuft. Prüfe, ob Neural Rendering im `Einfg`-Menü an ist. |
| `nvngx_dlssnr.dll missing or wrong version` | `nvngx_dlssnr.dll` fehlt im Spielordner oder hat andere Gewichte. Neu besorgen (Schritt 2). |
| `not an RDNA2 GPU` | Deine Grafikkarte wird nicht unterstützt (siehe „Geht das auf meinem PC?“). |
| `hipModuleLoad ...` | Deine Karte konnte das Programm nicht laden. Bitte in [Issues](../../issues) mit Log melden. |
| Es gibt gar keine `dlssnr_nr.log` | Die Mod wurde nicht geladen. `setup_windows.bat` nochmal ausführen und prüfen, ob im Spiel ein Upscaler an ist. |

## Deinstallieren

Doppelklick im Spielordner auf **`Remove_OptiScaler.bat`**. Danach `nr_data`, `nr_runtime.dll`,
`nr_port.ini` und `nvngx.dll_dlssnr.dll` löschen.

## Wird das Spiel schneller oder langsamer?

Langsamer. Die AMD-Karte muss Arbeit machen, für die NVIDIA-Karten Spezial-Hardware haben. Auf einer
RX 6900 XT braucht ein Modell-Frame ca. 40 ms, kleinere Karten länger (siehe Tabelle oben). Rechne mit deutlich weniger FPS. Das ist eine Tech-Demo,
keine Performance-Mod.

## Wer hat was gemacht?

| Teil | Von |
|---|---|
| Das KI-Modell (DLSS Neural Rendering) | **NVIDIA**. Steckt in `nvngx_dlssnr.dll`, die du selbst besorgst. |
| Das Modell auf AMD RX 6000 zum Laufen bringen | **Dieses Projekt** (`nr_runtime.dll`, `nvngx.dll_dlssnr.dll`, `nr_data`) |
| Einbau ins Spiel (Menü, Einstellungen) | [**janblades OptiScaler-DLSSNR**](https://github.com/janblade/OptiScaler-DLSSNR-PreSR-Multipass) und das [OptiScaler](https://github.com/optiscaler/OptiScaler)-Team, siehe `docs/CREDITS.md` im Release |

Dieses Projekt enthält keine NVIDIA-Dateien oder -Gewichte. DLSS und NVIDIA sind Marken der NVIDIA
Corporation. Dieses Projekt hat nichts mit NVIDIA oder AMD zu tun und wird nicht von ihnen unterstützt.

## Für Entwickler

Der Quellcode von `nr_runtime.dll` und `nvngx.dll_dlssnr.dll` liegt in [`src/`](src). Bauen mit
`src\build.bat` (braucht AMD ROCm 6.4 für Windows und Visual Studio 2022 Build Tools).

## Lizenz

[GPL-3.0](LICENSE)
