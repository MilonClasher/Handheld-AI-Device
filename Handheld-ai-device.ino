/**
 * ESP32 Dev Kit V1 - Voice + Joystick Keyboard AI Terminal
 * INMP441 mic -> cloud speech-to-text -> LLM, OR type with an on-screen
 * keyboard navigated by an analog joystick. Two layouts, one SEND button.
 *
 * Wiring:
 *   SSD1306   SDA -> GPIO21, SCL -> GPIO22, VCC -> 3V3, GND -> GND
 *   INMP441   SCK -> GPIO32, WS -> GPIO25, SD -> GPIO34, L/R -> GND,
 *             VDD -> 3V3, GND -> GND
 *   BTN_MIC   GPIO18 -> button -> GND
 *   BTN_SEND  GPIO23 -> button -> GND
 *   Joystick  VRx -> GPIO35, VRy -> GPIO36, SW -> GPIO33 -> GND when pressed
 *
 *   IMPORTANT: wire the joystick's "5V" VCC pin to the ESP32's 3V3 pin,
 *   NOT to a true 5V source. The ESP32's analog pins are only rated for
 *   0 to 3.3V. The joystick is just two potentiometers, it works fine on
 *   3.3V, and feeding it 5V risks damaging GPIO35/GPIO36.
 *
 * Controls:
 *   Mic screen (idle):
 *     MIC   = start recording
 *     SEND  = show the last answer again, if one exists
 *     Joystick SW = open the on-screen keyboard
 *   Recording:
 *     MIC   = stop recording and transcribe
 *   Voice review ("You said: ..."):
 *     MIC   = discard, record again
 *     SEND  = ask the AI
 *   Keyboard:
 *     Joystick (any direction) = move the highlighted key
 *     Joystick SW = select the highlighted key (type the letter, or
 *                   trigger CAP / SYM / SPC / DEL / AC / SEND)
 *     MIC   = leave the keyboard, back to the mic screen (does not record)
 *     SEND  = ask the AI with whatever has been typed so far
 *   Answer:
 *     SEND  = next page
 *     MIC   = start a new voice recording
 *     Joystick SW = open the keyboard to type a new question
 *
 *   Hold MIC + SEND together while powering on to reopen BLE setup.
 *
 * BLE setup (write one text block, lines separated by new lines):
 *   WIFI:<ssid>                required
 *   PASS:<password>            required
 *   API:<chat api key>         required
 *   ENDPOINT:<chat url>        optional (default OpenRouter)
 *   MODEL:<chat model>         optional
 *   STT:<speech api key>       optional (default: same key as API:)
 *   STTURL:<transcription url> optional (default Groq Whisper)
 *   STTMODEL:<whisper model>   optional
 *
 * Libraries:
 *   Adafruit SSD1306 2.5.x, Adafruit GFX 1.11.x,
 *   NimBLE-Arduino 2.x, ArduinoJson 6.x
 * Board: "ESP32 Dev Module"
 * Partition Scheme: "No OTA (2MB APP/2MB SPIFFS)"
 *   Recordings are spooled to flash (LittleFS) instead of held in RAM, so
 *   they no longer compete with the TLS connection for a large contiguous
 *   block of heap. A 30 second recording needs about 960KB, so the 2MB
 *   SPIFFS partition here leaves comfortable headroom.
 */

#pragma GCC optimize("Os")

#include <Arduino.h>
#include <Wire.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <NimBLEDevice.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include <LittleFS.h>

#define REC_FILE_PATH  "/rec.raw"
bool fsReady = false;
// analogRead() uses the newer ADC driver, which cannot coexist with the
// legacy I2S driver the microphone uses below. The joystick is read with
// the legacy ADC driver instead (adc1_get_raw) to avoid that conflict.
#define JOY_VRX_CHANNEL  ADC1_CHANNEL_7   // GPIO35
#define JOY_VRY_CHANNEL  ADC1_CHANNEL_0   // GPIO36

// ─── Pins ────────────────────────────────────────────────────────────────────
#define OLED_SDA   21
#define OLED_SCL   22
#define I2S_SCK    32
#define I2S_WS     25
#define I2S_SD     34
#define BTN_MIC    18
#define BTN_SEND   23
#define JOY_VRX    35
#define JOY_VRY    36
#define JOY_SW     33

// ─── OLED ────────────────────────────────────────────────────────────────────
#define SCREEN_W   128
#define SCREEN_H   64
#define OLED_ADDR  0x3C
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

// ─── Audio settings ──────────────────────────────────────────────────────────
#define I2S_PORT        I2S_NUM_0
#define SAMPLE_RATE     16000
#define MAX_REC_SEC     30   // ~960KB of the 2MB SPIFFS partition, comfortable headroom left
#define MIC_GAIN_SHIFT  14
#define MIN_PEAK        300

// ─── Speech-to-text defaults ───────────────────────────────────────────────────
#define DEFAULT_STT_URL    "https://api.groq.com/openai/v1/audio/transcriptions"
#define DEFAULT_STT_MODEL  "whisper-large-v3-turbo"
#define STT_LANG           "en"

// ─── Chat defaults ───────────────────────────────────────────────────────────
#define DEFAULT_ENDPOINT  "https://openrouter.ai/api/v1/chat/completions"
#define DEFAULT_MODEL     "openrouter/free"

// ─── Constants ───────────────────────────────────────────────────────────────
#define MAX_PROMPT_LEN    128
#define WRAP_CHARS        21
#define VISIBLE_LINES     6
#define DEBOUNCE_MS       40
#define API_TIMEOUT_MS    20000
#define MAX_ANS_LINES     100
#define MAX_PROMPT_LINES  8

// ─── UI state machine ────────────────────────────────────────────────────────
enum UIMode {
  MODE_BLE, MODE_IDLE, MODE_RECORDING, MODE_TRANSCRIBING,
  MODE_PROMPT, MODE_KEYBOARD, MODE_LOADING, MODE_ANSWER
};
UIMode currentMode = MODE_BLE;
bool   needRedraw  = true;

struct JoyStep { int dx; int dy; };
struct Btn;

// Forward declarations to satisfy Arduino's auto-prototyper
bool btnPressed(Btn& b);
JoyStep readJoystickXY();

// ─── Keyboard layout (unchanged grid, now driven by a joystick) ──────────────
#define KEY_CAP   200
#define KEY_SYM   201
#define KEY_SPC   202
#define KEY_DEL   203
#define KEY_SEND  204
#define KEY_AC    205

enum KbPage { PAGE_ALPHA, PAGE_SYM };

static const char ROW_NUMS[]   = "0123456789";
static const char ROW_ALPHA1[] = "ABCDEFGHI";
static const char ROW_ALPHA2[] = "JKLMNOPQR";
static const char ROW_ALPHA3[] = "STUVWXYZ";
static const char ROW_SYM1[]   = "!@#$%^&*()";
static const char ROW_SYM2[]   = "-_=+[]{}\\|";
static const char ROW_SYM3[]   = ";:'\",.<>/?";
static const char ROW_SYM4[]   = "`~";

#define NROW_NUMS   10
#define NROW_A1      9
#define NROW_A2      9
#define NROW_A3      8
#define NROW_SPEC    6   // CAP SYM SPC DEL AC SEND

#define TOTAL_KEYS_ALPHA  (NROW_NUMS + NROW_A1 + NROW_A2 + NROW_A3 + NROW_SPEC)  // 42
#define TOTAL_KEYS_SYM    (10 + 10 + 9 + 2 + NROW_SPEC)                           // 37

int    kbCursor         = 0;
int    kbRow            = 0;   // 0..4: nums/sym1, row1, row2, row3, specials
int    kbCol            = 0;
bool   kbCaps           = true;
KbPage kbPage           = PAGE_ALPHA;
bool   kbSendRequested  = false;

#define KB_NUM_ROWS  5

int kbRowSize(KbPage page, int row) {
  static const int alphaSizes[KB_NUM_ROWS] = { NROW_NUMS, NROW_A1, NROW_A2, NROW_A3, NROW_SPEC };
  static const int symSizes[KB_NUM_ROWS]   = { 10, 10, 9, 2, NROW_SPEC };
  return (page == PAGE_ALPHA) ? alphaSizes[row] : symSizes[row];
}
int kbRowStart(KbPage page, int row) {
  int s = 0;
  for (int i = 0; i < row; i++) s += kbRowSize(page, i);
  return s;
}
void kbSyncCursor() { kbCursor = kbRowStart(kbPage, kbRow) + kbCol; }
void kbMoveTo(int row, int col) {
  kbRow = row;
  int sz = kbRowSize(kbPage, kbRow);
  kbCol = (col < 0) ? 0 : (col >= sz ? sz - 1 : col);
  kbSyncCursor();
}

// ─── App state ───────────────────────────────────────────────────────────────
char   promptBuf[MAX_PROMPT_LEN + 1] = {0};
int    promptLen        = 0;
String promptLines[MAX_PROMPT_LINES];
int    promptLineCount  = 0;
String ansLines[MAX_ANS_LINES];
int    totalLines       = 0;
int    scrollLine       = 0;

// ─── Credentials ─────────────────────────────────────────────────────────────
Preferences prefs;
String storedSSID, storedPass, storedApiKey;
String storedEndpoint, storedModel;
String storedSttKey, storedSttUrl, storedSttModel;

// ─── BLE ─────────────────────────────────────────────────────────────────────
#define BLE_SERVICE_UUID  "12345678-1234-1234-1234-123456789abc"
#define BLE_CHAR_UUID     "abcdefab-cdef-abcd-efab-cdefabcdefab"
NimBLEServer*         bleServer    = nullptr;
NimBLECharacteristic* bleChar      = nullptr;
bool                  bleConnected = false;
bool                  credReceived = false;

// ─── Buttons (polled, debounced) ─────────────────────────────────────────────
struct Btn { uint8_t pin; bool last; unsigned long t; };
Btn bMic   = { BTN_MIC,  HIGH, 0 };
Btn bSend  = { BTN_SEND, HIGH, 0 };
Btn bJoySw = { JOY_SW,   HIGH, 0 };

bool btnPressed(Btn& b) {
  bool level = digitalRead(b.pin);
  if (level != b.last && millis() - b.t > DEBOUNCE_MS) {
    b.last = level;
    b.t    = millis();
    return level == LOW;
  }
  return false;
}
void resetButtons() {
  bMic.last   = digitalRead(bMic.pin);
  bSend.last  = digitalRead(bSend.pin);
  bJoySw.last = digitalRead(bJoySw.pin);
  bMic.t = bSend.t = bJoySw.t = millis();
}

// ─── Joystick axis stepping ──────────────────────────────────────────────────
// Treats each axis like a rotary detent: push past the outer threshold to
// fire one step, the axis must return to the center dead zone before it can
// fire again. Both axes feed the same cursor so the joystick can be nudged
// in whichever direction feels natural.
#define JOY_OUTER_LOW   1200
#define JOY_OUTER_HIGH  2900
#define JOY_INNER_LOW   1700
#define JOY_INNER_HIGH  2400

bool armXNeg = false, armXPos = false, armYNeg = false, armYPos = false;


JoyStep readJoystickXY() {
  int vx = adc1_get_raw(JOY_VRX_CHANNEL);
  int vy = adc1_get_raw(JOY_VRY_CHANNEL);
  JoyStep j = {0, 0};

  if (vx < JOY_OUTER_LOW)                         { if (!armXNeg) { j.dx -= 1; armXNeg = true; } }
  else if (vx > JOY_INNER_LOW && vx < JOY_INNER_HIGH) armXNeg = false;
  if (vx > JOY_OUTER_HIGH)                        { if (!armXPos) { j.dx += 1; armXPos = true; } }
  else if (vx > JOY_INNER_LOW && vx < JOY_INNER_HIGH) armXPos = false;

  if (vy < JOY_OUTER_LOW)                         { if (!armYNeg) { j.dy -= 1; armYNeg = true; } }
  else if (vy > JOY_INNER_LOW && vy < JOY_INNER_HIGH) armYNeg = false;
  if (vy > JOY_OUTER_HIGH)                        { if (!armYPos) { j.dy += 1; armYPos = true; } }
  else if (vy > JOY_INNER_LOW && vy < JOY_INNER_HIGH) armYPos = false;

  return j;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Autocorrect dictionary
// ═══════════════════════════════════════════════════════════════════════════════
static const char* const acDict[] = {
  "able","about","above","across","after","again","against","ago","ahead",
  "almost","along","also","although","always","among","another","answer",
  "around","away","back","because","before","behind","being","below","between",
  "beyond","both","bring","build","called","came","can","cannot","come",
  "could","does","done","down","during","each","early","either","else",
  "enough","even","every","everything","exactly","find","first","for",
  "forward","found","from","front","gave","give","going","good","great",
  "grow","have","help","here","high","home","house","how","however",
  "into","just","keep","know","large","last","later","left","less","like",
  "line","little","long","look","make","many","maybe","mean","might","more",
  "most","much","must","name","near","never","next","night","nothing","now",
  "often","once","only","open","other","over","own","part","people","place",
  "please","point","possible","put","question","quick","right","same","send",
  "should","since","small","some","something","soon","start","still","such",
  "take","than","that","their","them","then","there","these","they","thing",
  "think","this","those","through","time","today","together","tomorrow","too",
  "towards","try","under","until","upon","used","very","want","water","well",
  "were","what","when","where","which","while","will","with","without","would",
  "write","year","your"
};
#define AC_DICT_SIZE  (sizeof(acDict) / sizeof(acDict[0]))
char acSuggestion[32] = {0};

void acGetCurrentWord(char* out, int maxLen) {
  out[0] = '\0';
  int len = 0;
  for (int i = promptLen - 1; i >= 0 && len < maxLen - 1; i--) {
    char c = promptBuf[i];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) len++;
    else break;
  }
  if (len == 0) return;
  int start = promptLen - len;
  for (int i = 0; i < len; i++) {
    char c = promptBuf[start + i];
    out[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
  }
  out[len] = '\0';
}

void acUpdate() {
  acSuggestion[0] = '\0';
  char word[32];
  acGetCurrentWord(word, sizeof(word));
  int wlen = strlen(word);
  if (wlen < 2) return;
  for (int i = 0; i < (int)AC_DICT_SIZE; i++) {
    const char* entry = acDict[i];
    if (strncmp(word, entry, wlen) == 0 && strlen(entry) > (size_t)wlen) {
      strncpy(acSuggestion, entry, sizeof(acSuggestion) - 1);
      acSuggestion[sizeof(acSuggestion) - 1] = '\0';
      return;
    }
  }
}

void acAccept() {
  if (acSuggestion[0] == '\0') return;
  char word[32];
  acGetCurrentWord(word, sizeof(word));
  int wlen = strlen(word);
  promptLen -= wlen;
  promptBuf[promptLen] = '\0';
  const char* sug = acSuggestion;
  int slen = strlen(sug);
  for (int i = 0; i < slen && promptLen < MAX_PROMPT_LEN; i++) {
    char c = sug[i];
    if (i == 0 && kbCaps && c >= 'a' && c <= 'z') c -= 32;
    promptBuf[promptLen++] = c;
  }
  promptBuf[promptLen] = '\0';
  acSuggestion[0] = '\0';
  needRedraw = true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLE callbacks
// ═══════════════════════════════════════════════════════════════════════════════
class BLECallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* pChar, NimBLEConnInfo& connInfo) override {
    String val = pChar->getValue().c_str();
    val.trim();
    int start = 0;
    while (start < (int)val.length()) {
      int nl = val.indexOf('\n', start);
      String line = (nl == -1) ? val.substring(start) : val.substring(start, nl);
      line.trim();
      if      (line.startsWith("WIFI:"))     storedSSID     = line.substring(5);
      else if (line.startsWith("PASS:"))     storedPass     = line.substring(5);
      else if (line.startsWith("API:"))      storedApiKey   = line.substring(4);
      else if (line.startsWith("ENDPOINT:")) storedEndpoint = line.substring(9);
      else if (line.startsWith("MODEL:"))    storedModel    = line.substring(6);
      else if (line.startsWith("STT:"))      storedSttKey   = line.substring(4);
      else if (line.startsWith("STTURL:"))   storedSttUrl   = line.substring(7);
      else if (line.startsWith("STTMODEL:")) storedSttModel = line.substring(9);
      start = (nl == -1) ? val.length() : nl + 1;
    }
    if (storedSSID.length() && storedPass.length() && storedApiKey.length()) {
      prefs.begin("voiceai", false);
      prefs.putString("ssid",     storedSSID);
      prefs.putString("pass",     storedPass);
      prefs.putString("apikey",   storedApiKey);
      prefs.putString("endpoint", storedEndpoint);
      prefs.putString("model",    storedModel);
      prefs.putString("sttkey",   storedSttKey);
      prefs.putString("stturl",   storedSttUrl);
      prefs.putString("sttmodel", storedSttModel);
      prefs.end();
      credReceived = true;
    }
  }
};
class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* s, NimBLEConnInfo& c) override {
    bleConnected = true; needRedraw = true;
  }
  void onDisconnect(NimBLEServer* s, NimBLEConnInfo& c, int r) override {
    bleConnected = false; needRedraw = true;
    NimBLEDevice::startAdvertising();
  }
};

void startBLE() {
  NimBLEDevice::init("ESP32-VoiceAI");
  NimBLEDevice::setPower(9);
  bleServer = NimBLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());
  NimBLEService* svc = bleServer->createService(BLE_SERVICE_UUID);
  bleChar = svc->createCharacteristic(BLE_CHAR_UUID,
    NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  bleChar->setCallbacks(new BLECallbacks());
  svc->start();
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(BLE_SERVICE_UUID);
  adv->enableScanResponse(true);
  NimBLEDevice::startAdvertising();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Audio recording task
// ═══════════════════════════════════════════════════════════════════════════════
size_t          recMaxBytes = (size_t)SAMPLE_RATE * MAX_REC_SEC * 2;
volatile size_t recSamples  = 0;   // samples written so far, for the UI
volatile bool   recRunning  = false;
volatile bool   recActive   = false;
volatile int    recPeak     = 0;
volatile int    recMaxPeak  = 0;

static bool i2sStart() {
  i2s_config_t cfg = {};
  cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate          = SAMPLE_RATE;
  cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count        = 8;
  cfg.dma_buf_len          = 256;
  cfg.use_apll             = false;
  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;

  i2s_pin_config_t pins = {};
  pins.mck_io_num   = I2S_PIN_NO_CHANGE;
  pins.bck_io_num   = I2S_SCK;
  pins.ws_io_num    = I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = I2S_SD;
  if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) {
    i2s_driver_uninstall(I2S_PORT);
    return false;
  }
  i2s_zero_dma_buffer(I2S_PORT);
  return true;
}

static void recTask(void*) {
  static int32_t raw[256];
  if (!i2sStart()) { recActive = false; vTaskDelete(NULL); return; }

  File f = LittleFS.open(REC_FILE_PATH, "w");
  if (!f) {
    i2s_driver_uninstall(I2S_PORT);
    recActive = false;
    vTaskDelete(NULL);
    return;
  }

  int    settle     = SAMPLE_RATE / 10;
  size_t bytesSoFar = 0;
  size_t bytes      = 0;
  static int16_t outChunk[256];

  while (recRunning && bytesSoFar < recMaxBytes) {
    if (i2s_read(I2S_PORT, raw, sizeof(raw), &bytes, pdMS_TO_TICKS(100)) != ESP_OK) continue;
    int n    = bytes / 4;
    int peak = 0;
    int outN = 0;
    for (int i = 0; i < n && bytesSoFar + (size_t)(outN + 1) * 2 <= recMaxBytes; i++) {
      if (settle > 0) { settle--; continue; }
      int32_t s = raw[i] >> MIC_GAIN_SHIFT;
      if (s >  32767) s =  32767;
      if (s < -32768) s = -32768;
      outChunk[outN++] = (int16_t)s;
      int a = (s < 0) ? -s : s;
      if (a > peak) peak = a;
    }
    if (outN > 0) {
      f.write((const uint8_t*)outChunk, outN * 2);
      bytesSoFar += (size_t)outN * 2;
      recSamples  = bytesSoFar / 2;
    }
    recPeak = peak;
    if (peak > recMaxPeak) recMaxPeak = peak;
  }
  f.close();
  recSamples = bytesSoFar / 2;
  i2s_driver_uninstall(I2S_PORT);
  recActive = false;
  vTaskDelete(NULL);
}

void freeRecBuffer() {
  if (fsReady) LittleFS.remove(REC_FILE_PATH);
}

bool startRecording() {
  if (!fsReady) return false;
  freeRecBuffer();
  recSamples = 0; recPeak = 0; recMaxPeak = 0;
  recRunning = true;
  recActive  = true;
  xTaskCreatePinnedToCore(recTask, "rec", 4096, NULL, 3, NULL, 0);
  return true;
}

void stopRecording() {
  recRunning = false;
  unsigned long t = millis();
  while (recActive && millis() - t < 1000) delay(5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Keyboard helpers (grid logic unchanged, selection source is now the joystick)
// ═══════════════════════════════════════════════════════════════════════════════
int kbTotalKeys() { return (kbPage == PAGE_ALPHA) ? TOTAL_KEYS_ALPHA : TOTAL_KEYS_SYM; }

int kbGetKey(int idx) {
  const char* r0; const char* r1; const char* r2; const char* r3;
  int s0, s1, s2, s3;
  if (kbPage == PAGE_ALPHA) {
    r0=ROW_NUMS; s0=NROW_NUMS; r1=ROW_ALPHA1; s1=NROW_A1;
    r2=ROW_ALPHA2; s2=NROW_A2; r3=ROW_ALPHA3; s3=NROW_A3;
  } else {
    r0=ROW_SYM1; s0=10; r1=ROW_SYM2; s1=10;
    r2=ROW_SYM3; s2=9;  r3=ROW_SYM4; s3=2;
  }
  if (idx < s0) return (unsigned char)r0[idx]; idx -= s0;
  if (idx < s1) return (unsigned char)r1[idx]; idx -= s1;
  if (idx < s2) return (unsigned char)r2[idx]; idx -= s2;
  if (idx < s3) return (unsigned char)r3[idx]; idx -= s3;
  const int specials[] = {KEY_CAP, KEY_SYM, KEY_SPC, KEY_DEL, KEY_AC, KEY_SEND};
  if (idx < NROW_SPEC) return specials[idx];
  return ' ';
}

void kbKeyLabel(int keyId, char* buf, int bufLen) {
  if (keyId == KEY_CAP)  { strncpy(buf, kbCaps ? "CAP" : "cap", bufLen); return; }
  if (keyId == KEY_SYM)  { strncpy(buf, kbPage == PAGE_SYM ? "ABC" : "SYM", bufLen); return; }
  if (keyId == KEY_SPC)  { strncpy(buf, "SPC", bufLen); return; }
  if (keyId == KEY_DEL)  { strncpy(buf, "DEL", bufLen); return; }
  if (keyId == KEY_AC)   { strncpy(buf, " AC", bufLen); return; }
  if (keyId == KEY_SEND) { strncpy(buf, "SND", bufLen); return; }
  char c = (char)keyId;
  if (kbPage == PAGE_ALPHA && !kbCaps && c >= 'A' && c <= 'Z') c += 32;
  buf[0] = c; buf[1] = 0;
}

void doDelete() {
  if (promptLen > 0) { promptBuf[--promptLen] = '\0'; }
  acUpdate();
  needRedraw = true;
}
void doSpace() {
  if (promptLen < MAX_PROMPT_LEN) { promptBuf[promptLen++] = ' '; promptBuf[promptLen] = '\0'; }
  acSuggestion[0] = '\0';
  needRedraw = true;
}
void doAC() { acAccept(); needRedraw = true; }

void kbActivateKey(int keyId) {
  if (keyId == KEY_CAP)  { kbCaps = !kbCaps; needRedraw = true; return; }
  if (keyId == KEY_SYM)  { kbPage = (kbPage == PAGE_ALPHA) ? PAGE_SYM : PAGE_ALPHA; kbRow = 0; kbCol = 0; kbSyncCursor(); needRedraw = true; return; }
  if (keyId == KEY_SPC)  { doSpace(); return; }
  if (keyId == KEY_DEL)  { doDelete(); return; }
  if (keyId == KEY_AC)   { doAC(); return; }
  if (keyId == KEY_SEND) { kbSendRequested = true; return; }
  if (promptLen < MAX_PROMPT_LEN) {
    char c = (char)keyId;
    if (kbPage == PAGE_ALPHA) {
      if ( kbCaps && c >= 'a' && c <= 'z') c -= 32;
      if (!kbCaps && c >= 'A' && c <= 'Z') c += 32;
    }
    promptBuf[promptLen++] = c;
    promptBuf[promptLen] = '\0';
    acUpdate();
  }
  needRedraw = true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Drawing
// ═══════════════════════════════════════════════════════════════════════════════
void drawMicIcon() {
  display.fillRoundRect(58, 10, 12, 20, 6, SSD1306_WHITE);
  display.drawCircleHelper(64, 24, 10, 0x4 | 0x8, SSD1306_WHITE);
  display.drawFastVLine(54, 20, 5, SSD1306_WHITE);
  display.drawFastVLine(74, 20, 5, SSD1306_WHITE);
  display.drawFastVLine(64, 35, 6, SSD1306_WHITE);
  display.drawFastHLine(56, 41, 17, SSD1306_WHITE);
}

void drawCredConfirm() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Saved new keys:");
  display.println();
  String apiTail = storedApiKey.length() >= 4 ? storedApiKey.substring(storedApiKey.length() - 4) : storedApiKey;
  display.print("API  ...");
  display.println(apiTail);
  String sttKey = storedSttKey.length() ? storedSttKey : storedApiKey;
  String sttTail = sttKey.length() >= 4 ? sttKey.substring(sttKey.length() - 4) : sttKey;
  display.print("STT  ...");
  display.println(sttTail);
  display.display();
  delay(1800);
}

void drawBLEMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("== BLE SETUP ==");
  display.println("Required:");
  display.println("WIFI: PASS: API:");
  display.println("Optional: STT:");
  display.println("ENDPOINT: MODEL:");
  if (bleConnected) { display.setCursor(0, 56); display.print("* Connected *"); }
  display.display();
}

void drawIdleMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print("VOICE AI");
  drawMicIcon();
  display.setCursor(0, 44);
  display.print("MIC=speak JOY=kbd");
  if (totalLines > 0) {
    display.setCursor(0, 56);
    display.print("SEND = last answer");
  }
  display.display();
}

void drawRecordingMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  if ((millis() / 500) % 2 == 0) display.fillCircle(4, 4, 3, SSD1306_WHITE);
  display.setCursor(12, 0);
  display.print("REC ");
  display.print((float)recSamples / SAMPLE_RATE, 1);
  display.print("/");
  display.print((int)(recMaxBytes / 2 / SAMPLE_RATE));
  display.print("s");
  drawMicIcon();
  int pk  = recPeak;
  int lvl = constrain(pk, 0, 12000) / 120;
  display.drawRect(14, 46, 100, 6, SSD1306_WHITE);
  display.fillRect(14, 46, lvl, 6, SSD1306_WHITE);
  display.setCursor(0, 56);
  display.print("MIC = stop");
  display.display();
}

void drawTranscribingMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 20);
  display.println("Transcribing...");
  display.println();
  display.println("Please wait");
  display.display();
}

void drawPromptMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print("You said:");
  display.drawLine(0, 9, 127, 9, SSD1306_WHITE);
  int y = 11;
  for (int i = 0; i < promptLineCount && i < 5; i++) {
    display.setCursor(0, y);
    display.print(promptLines[i]);
    y += 8;
  }
  display.drawLine(0, 53, 127, 53, SSD1306_WHITE);
  display.setCursor(0, 56);
  display.print("MIC=redo  SEND=ask");
  display.display();
}

void drawKeyRow(const char* keys, int nKeys, int rowStart, int y) {
  int cellW  = (SCREEN_W - 2) / nKeys;
  int totalW = nKeys * cellW;
  int xOff   = (SCREEN_W - totalW) / 2;
  for (int i = 0; i < nKeys; i++) {
    int  x     = xOff + i * cellW;
    bool isSel = (rowStart + i == kbCursor);
    char label[4];
    kbKeyLabel(kbGetKey(rowStart + i), label, 4);
    if (isSel) {
      display.fillRect(x, y, cellW - 1, 7, SSD1306_WHITE);
      display.setTextColor(SSD1306_BLACK);
    } else {
      display.setTextColor(SSD1306_WHITE);
    }
    display.setCursor(x + 1, y + 1);
    display.print(label[0]);
    if (isSel) display.setTextColor(SSD1306_WHITE);
  }
}

void drawSpecialRow(int rowStart, int y) {
  const int W = SCREEN_W / NROW_SPEC;
  const int specials[] = {KEY_CAP, KEY_SYM, KEY_SPC, KEY_DEL, KEY_AC, KEY_SEND};
  for (int i = 0; i < NROW_SPEC; i++) {
    int  x     = i * W;
    bool isSel = (rowStart + i == kbCursor);
    char label[4];
    kbKeyLabel(specials[i], label, 4);
    if (isSel) {
      display.fillRect(x, y, W - 1, 7, SSD1306_WHITE);
      display.setTextColor(SSD1306_BLACK);
    } else {
      display.drawRect(x, y, W - 1, 7, SSD1306_WHITE);
      display.setTextColor(SSD1306_WHITE);
    }
    display.setCursor(x + 1, y + 1);
    display.print(label);
    if (isSel) display.setTextColor(SSD1306_WHITE);
  }
}

void drawKeyboardMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.print("> ");
  int pStart = (promptLen > 19) ? promptLen - 19 : 0;
  for (int i = pStart; i < promptLen; i++) display.print(promptBuf[i]);
  display.print("_");
  display.drawLine(0, 8, 127, 8, SSD1306_WHITE);

  acUpdate();
  if (acSuggestion[0] != '\0') {
    display.fillRect(0, 9, 128, 8, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
    display.setCursor(1, 10);
    display.print("~>");
    display.print(acSuggestion);
    display.setTextColor(SSD1306_WHITE);
  } else {
    display.setCursor(1, 10);
    display.print("~> ...");
  }
  display.drawLine(0, 17, 127, 17, SSD1306_WHITE);

  if (kbPage == PAGE_ALPHA) {
    int r0=0, r1=r0+NROW_NUMS, r2=r1+NROW_A1, r3=r2+NROW_A2, r4=r3+NROW_A3;
    drawKeyRow(ROW_NUMS,   NROW_NUMS, r0, 19);
    drawKeyRow(ROW_ALPHA1, NROW_A1,   r1, 27);
    drawKeyRow(ROW_ALPHA2, NROW_A2,   r2, 35);
    drawKeyRow(ROW_ALPHA3, NROW_A3,   r3, 43);
    drawSpecialRow(r4, 51);
  } else {
    int r0=0, r1=r0+10, r2=r1+10, r3=r2+9, r4=r3+2;
    drawKeyRow(ROW_SYM1, 10, r0, 19);
    drawKeyRow(ROW_SYM2, 10, r1, 27);
    drawKeyRow(ROW_SYM3,  9, r2, 35);
    drawKeyRow(ROW_SYM4,  2, r3, 43);
    drawSpecialRow(r4, 51);
  }

  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 59);
  display.print(kbCaps ? "ABC" : "abc");
  display.print(kbPage == PAGE_ALPHA ? "|alp" : "|sym");
  display.setCursor(68, 59);
  display.print("SW=select");

  display.display();
}

void drawLoadingMode() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 24);
  display.print("Asking AI");
  display.display();
  display.setTextSize(1);
}

void drawAnswerMode() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  int pages = (totalLines + VISIBLE_LINES - 1) / VISIBLE_LINES;
  int page  = scrollLine / VISIBLE_LINES + 1;
  char hdr[32];
  snprintf(hdr, sizeof(hdr), "%d/%d SEND=more MIC=new", page, pages);
  display.setCursor(0, 0);
  display.print(hdr);
  display.drawLine(0, 9, 127, 9, SSD1306_WHITE);
  int y = 11;
  for (int i = scrollLine; i < totalLines && i < scrollLine + VISIBLE_LINES; i++) {
    display.setCursor(0, y);
    display.print(ansLines[i]);
    y += 8;
  }
  display.display();
}

void showMessage(const char* l1, const String& l2) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(true);
  display.setCursor(0, 16);
  display.println(l1);
  display.println();
  display.println(l2);
  display.display();
  display.setTextWrap(false);
  delay(2200);
  currentMode = MODE_IDLE;
  needRedraw  = true;
  resetButtons();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Text helpers
// ═══════════════════════════════════════════════════════════════════════════════
String stripMarkdown(String s) {
  String out;
  out.reserve(s.length());
  int len = s.length();
  int i   = 0;
  while (i < len) {
    char c = s[i];
    if (c == '`' && i + 2 < len && s[i+1] == '`' && s[i+2] == '`') {
      i += 3;
      while (i < len && s[i] != '\n') i++;
      if (i < len) i++;
      while (i < len) {
        if (s[i] == '`' && i + 2 < len && s[i+1] == '`' && s[i+2] == '`') { i += 3; break; }
        out += s[i++];
      }
      out += '\n';
      continue;
    }
    if (c == '`') {
      i++;
      while (i < len && s[i] != '`') out += s[i++];
      if (i < len) i++;
      continue;
    }
    if ((c == '-' || c == '*' || c == '_') && i + 2 < len && s[i+1] == c && s[i+2] == c) {
      int j = i + 3;
      while (j < len && (s[j] == c || s[j] == ' ')) j++;
      if (j >= len || s[j] == '\n') { out += '\n'; i = (j < len) ? j + 1 : j; continue; }
    }
    if (c == '#' && (i == 0 || s[i-1] == '\n')) {
      while (i < len && s[i] == '#') i++;
      while (i < len && s[i] == ' ') i++;
      continue;
    }
    if (c == '>' && (i == 0 || s[i-1] == '\n')) {
      i++;
      while (i < len && s[i] == ' ') i++;
      continue;
    }
    if ((c == '-' || c == '*' || c == '+') && (i == 0 || s[i-1] == '\n') && i + 1 < len && s[i+1] == ' ') {
      out += "- "; i += 2; continue;
    }
    if ((c == '*' || c == '_') && i + 2 < len && s[i+1] == c && s[i+2] == c) {
      i += 3;
      while (i < len) {
        if (s[i] == c && i + 2 < len && s[i+1] == c && s[i+2] == c) { i += 3; break; }
        out += s[i++];
      }
      continue;
    }
    if ((c == '*' || c == '_') && i + 1 < len && s[i+1] == c) {
      i += 2;
      while (i < len) {
        if (s[i] == c && i + 1 < len && s[i+1] == c) { i += 2; break; }
        out += s[i++];
      }
      continue;
    }
    if (c == '*' || c == '_') {
      i++;
      while (i < len) { if (s[i] == c) { i++; break; } out += s[i++]; }
      continue;
    }
    if (c == '[') {
      int close = s.indexOf(']', i);
      if (close != -1 && close + 1 < len && s[close+1] == '(') {
        int urlEnd = s.indexOf(')', close + 2);
        if (urlEnd != -1) {
          for (int k = i + 1; k < close; k++) out += s[k];
          i = urlEnd + 1;
          continue;
        }
      }
    }
    if (c == '\\' && i + 1 < len) { i++; out += s[i++]; continue; }
    if (c == '\n') {
      out += '\n'; i++;
      int newlines = 0;
      while (i < len && s[i] == '\n') { newlines++; i++; }
      if (newlines > 0) out += '\n';
      continue;
    }
    out += c; i++;
  }
  out.trim();
  return out;
}

int wrapInto(const String& src, int maxChars, String* lines, int maxLines) {
  int count = 0, len = src.length(), i = 0;
  while (i < len && count < maxLines) {
    int nlPos = src.indexOf('\n', i);
    int end   = (nlPos == -1) ? len : nlPos;
    String seg = src.substring(i, end);
    while (seg.length() > 0 && count < maxLines) {
      if ((int)seg.length() <= maxChars) { lines[count++] = seg; seg = ""; }
      else {
        int cut = maxChars;
        for (int j = maxChars; j > 0; j--) { if (seg[j] == ' ') { cut = j; break; } }
        lines[count++] = seg.substring(0, cut);
        seg = seg.substring(cut);
        if (seg.startsWith(" ")) seg = seg.substring(1);
      }
    }
    i = (nlPos == -1) ? len : nlPos + 1;
  }
  return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// WiFi
// ═══════════════════════════════════════════════════════════════════════════════
const char* wifiFailReason(wl_status_t s) {
  switch (s) {
    case WL_NO_SSID_AVAIL:  return "SSID not found";
    case WL_CONNECT_FAILED: return "Wrong password";
    case WL_CONNECTION_LOST:return "Connection lost";
    case WL_DISCONNECTED:   return "Disconnected";
    default:                return "Timed out";
  }
}

bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true, true);
  delay(100);
  WiFi.begin(storedSSID.c_str(), storedPass.c_str());
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting WiFi...");
  display.println(storedSSID.c_str());
  display.display();

  unsigned long t = millis();
  wl_status_t st;
  while ((st = WiFi.status()) != WL_CONNECTED) {
    if (millis() - t > 15000) {
      display.clearDisplay();
      display.setCursor(0, 0);
      display.println("WiFi failed:");
      display.println(wifiFailReason(st));
      display.println(storedSSID.c_str());
      display.println();
      display.println("Check 2.4GHz band");
      display.println("and WPA2 security");
      display.display();
      delay(3500);
      return false;
    }
    delay(200);
  }
  return true;
}

bool ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  return connectWiFi();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Speech-to-text
// ═══════════════════════════════════════════════════════════════════════════════
static void makeWavHeader(uint8_t* h, uint32_t dataBytes) {
  uint32_t fileSize = dataBytes + 36;
  uint32_t fmtLen   = 16;
  uint16_t fmt = 1, ch = 1, ba = 2, bps = 16;
  uint32_t sr = SAMPLE_RATE, br = SAMPLE_RATE * 2;
  memcpy(h,      "RIFF", 4); memcpy(h + 4,  &fileSize, 4);
  memcpy(h + 8,  "WAVEfmt ", 8);
  memcpy(h + 16, &fmtLen, 4);
  memcpy(h + 20, &fmt, 2);   memcpy(h + 22, &ch, 2);
  memcpy(h + 24, &sr, 4);    memcpy(h + 28, &br, 4);
  memcpy(h + 32, &ba, 2);    memcpy(h + 34, &bps, 2);
  memcpy(h + 36, "data", 4); memcpy(h + 40, &dataBytes, 4);
}

static bool writeAll(WiFiClientSecure& c, const uint8_t* p, size_t n) {
  int stalls = 0;
  while (n > 0) {
    size_t chunk = (n > 1024) ? 1024 : n;
    size_t w = c.write(p, chunk);
    if (w == 0) {
      if (!c.connected() || ++stalls > 1000) return false;
      delay(5);
      continue;
    }
    stalls = 0;
    p += w; n -= w;
  }
  return true;
}

bool transcribeAudio(String& text, String& err) {
  String url   = storedSttUrl.length()   ? storedSttUrl   : String(DEFAULT_STT_URL);
  String key   = storedSttKey.length()   ? storedSttKey   : storedApiKey;
  String model = storedSttModel.length() ? storedSttModel : String(DEFAULT_STT_MODEL);

  int p = url.indexOf("://");
  if (p < 0) { err = "Bad STT URL"; return false; }
  String rest = url.substring(p + 3);
  int sl = rest.indexOf('/');
  String host = (sl < 0) ? rest : rest.substring(0, sl);
  String path = (sl < 0) ? String("/") : rest.substring(sl);

  const String B = "----ESP32Voice7d3f";
  String pre;
  pre.reserve(420);
  pre += "--" + B + "\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n" + model + "\r\n";
  pre += "--" + B + "\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n" STT_LANG "\r\n";
  pre += "--" + B + "\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n";
  pre += "--" + B + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n";
  String post = "\r\n--" + B + "--\r\n";

  uint32_t dataBytes = (uint32_t)recSamples * 2;
  uint32_t total     = pre.length() + 44 + dataBytes + post.length();

  File recFile = LittleFS.open(REC_FILE_PATH, "r");
  if (!recFile) { err = "No recording file"; return false; }

  IPAddress resolvedIP;
  bool dnsOk = WiFi.hostByName(host.c_str(), resolvedIP);
  Serial.printf("DNS for %s: %s (%s)\n", host.c_str(),
                dnsOk ? resolvedIP.toString().c_str() : "FAILED",
                dnsOk ? "ok" : "check hotspot/DNS");
  if (!dnsOk) { err = "DNS fail"; return false; }

  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(20);
  Serial.printf("Free heap before STT connect: %u bytes (largest block: %u)\n",
                ESP.getFreeHeap(), ESP.getMaxAllocHeap());
  bool connected = client.connect(host.c_str(), 443);
  if (!connected) {
    delay(300);
    Serial.printf("STT connect retry, free heap: %u bytes (largest block: %u)\n",
                  ESP.getFreeHeap(), ESP.getMaxAllocHeap());
    connected = client.connect(host.c_str(), 443);
  }
  if (!connected) {
    Serial.printf("STT connect failed, free heap: %u bytes (largest block: %u)\n",
                  ESP.getFreeHeap(), ESP.getMaxAllocHeap());
    err = "STT connect fail";
    return false;
  }

  client.print("POST " + path + " HTTP/1.1\r\n");
  client.print("Host: " + host + "\r\n");
  client.print("User-Agent: esp32-voice/1.0\r\n");
  client.print("Authorization: Bearer " + key + "\r\n");
  client.print("Content-Type: multipart/form-data; boundary=" + B + "\r\n");
  client.print("Content-Length: " + String(total) + "\r\n");
  client.print("Connection: close\r\n\r\n");
  client.print(pre);

  uint8_t hdr[44];
  makeWavHeader(hdr, dataBytes);
  bool uploadOk = writeAll(client, hdr, 44);

  static uint8_t fileChunk[1024];
  while (uploadOk && recFile.available()) {
    int n = recFile.read(fileChunk, sizeof(fileChunk));
    if (n <= 0) break;
    uploadOk = writeAll(client, fileChunk, (size_t)n);
  }
  recFile.close();

  if (!uploadOk) { err = "Upload failed"; client.stop(); return false; }
  client.print(post);

  String resp;
  unsigned long last = millis();
  while (millis() - last < 20000 && resp.length() < 3000) {
    while (client.available()) { resp += (char)client.read(); last = millis(); }
    if (!client.connected() && !client.available()) break;
    delay(5);
  }
  client.stop();

  if (resp.length() < 12) { err = "No reply"; return false; }
  int code = resp.substring(9, 12).toInt();
  if (code != 200) { err = "HTTP " + String(code); return false; }

  int hdrEnd = resp.indexOf("\r\n\r\n");
  int b1 = resp.indexOf('{', hdrEnd < 0 ? 0 : hdrEnd);
  int b2 = resp.lastIndexOf('}');
  if (b1 < 0 || b2 <= b1) { err = "Bad reply"; return false; }

  DynamicJsonDocument doc(1024);
  if (deserializeJson(doc, resp.substring(b1, b2 + 1))) { err = "JSON error"; return false; }
  const char* t = doc["text"];
  text = t ? String(t) : String("");
  text.trim();
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chat API dispatcher
// ═══════════════════════════════════════════════════════════════════════════════
static const char SYSTEM_PROMPT[] =
  "You are a concise assistant on a tiny 128x64 OLED display. "
  "Reply in plain text only. No markdown, asterisks, underscores, hashes, "
  "backticks, bullet symbols, bold, italic, or code fences. "
  "Use plain words and short sentences.";

static int doPost(HTTPClient& http, WiFiClientSecure& client,
                  const String& url, const String& body,
                  const String& authHeader, const String& authValue,
                  const String& extraHeaderKey = "",
                  const String& extraHeaderVal = "") {
  http.setTimeout(API_TIMEOUT_MS);
  http.begin(client, url);
  http.addHeader("Content-Type", "application/json");
  if (authHeader.length())     http.addHeader(authHeader, authValue);
  if (extraHeaderKey.length()) http.addHeader(extraHeaderKey, extraHeaderVal);
  return http.POST(body);
}

String callOpenAICompat(const String& prompt, const String& endpoint, const String& model) {
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  DynamicJsonDocument doc(1200);
  doc["model"]      = model;
  doc["max_tokens"] = 300;
  JsonArray  msgs = doc.createNestedArray("messages");
  JsonObject sys  = msgs.createNestedObject();
  sys["role"]     = "system";
  sys["content"]  = SYSTEM_PROMPT;
  JsonObject msg  = msgs.createNestedObject();
  msg["role"]     = "user";
  msg["content"]  = prompt;
  String body; serializeJson(doc, body);

  int code = doPost(http, client, endpoint, body,
                    "Authorization", "Bearer " + storedApiKey,
                    "HTTP-Referer", "http://esp32-voice");
  if (code != 200) { http.end(); return "Error: HTTP " + String(code); }
  String payload = http.getString(); http.end();

  DynamicJsonDocument resp(4096);
  if (deserializeJson(resp, payload)) return "JSON parse error";
  const char* content = resp["choices"][0]["message"]["content"];
  return content ? stripMarkdown(String(content)) : "No content";
}

String callAnthropic(const String& prompt, const String& endpoint, const String& model) {
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  DynamicJsonDocument doc(1200);
  doc["model"]      = model.length() ? model : "claude-haiku-4-5";
  doc["max_tokens"] = 300;
  doc["system"]     = SYSTEM_PROMPT;
  JsonArray  msgs = doc.createNestedArray("messages");
  JsonObject msg  = msgs.createNestedObject();
  msg["role"]     = "user";
  msg["content"]  = prompt;
  String body; serializeJson(doc, body);

  http.setTimeout(API_TIMEOUT_MS);
  http.begin(client, endpoint);
  http.addHeader("Content-Type",      "application/json");
  http.addHeader("x-api-key",         storedApiKey);
  http.addHeader("anthropic-version", "2023-06-01");
  int code = http.POST(body);
  if (code != 200) { http.end(); return "Error: HTTP " + String(code); }
  String payload = http.getString(); http.end();

  DynamicJsonDocument resp(4096);
  if (deserializeJson(resp, payload)) return "JSON parse error";
  const char* content = resp["content"][0]["text"];
  return content ? stripMarkdown(String(content)) : "No content";
}

String callGemini(const String& prompt, const String& baseEndpoint, const String& model) {
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  String url = baseEndpoint;
  String mdl = model.length() ? model : "gemini-2.0-flash";
  if (!url.endsWith(":generateContent")) {
    if (url.endsWith("/")) url = url.substring(0, url.length() - 1);
    if (url.indexOf("models/") == -1) {
      url = "https://generativelanguage.googleapis.com/v1beta/models/" + mdl + ":generateContent";
    } else {
      url += ":generateContent";
    }
  }
  url += "?key=" + storedApiKey;

  DynamicJsonDocument doc(1200);
  JsonObject sysInstr = doc.createNestedObject("systemInstruction");
  JsonArray  sysParts = sysInstr.createNestedArray("parts");
  JsonObject sysPart  = sysParts.createNestedObject();
  sysPart["text"]     = SYSTEM_PROMPT;
  JsonArray  contents = doc.createNestedArray("contents");
  JsonObject turn     = contents.createNestedObject();
  turn["role"]        = "user";
  JsonArray  parts    = turn.createNestedArray("parts");
  JsonObject part     = parts.createNestedObject();
  part["text"]        = prompt;
  JsonObject genCfg   = doc.createNestedObject("generationConfig");
  genCfg["maxOutputTokens"] = 300;
  String body; serializeJson(doc, body);

  http.setTimeout(API_TIMEOUT_MS);
  http.begin(client, url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code != 200) { http.end(); return "Error: HTTP " + String(code); }
  String payload = http.getString(); http.end();

  DynamicJsonDocument resp(4096);
  if (deserializeJson(resp, payload)) return "JSON parse error";
  const char* content = resp["candidates"][0]["content"]["parts"][0]["text"];
  return content ? stripMarkdown(String(content)) : "No content";
}

String callCohere(const String& prompt, const String& endpoint, const String& model) {
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  DynamicJsonDocument doc(1200);
  doc["model"]      = model.length() ? model : "command-r-08-2024";
  doc["max_tokens"] = 300;
  JsonArray  msgs = doc.createNestedArray("messages");
  JsonObject sys  = msgs.createNestedObject();
  sys["role"]     = "system";
  sys["content"]  = SYSTEM_PROMPT;
  JsonObject msg  = msgs.createNestedObject();
  msg["role"]     = "user";
  msg["content"]  = prompt;
  String body; serializeJson(doc, body);

  int code = doPost(http, client, endpoint, body, "Authorization", "Bearer " + storedApiKey);
  if (code != 200) { http.end(); return "Error: HTTP " + String(code); }
  String payload = http.getString(); http.end();

  DynamicJsonDocument resp(4096);
  if (deserializeJson(resp, payload)) return "JSON parse error";
  const char* content = resp["message"]["content"][0]["text"];
  return content ? stripMarkdown(String(content)) : "No content";
}

String callChat(const String& prompt) {
  String endpoint = storedEndpoint.length() ? storedEndpoint : String(DEFAULT_ENDPOINT);
  String model    = storedModel.length()    ? storedModel    : String(DEFAULT_MODEL);

  if (endpoint.indexOf("anthropic.com") != -1)  return callAnthropic(prompt, endpoint, model);
  if (endpoint.indexOf("googleapis.com") != -1 ||
      endpoint.indexOf("generativelanguage") != -1) return callGemini(prompt, endpoint, model);
  if (endpoint.indexOf("cohere.ai") != -1 ||
      endpoint.indexOf("cohere.com") != -1)     return callCohere(prompt, endpoint, model);
  return callOpenAICompat(prompt, endpoint, model);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Voice flow
// ═══════════════════════════════════════════════════════════════════════════════
void beginRecording() {
  if (!startRecording()) { showMessage("Storage error", "Check partition scheme"); return; }
  currentMode = MODE_RECORDING;
  needRedraw  = true;
}

void finishRecordingAndTranscribe() {
  stopRecording();

  if (recSamples < (size_t)(SAMPLE_RATE / 2)) { freeRecBuffer(); showMessage("Too short", "Speak longer"); return; }
  if (recMaxPeak < MIN_PEAK) { freeRecBuffer(); showMessage("Too quiet", "Check mic wiring"); return; }

  currentMode = MODE_TRANSCRIBING;
  drawTranscribingMode();

  if (!ensureWiFi()) { freeRecBuffer(); showMessage("WiFi lost", "Check router"); return; }

  String text, err;
  bool ok = transcribeAudio(text, err);
  freeRecBuffer();

  if (!ok)                { showMessage("Speech error", err); return; }
  if (text.length() == 0) { showMessage("No speech heard", "Try again"); return; }

  text.toCharArray(promptBuf, MAX_PROMPT_LEN + 1);
  promptLen       = strlen(promptBuf);
  promptLineCount = wrapInto(String(promptBuf), WRAP_CHARS, promptLines, MAX_PROMPT_LINES);
  currentMode     = MODE_PROMPT;
  needRedraw      = true;
  resetButtons();
}

void askAI() {
  if (promptLen == 0) return;
  currentMode = MODE_LOADING;
  drawLoadingMode();

  if (!ensureWiFi()) { showMessage("WiFi lost", "Check router"); return; }

  String response = callChat(String(promptBuf));
  if (response.length() > 2048) response = response.substring(0, 2048);
  totalLines  = wrapInto(response, WRAP_CHARS, ansLines, MAX_ANS_LINES);
  scrollLine  = 0;

  promptLen = 0; promptBuf[0] = '\0';
  acSuggestion[0] = '\0';

  currentMode = MODE_ANSWER;
  needRedraw  = true;
  resetButtons();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Setup
// ═══════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  pinMode(BTN_MIC,  INPUT_PULLUP);
  pinMode(BTN_SEND, INPUT_PULLUP);
  pinMode(JOY_SW,   INPUT_PULLUP);

  fsReady = LittleFS.begin(true);   // true = format if the partition is unformatted
  if (!fsReady) Serial.println("LittleFS mount failed, voice recording will not work");

  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(JOY_VRX_CHANNEL, ADC_ATTEN_DB_11);  // full 0-3.3V range
  adc1_config_channel_atten(JOY_VRY_CHANNEL, ADC_ATTEN_DB_11);

  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) { while (1) delay(1000); }
  display.setTextWrap(false);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Booting...");
  display.display();
  delay(300);

  bool forceSetup = (digitalRead(BTN_MIC) == LOW && digitalRead(BTN_SEND) == LOW);

  prefs.begin("voiceai", true);
  storedSSID     = prefs.getString("ssid",     "");
  storedPass     = prefs.getString("pass",     "");
  storedApiKey   = prefs.getString("apikey",   "");
  storedEndpoint = prefs.getString("endpoint", "");
  storedModel    = prefs.getString("model",    "");
  storedSttKey   = prefs.getString("sttkey",   "");
  storedSttUrl   = prefs.getString("stturl",   "");
  storedSttModel = prefs.getString("sttmodel", "");
  prefs.end();

  bool hasCreds = storedSSID.length() && storedPass.length() && storedApiKey.length();
  if (hasCreds && !forceSetup && connectWiFi()) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi Connected!");
    display.println(WiFi.localIP().toString());
    display.display();
    delay(1200);
    currentMode = MODE_IDLE;
  } else {
    startBLE();
    currentMode = MODE_BLE;
  }
  resetButtons();
  needRedraw = true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loop
// ═══════════════════════════════════════════════════════════════════════════════
void loop() {
  bool micP   = btnPressed(bMic);
  bool sendP  = btnPressed(bSend);
  bool joySwP = btnPressed(bJoySw);

  switch (currentMode) {

    case MODE_BLE:
      if (credReceived) {
        credReceived = false;
        delay(500);
        NimBLEDevice::stopAdvertising();
        if (connectWiFi()) {
          NimBLEDevice::deinit(true);
          currentMode = MODE_IDLE;
        } else {
          NimBLEDevice::startAdvertising();
        }
        resetButtons();
        needRedraw = true;
      }
      break;

    case MODE_IDLE:
      if (joySwP) {
        currentMode = MODE_KEYBOARD;
        kbMoveTo(0, 0);
        needRedraw  = true;
        resetButtons();
      } else if (micP) {
        beginRecording();
      } else if (sendP && totalLines > 0) {
        scrollLine  = 0;
        currentMode = MODE_ANSWER;
        needRedraw  = true;
      }
      break;

    case MODE_RECORDING: {
      static unsigned long lastDraw = 0;
      if (micP || !recActive) {
        finishRecordingAndTranscribe();
      } else if (millis() - lastDraw > 100) {
        lastDraw   = millis();
        needRedraw = true;
      }
      break;
    }

    case MODE_PROMPT:
      if (micP)       beginRecording();
      else if (sendP) askAI();
      break;

    case MODE_KEYBOARD: {
      JoyStep raw = readJoystickXY();
      JoyStep js = {raw.dy, raw.dx};
      if (js.dx != 0) {
        int sz = kbRowSize(kbPage, kbRow);
        kbCol  = (kbCol + js.dx + sz) % sz;
        kbSyncCursor();
        needRedraw = true;
      }
      if (js.dy != 0) {
        int newRow = (kbRow + js.dy + KB_NUM_ROWS) % KB_NUM_ROWS;
        kbMoveTo(newRow, kbCol);
        needRedraw = true;
      }
      if (joySwP) {
        int keyId = kbGetKey(kbCursor);
        kbActivateKey(keyId);
      }
      if (micP) {
        currentMode = MODE_IDLE;
        needRedraw  = true;
        resetButtons();
      } else if (sendP || kbSendRequested) {
        kbSendRequested = false;
        askAI();
      }
      break;
    }

    case MODE_ANSWER: {
      JoyStep raw = readJoystickXY();
      JoyStep js = {raw.dy, raw.dx};
      if (js.dy != 0) {
        int maxScroll = (totalLines > VISIBLE_LINES) ? totalLines - VISIBLE_LINES : 0;
        scrollLine = constrain(scrollLine + js.dy, 0, maxScroll);
        needRedraw = true;
      }
      if (joySwP) {
        currentMode = MODE_KEYBOARD;
        kbMoveTo(0, 0);
        needRedraw  = true;
        resetButtons();
      } else if (micP) {
        beginRecording();
      } else if (sendP) {
        scrollLine += VISIBLE_LINES;
        if (scrollLine >= totalLines) scrollLine = 0;
        needRedraw = true;
      }
      break;
    }

    default:
      break;
  }

  if (needRedraw) {
    needRedraw = false;
    switch (currentMode) {
      case MODE_BLE:          drawBLEMode();          break;
      case MODE_IDLE:         drawIdleMode();         break;
      case MODE_RECORDING:    drawRecordingMode();    break;
      case MODE_TRANSCRIBING: drawTranscribingMode(); break;
      case MODE_PROMPT:       drawPromptMode();       break;
      case MODE_KEYBOARD:     drawKeyboardMode();     break;
      case MODE_LOADING:      drawLoadingMode();      break;
      case MODE_ANSWER:       drawAnswerMode();       break;
    }
  }

  delay(10);
}
