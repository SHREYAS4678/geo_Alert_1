/*
 * ============================================================================
 *  GeoAlert Node Firmware — ESP32-WROOM-32
 * ============================================================================
 *  Matches the exact wiring in geoalert-direct-wiring-guide.html (v3):
 *    Soil moisture (analog)  -> G34
 *    Rainfall      (analog)  -> G35
 *    DHT22         (digital) -> G4   (4.7k pull-up to 3V3, built at the sensor)
 *    SW-420 v3     (digital) -> G5   (3-pin Robocraze module, DOUT, onboard pull-up)
 *    MPU-6050      (I2C)     -> SDA G21 / SCL G22
 *    SIM800L V2    (UART)    -> RX2 G16 (direct) / TX2 G17 (via 10k/20k divider)
 *
 *  Architecture (why it's built this way):
 *    The ML ensemble (RF+XGBoost+SVM+LSTM) and the Gemini AI agent both live
 *    in the Python backend (backend/app.py) — that's where "intelligence"
 *    happens, not on the microcontroller. This firmware's ONLY jobs are:
 *      1. Read all sensors accurately
 *      2. Get the reading to the backend over WiFi
 *      3. If WiFi/backend is unreachable AND local readings look dangerous,
 *         fire an SMS directly over the SIM800L as a hardware-level safety
 *         net — because in the field, cellular often survives when WiFi/
 *         internet doesn't. The cloud-based agent is the primary decision
 *         maker; this local fallback exists only for the "no connectivity"
 *         worst case.
 *
 *  Libraries (install via Arduino IDE Library Manager — all free):
 *    - "DHT sensor library" by Adafruit  (+ "Adafruit Unified Sensor")
 *    - "Adafruit MPU6050"                (+ "Adafruit BusIO")
 *    - "TinyGSM" by Volodymyr Shymanskyy
 *    - "ArduinoJson" by Benoit Blanchon
 * ============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define TINY_GSM_MODEM_SIM800
#include <TinyGsmClient.h>

// ==================== PIN MAP (matches the wiring guide exactly) =========
#define SOIL_PIN     34
#define RAIN_PIN     35
#define DHT_PIN      4
#define DHT_TYPE     DHT22
#define VIB_PIN      5
#define GSM_RX_PIN   16   // ESP32 RX2  <- SIM800L TXD  (direct)
#define GSM_TX_PIN   17   // ESP32 TX2  -> 10k/20k divider -> SIM800L RXD
#define GSM_BAUD     9600

// ==================== USER CONFIG — edit before flashing =================
const char* WIFI_SSID   = "YOUR_WIFI_SSID";
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";

// Your Flask backend's /api/ingest URL. During development this is your
// laptop's LAN IP; see README.md "Free Hosting" section for a public URL
// once you deploy the backend for real.
const char* SERVER_URL  = "http://192.168.1.50:5000/api/ingest";

const char* NODE_ID     = "node-01";

// SIM800L failsafe SMS recipients (used ONLY when WiFi/backend is unreachable)
const char* SMS_NUMBERS[] = {"+91XXXXXXXXXX", "+91XXXXXXXXXX"};
const int   NUM_SMS_NUMBERS = 2;

// Local "obviously dangerous" thresholds — a deliberately simple, dumb,
// reliable safety net. The REAL decision-making (ensemble + AI agent) runs
// in the cloud; these numbers only matter when the cloud can't be reached.
const float LOCAL_SOIL_CRITICAL  = 88.0;   // %
const float LOCAL_RAIN1H_CRITICAL = 14.0;  // mm in the last hour
const float LOCAL_TILT_CRITICAL  = 4.5;    // degrees
const int   LOCAL_VIB_CRITICAL   = 9;      // events in the last 10 min

const unsigned long READ_INTERVAL_MS = 30UL * 1000;   // 30s during testing;
                                                        // use deep sleep (below)
                                                        // for a real field deployment
// ==========================================================================

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_MPU6050 mpu;
HardwareSerial SerialGSM(2);
TinyGsm modem(SerialGSM);

volatile unsigned long vibEventCount = 0;
volatile unsigned long lastVibIsrMs = 0;

// Rolling "tilt" is derived from the MPU-6050's accelerometer: the angle
// between the measured gravity vector and the node's own resting vector
// recorded at boot (calibration), in degrees. A genuine slope-inclinometer
// would use this same idea with a more careful mounting calibration.
float restVecX = 0, restVecY = 0, restVecZ = 1;

// ---------------------------------------------------------------------------
void IRAM_ATTR vibISR() {
  // Simple debounce: ignore interrupts closer than 40ms together (contact
  // bounce on the SW-420's comparator output).
  unsigned long now = millis();
  if (now - lastVibIsrMs > 40) {
    vibEventCount++;
    lastVibIsrMs = now;
  }
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== GeoAlert Node booting ===");

  dht.begin();

  Wire.begin(21, 22);          // SDA=G21, SCL=G22
  if (!mpu.begin()) {
    Serial.println("WARNING: MPU-6050 not found — check wiring on G21/G22.");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    calibrateTilt();
  }

  pinMode(VIB_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(VIB_PIN), vibISR, RISING);

  SerialGSM.begin(GSM_BAUD, SERIAL_8N1, GSM_RX_PIN, GSM_TX_PIN);
  Serial.println("Initialising SIM800L V2 (RST pull-up is hardware-only, no GPIO needed)...");
  modem.restart();             // safe even if the module is already up
  Serial.println(modem.getModemInfo());

  connectWiFi();
}

// ---------------------------------------------------------------------------
void loop() {
  SensorReading r = readAllSensors();
  printReading(r);

  bool posted = false;
  if (WiFi.status() == WL_CONNECTED) {
    posted = postToBackend(r);
  } else {
    Serial.println("WiFi down — attempting reconnect...");
    connectWiFi();
    if (WiFi.status() == WL_CONNECTED) posted = postToBackend(r);
  }

  if (!posted) {
    Serial.println("Backend unreachable — running LOCAL safety-net check.");
    if (isLocallyCritical(r)) {
      sendFailsafeSms(r);
    }
  }

  vibEventCount = 0;   // reset the 10-min interrupt counter for the next window
  delay(READ_INTERVAL_MS);

  // --- For a real solar/battery deployment, replace the delay() above with:
  //   esp_sleep_enable_timer_wakeup(READ_INTERVAL_MS * 1000ULL);
  //   esp_deep_sleep_start();
  // Deep sleep resets RAM, so persistent state (like vibEventCount) would
  // need to move to the RTC_DATA_ATTR memory region. Left as delay() here
  // so the sketch is simple to test on a bench with a serial monitor.
}

// ============================================================================
// Sensor reading
// ============================================================================
struct SensorReading {
  float soilPct, rain1h, rain24h, tempC, humPct, tiltDeg;
  int vibEvents;
};

// crude rolling 24h rainfall accumulator using the analog rain sensor as a
// proxy (a real tipping-bucket gauge would replace this with pulse counting)
float rain24hAccumulator = 0;
unsigned long rain24hWindowStart = 0;

SensorReading readAllSensors() {
  SensorReading r;

  int soilRaw = analogRead(SOIL_PIN);           // 0-4095, wetter = lower on most capacitive probes
  r.soilPct = mapFloat(soilRaw, 3000, 1200, 0, 100);   // CALIBRATE these two raw endpoints for your probe
  r.soilPct = constrain(r.soilPct, 0, 100);

  int rainRaw = analogRead(RAIN_PIN);
  float rainIntensity = mapFloat(rainRaw, 4095, 1500, 0, 20);  // CALIBRATE for your board
  r.rain1h = constrain(rainIntensity, 0, 60);
  rain24hAccumulator = rain24hAccumulator * 0.98 + r.rain1h * 0.02 * 24; // leaky 24h proxy
  r.rain24h = rain24hAccumulator;

  r.tempC = dht.readTemperature();
  r.humPct = dht.readHumidity();
  if (isnan(r.tempC) || isnan(r.humPct)) {
    Serial.println("WARNING: DHT22 read failed — check G4 wiring / pull-up.");
    r.tempC = 0; r.humPct = 0;
  }

  r.tiltDeg = readTiltDegrees();
  r.vibEvents = (int)vibEventCount;

  return r;
}

float readTiltDegrees() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float nx = a.acceleration.x, ny = a.acceleration.y, nz = a.acceleration.z;
  float norm = sqrt(nx*nx + ny*ny + nz*nz);
  if (norm < 0.01) return 0;
  nx /= norm; ny /= norm; nz /= norm;
  float dot = nx*restVecX + ny*restVecY + nz*restVecZ;
  dot = constrain(dot, -1.0, 1.0);
  return acos(dot) * 180.0 / PI;
}

void calibrateTilt() {
  Serial.println("Calibrating rest orientation — keep the node still for 2s...");
  delay(2000);
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float norm = sqrt(a.acceleration.x*a.acceleration.x +
                    a.acceleration.y*a.acceleration.y +
                    a.acceleration.z*a.acceleration.z);
  if (norm > 0.01) {
    restVecX = a.acceleration.x / norm;
    restVecY = a.acceleration.y / norm;
    restVecZ = a.acceleration.z / norm;
  }
  Serial.println("Tilt calibration done — this orientation is now \"0 degrees\".");
}

float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

void printReading(const SensorReading &r) {
  Serial.printf("Soil %.1f%% | Rain1h %.1fmm Rain24h %.1fmm | Temp %.1fC Hum %.1f%% | Vib %d | Tilt %.2fdeg\n",
                r.soilPct, r.rain1h, r.rain24h, r.tempC, r.humPct, r.vibEvents, r.tiltDeg);
}

// ============================================================================
// WiFi + backend POST
// ============================================================================
void connectWiFi() {
  Serial.printf("Connecting to WiFi \"%s\"...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connect timed out.");
  }
}

bool postToBackend(const SensorReading &r) {
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(8000);

  StaticJsonDocument<384> doc;
  doc["node_id"] = NODE_ID;
  doc["soil_moisture_pct"] = r.soilPct;
  doc["rainfall_1h_mm"] = r.rain1h;
  doc["rainfall_24h_mm"] = r.rain24h;
  doc["temperature_c"] = r.tempC;
  doc["humidity_pct"] = r.humPct;
  doc["vibration_events_10min"] = r.vibEvents;
  doc["tilt_deg"] = r.tiltDeg;

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  bool ok = (code == 200);
  if (ok) {
    String resp = http.getString();
    Serial.print("Backend response: ");
    Serial.println(resp);

    // Surface the ensemble+agent's decision back to the serial monitor —
    // handy for a live viva demo without opening the dashboard.
    StaticJsonDocument<1024> respDoc;
    if (deserializeJson(respDoc, resp) == DeserializationError::Ok) {
      const char* riskName = respDoc["risk_name"] | "?";
      bool alertTriggered = respDoc["agent"]["alert_triggered"] | false;
      Serial.printf(">>> Risk: %s | Agent alert triggered: %s\n",
                    riskName, alertTriggered ? "YES" : "no");
    }
  } else {
    Serial.printf("POST failed, HTTP code: %d\n", code);
  }
  http.end();
  return ok;
}

// ============================================================================
// Local safety-net (used ONLY when the backend can't be reached)
// ============================================================================
bool isLocallyCritical(const SensorReading &r) {
  return r.soilPct >= LOCAL_SOIL_CRITICAL ||
         r.rain1h >= LOCAL_RAIN1H_CRITICAL ||
         r.tiltDeg >= LOCAL_TILT_CRITICAL ||
         r.vibEvents >= LOCAL_VIB_CRITICAL;
}

void sendFailsafeSms(const SensorReading &r) {
  char text[160];
  snprintf(text, sizeof(text),
           "GeoAlert LOCAL FAILSAFE (%s): cloud unreachable. Soil %.0f%%, "
           "rain1h %.0fmm, tilt %.1fdeg, vib %d. Check site.",
           NODE_ID, r.soilPct, r.rain1h, r.tiltDeg, r.vibEvents);

  for (int i = 0; i < NUM_SMS_NUMBERS; i++) {
    Serial.printf("Sending failsafe SMS to %s...\n", SMS_NUMBERS[i]);
    bool ok = modem.sendSMS(SMS_NUMBERS[i], String(text));
    Serial.println(ok ? "  SMS sent." : "  SMS FAILED — check SIM800L V2 power/antenna.");
  }
}
