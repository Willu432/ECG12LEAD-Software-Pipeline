#include <Wire.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiManager.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>
#include "FS.h"
#include "SD.h"

// --- MODES OF OPERATION ---
enum OperatingMode {
  SELECT_MODE,
  AP_MODE,
  CLOUD_MODE,
  SD_CARD_MODE
};
OperatingMode currentMode = SELECT_MODE;
OperatingMode nextMode = SELECT_MODE; // New variable to safely request mode changes

// --- MQTT & CLOUD SETTINGS ---
const char* mqtt_broker = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_topic = "ecg/device_01/data";
const char* mqtt_client_id = "esp32-ecg-holter-01";
// --- LOCAL AP MODE SETTINGS ---
const char* ap_ssid = "ESP32-ECG-AP";
const char* ap_password = "password123";
const uint16_t local_server_port = 8888;

// --- NTP Time Server ---
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 25200; // WIB is UTC+7 (7 * 3600)
const int daylightOffset_sec = 0;

// --- WEB SERVERS ---
WebServer server(80);
WiFiServer localDataServer(local_server_port);
WiFiClient localDataClient;
// --- MQTT CLIENT ---
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// --- PIN DEFINITIONS (UPDATED from testSDECG.ino) ---
#define BATTERY_PIN 34
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// Pin untuk MicroSD Card (menggunakan VSPI default)
const int PIN_CS_SD = 15;

// Pin untuk Sensor EKG (menggunakan HSPI)
const int PIN_DRDYB     = 25;
const int PIN_MISO_ECG  = 12;
const int PIN_MOSI_ECG  = 13;
const int PIN_SCLK_ECG  = 14;
const int PIN_SS_MASTER = 33;
const int PIN_SS_SLAVE1 = 4;
const int PIN_SS_SLAVE2 = 2;

SPIClass hspi(HSPI); // SPI bus untuk EKG
static const int spiClk = 1000000; // 1 MHz

// --- INITIALIZE LIBRARIES ---
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
unsigned long previousMillisOLED = 0;
const long intervalOLED = 1000; // Update display every 1 second

// --- EKG DATA VARIABLES ---
int32_t lead_I, lead_II, lead_III, avr, avl, avf;
int32_t v1, v2, v3, v4, v5, v6;

// --- Framing & Binary Data Variables ---
const byte STX = 0x02; // Start of Text
const byte ETX = 0x03; // End of Text
int32_t leads_data[12]; // Array to hold all 12 leads for binary sending
uint32_t packet_counter = 0; // Packet counter for performance metrics

// --- SD Card Variables ---
const unsigned int MAX_SAMPLES = 5000; // Automatic stop count
volatile unsigned int sampleCount = 0; // Current sample count
File dataFile;
char fileName[30];
volatile bool isRecording = false;
String subjectName = "";

// --- Web Page Content ---
const char* REFRESH_PAGE_HTML = "<!DOCTYPE html><html><head><title>Switching Mode</title><meta http-equiv='refresh' content='2;url=/'></head><body><h1>Switching Mode...</h1><p>You will be redirected shortly.</p></body></html>";


// --- FUNCTION PROTOTYPES ---
void handleRoot();
void handleSelectModePage();
void handleAPCloudControlPage();
void handleSDPage();
void handleStatus();
void handleSDStart();
void handleSDStop();
void handleSDView();
void handleReset();
void handleNotFound();
String listSDFiles(fs::FS &fs);
void startSelectMode();
void startAPMode();
void startCloudMode();
void startSDCardMode();
void loopAPMode();
void loopCloudMode();
void loopSDCardMode();
void sendECGDataJson(PubSubClient &client);
void sendECGDataCSV(Stream &port);
void sendECGDataBinary(Stream &port);
void startAnimation();
void drawHeart(float scale);
float readBatteryVoltage();
int calculateBatteryPercentage(float batteryVoltage);
void updateDisplayStatus(bool forceUpdate = false);
bool syncTimeWithNTP();

// --- TIME SYNC FUNCTION ---
bool syncTimeWithNTP() {
    Serial.println("Attempting to sync time with NTP server...");
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    struct tm timeinfo;
    unsigned long startAttempt = millis();
    while (millis() - startAttempt < 60000) { 
        if (getLocalTime(&timeinfo)) {
            Serial.println("Time synchronized successfully!");
            Serial.println(&timeinfo, "%A, %B %d %Y %H:%M:%S");
            return true;
        }
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nFailed to obtain time within the timeout period.");
    return false;
}

// --- SETUP ---
void setup() {
  Serial.begin(921600);

  // --- HARDWARE INITIALIZATION ---
  pinMode(PIN_DRDYB, INPUT);
  pinMode(PIN_SS_MASTER, OUTPUT);
  pinMode(PIN_SS_SLAVE1, OUTPUT);
  pinMode(PIN_SS_SLAVE2, OUTPUT);
  deselectAllChips();
  hspi.begin(PIN_SCLK_ECG, PIN_MISO_ECG, PIN_MOSI_ECG);
  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("Display Halt!");
    for (;;);
  }

  startAnimation();

  WiFi.softAP(ap_ssid, ap_password);

  // --- SETUP A SINGLE, UNIFIED WEB SERVER ROUTING TABLE ---
  server.on("/", handleRoot); // Main router function
  // Mode selection triggers
  server.on("/ap", [](){ nextMode = AP_MODE; server.send(200, "text/html", REFRESH_PAGE_HTML); });
  server.on("/cloud", [](){ nextMode = CLOUD_MODE; server.send(200, "text/html", REFRESH_PAGE_HTML); });
  server.on("/sdcard", [](){ nextMode = SD_CARD_MODE; server.send(200, "text/html", REFRESH_PAGE_HTML); });
  // SD card specific actions
  server.on("/status", handleStatus);
  server.on("/start", handleSDStart);
  server.on("/stop", handleSDStop);
  server.on("/view", handleSDView);
  // Global action
  server.on("/reset", handleReset);
  server.onNotFound(handleNotFound);
  server.begin();

  updateDisplayStatus(true); // Force initial display update
}

// --- MAIN LOOP ---
void loop() {
  // --- State machine for switching modes safely ---
  if (nextMode != currentMode) {
    if (isRecording) { // Clean up from previous mode if necessary
      isRecording = false;
      dataFile.close();
    }
    
    currentMode = nextMode; // Commit to the new mode

    switch (currentMode) {
      case AP_MODE:       startAPMode(); break;
      case CLOUD_MODE:    startCloudMode(); break;
      case SD_CARD_MODE:  startSDCardMode(); break;
      case SELECT_MODE:   startSelectMode(); break;
    }
    // Force an immediate display update after any mode switch
    updateDisplayStatus(true);
  }

  server.handleClient(); // Handle web requests in all modes
  updateDisplayStatus();   // Call centralized display update regularly

  // --- Main operation loops for the current active mode ---
  switch (currentMode) {
    case AP_MODE:       loopAPMode(); break;
    case CLOUD_MODE:    loopCloudMode(); break;
    case SD_CARD_MODE:  loopSDCardMode(); break;
    case SELECT_MODE:   /* No per-loop action needed */ break;
  }
}

// --- CENTRAL DISPLAY FUNCTION ---
void updateDisplayStatus(bool forceUpdate) {
  unsigned long currentMillis = millis();
  if (forceUpdate || (currentMillis - previousMillisOLED >= intervalOLED)) {
    previousMillisOLED = currentMillis;
    
    float batteryVoltage = readBatteryVoltage();
    int batteryPercentage = calculateBatteryPercentage(batteryVoltage);

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);

    switch(currentMode) {
      case AP_MODE:
        display.println(F("--- AP Mode ---"));
        display.println(F("Status: Transmitting BIN"));
        display.print(F("IP: ")); display.println(WiFi.softAPIP().toString());
        break;
      case CLOUD_MODE:
        display.println(F("--- Cloud Mode ---"));
        display.println(mqttClient.connected() ? F("Status: Streaming BIN") : F("Status: Reconnecting"));
        display.print(F("IP: ")); display.println(WiFi.localIP());
        break;
      case SD_CARD_MODE:
        display.println(F("--- SD Card Mode ---"));
        if(isRecording){
          display.print(F("Rec: "));
          display.print(sampleCount);
          display.print("/");
          display.println(MAX_SAMPLES);
          display.println(fileName);
        } else {
          display.println(F("Status: Standby"));
          display.println("Waiting for input...");
        }
        break;
      case SELECT_MODE:
        { // Use braces to create a local scope for apIP
          IPAddress apIP = WiFi.softAPIP();
          display.println("--- Mode Selection ---");
          display.println("Connect to WiFi:");
          display.println(ap_ssid);
          display.println("Then go to browser:");
          display.println(apIP.toString());
        }
        break;
    }

    if (currentMode != SELECT_MODE) {
        display.setCursor(0, 48);
        display.print(F("Batt: ")); display.print(batteryPercentage);
        display.print(F("% (")); display.print(batteryVoltage, 2); display.print(F("V)"));
    }
    display.display();
  }
}


// --- UNIFIED & SIMPLIFIED WEB SERVER HANDLERS ---

void handleRoot() {
  switch (currentMode) {
    case SELECT_MODE:   handleSelectModePage(); break;
    case AP_MODE:
    case CLOUD_MODE:    handleAPCloudControlPage(); break;
    case SD_CARD_MODE:  handleSDPage(); break;
  }
}

String listSDFiles(fs::FS &fs) {
    File root = fs.open("/");
    if(!root || !root.isDirectory()){ return "<p>Failed to open directory</p>"; }
    String fileList = "";
    File file = root.openNextFile();
    while(file){
        if(!file.isDirectory()){
            String fName = String(file.name());
            fileList += "<p><a href='/view?file=" + fName + "'>" + fName + "</a> (" + String(file.size()) + " bytes)</p>";
        }
        file = root.openNextFile();
    }
    if (fileList == "") { return "<p>No files on SD Card.</p>"; }
    return fileList;
}

void handleSelectModePage() {
  String html = "<html><head><title>ECG Mode Selection</title><meta name='viewport' content='width=device-width, initial-scale=1'><style>body{font-family: Arial, sans-serif; text-align: center; margin-top: 50px;} button{width: 80%; padding: 20px; font-size: 1.5em; margin: 10px; border-radius: 10px; border: none; color: white; cursor: pointer;} .ap{background-color: #007BFF;} .cloud{background-color: #28A745;} .sd{background-color: #ffc107; color: black;}</style></head>";
  html += "<body><h1>Select Operating Mode</h1>";
  html += "<a href='/ap'><button class='ap'>Start Local AP Mode</button></a>";
  html += "<a href='/cloud'><button class='cloud'>Start Cloud Upload Mode</button></a>";
  html += "<a href='/sdcard'><button class='sd'>Start SD Card Logging</button></a>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleAPCloudControlPage() {
  String html = "<html><head><title>ECG Control</title><meta name='viewport' content='width=device-width, initial-scale=1'><style>body{font-family: Arial, sans-serif; text-align: center; margin-top: 50px;} button{width: 80%; padding: 20px; font-size: 1.5em; margin: 10px; border-radius: 10px; border: none; color: white; cursor: pointer; background-color: #DC3545;}</style></head>";
  html += "<body><h1>Device Control</h1><a href='/reset'><button>Return to Mode Selection</button></a></body></html>";
  server.send(200, "text/html", html);
}

void handleSDPage() {
  String html = R"rawliteral(
<!DOCTYPE html><html><head><title>ECG SD Logger</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
  body{font-family: Arial, sans-serif; text-align: center; margin-top: 30px;}
  .container{max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1);}
  input[type=text], input[type=submit]{width: calc(100% - 24px); padding: 12px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc;}
  input[type=submit]:disabled{background-color: #cccccc;}
  button, input[type=submit].start{background-color: #28A745;}
  button{width: 80%; padding: 20px; font-size: 1.5em; margin: 10px; border-radius: 10px; border: none; color: white; cursor: pointer;}
  .stop{background-color: #DC3545;} .reset{background-color: #6c757d;}
  .file-list{text-align:left; margin-top:20px; border:1px solid #ccc; padding:10px; border-radius:5px;}
  .progress-container{margin-top: 20px; background-color: #ddd; border-radius: 5px;}
  .progress-bar{width: 0%; height: 30px; background-color: #4CAF50; text-align: center; line-height: 30px; color: white; border-radius: 5px;}
</style>
</head><body>
<div class='container'>
  <h1>SD Card Logging</h1>)rawliteral";
  
  if (isRecording) {
    html += "<h3>File: " + String(fileName) + "</h3>";
    html += "<div id='progressContainer' class='progress-container'><div id='progressBar' class='progress-bar'>0%</div></div>";
    html += "<p id='progressStatus'>Recording...</p>";
    html += "<a href='/stop'><button class='stop'>Stop Recording</button></a>";
  } else {
    html += "<h2>Status: STANDBY</h2><form id='recordForm' action='/start' method='get'><input type='text' id='subject_name' name='subject_name' placeholder='Enter Subject Name' required><input type='submit' id='submitBtn' value='Start Recording' class='start'></form>";
  }
  
  html += "<a href='/reset'><button class='reset'>Return to Mode Selection</button></a>";
  html += "<div class='file-list'><h2>Files on SD Card:</h2>" + listSDFiles(SD) + "</div>";
  html += R"rawliteral(
</div>
<script>
  function updateProgress() {
    fetch('/status')
      .then(res => res.json())
      .then(data => {
        if (!data.is_recording) {
          clearInterval(progressInterval);
          document.getElementById('progressStatus').textContent = 'Recording Complete!';
          setTimeout(() => { window.location.reload(); }, 2000);
          return;
        }
        const progress = (data.samples / )rawliteral" + String(MAX_SAMPLES) + R"rawliteral() * 100;
        const progressBar = document.getElementById('progressBar');
        progressBar.style.width = progress + '%';
        progressBar.textContent = Math.round(progress) + '%';
        document.getElementById('progressStatus').textContent = 'Recording... (' + data.samples + ' / )rawliteral" + String(MAX_SAMPLES) + R"rawliteral( samples)';
      });
  }
  
  let progressInterval;
  if (document.getElementById('progressContainer')) {
    progressInterval = setInterval(updateProgress, 1000);
  }
</script>
</body></html>)rawliteral";

  server.send(200, "text/html", html);
}


void handleStatus() {
  String json = "{\"is_recording\":" + String(isRecording ? "true" : "false") + ", \"samples\":" + String(sampleCount) + "}";
  server.send(200, "application/json", json);
}

void handleSDStart() {
  if (currentMode != SD_CARD_MODE) return;
  if (server.hasArg("subject_name")) {
    subjectName = server.arg("subject_name");
    sprintf(fileName, "/%s.bin", subjectName.c_str());
    dataFile = SD.open(fileName, FILE_WRITE);
    if (dataFile) {
      sampleCount = 0;
      isRecording = true;
    }
  }
  server.sendHeader("Location", "/", true);
  server.send(302, "text/plain", "");
}

void handleSDStop() {
  if (currentMode != SD_CARD_MODE) return;
  if (isRecording) {
    isRecording = false;
    dataFile.close();
  }
  server.sendHeader("Location", "/", true);
  server.send(302, "text/plain", "");
}

void handleSDView() {
    if (!server.hasArg("file")) {
        server.send(400, "text/plain", "Bad Request: Missing file parameter");
        return;
    }
    String path = "/" + server.arg("file");
    File file = SD.open(path, FILE_READ);

    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/html", "");

    String header = "<html><head><title>File Content</title><style>table, th, td {border: 1px solid black; border-collapse: collapse;} th, td {padding: 5px; text-align: right;}</style></head><body><h1>Content of " + path + "</h1><a href='/'>Back to Logger</a><br><br><table>";
    header += "<thead><tr><th>Lead I</th><th>Lead II</th><th>Lead III</th><th>AVR</th><th>AVL</th><th>AVF</th><th>V1</th><th>V2</th><th>V3</th><th>V4</th><th>V5</th><th>V6</th></tr></thead><tbody>";
    server.sendContent(header);

    if (!file || file.isDirectory() || file.size() == 0) {
        server.sendContent("<tr><td colspan='12'>File not found or is empty.</td></tr>");
    } else {
        while (file.available()) {
            size_t bytesRead = file.read((uint8_t*)leads_data, sizeof(leads_data));
            if (bytesRead == sizeof(leads_data)) {
                String row = "<tr>";
                for (int i = 0; i < 12; i++) row += "<td>" + String(leads_data[i]) + "</td>";
                row += "</tr>";
                server.sendContent(row);
            } else { break; }
        }
    }
    file.close();
    server.sendContent("</tbody></table></body></html>");
    server.sendContent("");
}

void handleReset() {
  nextMode = SELECT_MODE;
  server.send(200, "text/html", REFRESH_PAGE_HTML);
}

void handleNotFound() {
  server.send(404, "text/plain", "404: Not Found");
}


// --- MODE STARTUP FUNCTIONS ---
void startSelectMode() {
  startAnimation();
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_password);
}

void startAPMode() {
  startAnimation();
  localDataServer.begin();
  setup_ECG_Master();
  setup_ECG_Slave(PIN_SS_SLAVE1);
  setup_ECG_Slave(PIN_SS_SLAVE2);
  start_ECG_Conversion_All();
}

void startCloudMode() {
  startAnimation();
  
  // Beri tahu pengguna apa yang terjadi melalui OLED
  display.clearDisplay();
  display.setCursor(0,0);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.println("--- Cloud Mode ---");
  display.println("Connecting to WiFi...");
  display.println("Portal if needed:");
  display.println(ap_ssid);
  display.display();

  WiFiManager wm;
  wm.setConfigPortalTimeout(180);
  
  // Biarkan WiFiManager yang mengatur koneksi. Jangan putuskan AP secara manual.
  if (!wm.autoConnect(ap_ssid, ap_password)) {
    Serial.println("Failed to connect and hit timeout"); 
    nextMode = SELECT_MODE; // Kembali ke mode pemilihan dengan aman
    return; // Batalkan sisa setup
  }
  
  // Jika berhasil terhubung, pastikan kita hanya dalam mode Station
  WiFi.mode(WIFI_STA);

  if (!syncTimeWithNTP()) {
      Serial.println("FATAL: Could not sync time. Rebooting in 5 seconds..."); 
      delay(5000); 
      ESP.restart();
  }
  mqttClient.setServer(mqtt_broker, mqtt_port);
  setup_ECG_Master();
  setup_ECG_Slave(PIN_SS_SLAVE1);
  setup_ECG_Slave(PIN_SS_SLAVE2);
  start_ECG_Conversion_All();
}

void startSDCardMode() {
  startAnimation();
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_password);
  delay(500);
  Serial.println("Initializing SD card for logging mode...");
  if (!SD.begin(PIN_CS_SD)) {
    Serial.println("Card Mount Failed!");
    display.clearDisplay(); display.setCursor(0,0);
    display.println("SD Card Mount Failed!"); display.display();
    delay(2000); 
    nextMode = SELECT_MODE; // Safely go back to select mode instead of restarting
    return;
  }
  Serial.println("SD card initialized.");
  strcpy(fileName, "No file selected");
  setup_ECG_Master();
  setup_ECG_Slave(PIN_SS_SLAVE1);
  setup_ECG_Slave(PIN_SS_SLAVE2);
  start_ECG_Conversion_All();
}


// --- MODE LOOP FUNCTIONS ---

void loopAPMode() {
  if (!localDataClient.connected()) {
    localDataClient = localDataServer.available();
  }
  if (digitalRead(PIN_DRDYB) == LOW) {
    lead_I = getValFromChannel(PIN_SS_MASTER, 1); lead_II = getValFromChannel(PIN_SS_MASTER, 2);
    v1 = getValFromChannel(PIN_SS_SLAVE1, 1); v2 = getValFromChannel(PIN_SS_SLAVE1, 2); v3 = getValFromChannel(PIN_SS_SLAVE1, 3);
    v4 = getValFromChannel(PIN_SS_SLAVE2, 1); v5 = getValFromChannel(PIN_SS_SLAVE2, 2); v6 = getValFromChannel(PIN_SS_SLAVE2, 3);
    lead_III = lead_II - lead_I; avr = -(lead_I + lead_II) / 2;
    avl = lead_I - (lead_II / 2); avf = lead_II - (lead_I / 2);
    if (localDataClient.connected()) sendECGDataBinary(localDataClient);
  }
}

void loopCloudMode() {
  if (!mqttClient.connected()) {
    if (!mqttClient.connect(mqtt_client_id)) { delay(2000); return; }
  }
  mqttClient.loop();
  if (digitalRead(PIN_DRDYB) == LOW) {
    lead_I = getValFromChannel(PIN_SS_MASTER, 1); lead_II = getValFromChannel(PIN_SS_MASTER, 2);
    v1 = getValFromChannel(PIN_SS_SLAVE1, 1); v2 = getValFromChannel(PIN_SS_SLAVE1, 2); v3 = getValFromChannel(PIN_SS_SLAVE1, 3);
    v4 = getValFromChannel(PIN_SS_SLAVE2, 1); v5 = getValFromChannel(PIN_SS_SLAVE2, 2); v6 = getValFromChannel(PIN_SS_SLAVE2, 3);
    lead_III = lead_II - lead_I; avr = -(lead_I + lead_II) / 2;
    avl = lead_I - (lead_II / 2); avf = lead_II - (lead_I / 2);
    leads_data[0] = lead_I;   leads_data[1] = lead_II; leads_data[2] = lead_III; leads_data[3] = avr;
    leads_data[4] = avl;      leads_data[5] = avf; leads_data[6] = v1;       leads_data[7] = v2;
    leads_data[8] = v3;       leads_data[9] = v4; leads_data[10] = v5;      leads_data[11] = v6;
    struct timeval tv; gettimeofday(&tv, NULL);
    uint64_t timestamp_us = (uint64_t)tv.tv_sec * 1000000L + (uint64_t)tv.tv_usec;
    packet_counter++;
    uint8_t payload[63];
    payload[0] = STX;
    memcpy(&payload[1], &timestamp_us, 8); memcpy(&payload[9], &packet_counter, 4); memcpy(&payload[13], leads_data, 48);
    uint8_t checksum = 0;
    for (int i = 1; i < 61; i++) checksum += payload[i];
    payload[61] = checksum; payload[62] = ETX;
    mqttClient.publish(mqtt_topic, payload, sizeof(payload));
  }
}

void loopSDCardMode() {
  if (isRecording && (digitalRead(PIN_DRDYB) == LOW)) {
    if (sampleCount < MAX_SAMPLES) {
      sampleCount++;
      lead_I = getValFromChannel(PIN_SS_MASTER, 1); lead_II = getValFromChannel(PIN_SS_MASTER, 2);
      v1 = getValFromChannel(PIN_SS_SLAVE1, 1); v2 = getValFromChannel(PIN_SS_SLAVE1, 2); v3 = getValFromChannel(PIN_SS_SLAVE1, 3);
      v4 = getValFromChannel(PIN_SS_SLAVE2, 1); v5 = getValFromChannel(PIN_SS_SLAVE2, 2); v6 = getValFromChannel(PIN_SS_SLAVE2, 3);
      lead_III = lead_II - lead_I; avr = -(lead_I + lead_II) / 2;
      avl = lead_I - (lead_II / 2); avf = lead_II - (lead_I / 2);
      leads_data[0] = lead_I;   leads_data[1] = lead_II; leads_data[2] = lead_III; leads_data[3] = avr;
      leads_data[4] = avl;      leads_data[5] = avf; leads_data[6] = v1;       leads_data[7] = v2;
      leads_data[8] = v3;       leads_data[9] = v4; leads_data[10] = v5;      leads_data[11] = v6;
      if (dataFile) {
          dataFile.write((const uint8_t*)leads_data, sizeof(leads_data));
      }
    } else {
      isRecording = false;
      dataFile.close();
      Serial.println("\nRecording finished automatically. File saved.");
    }
  }
}

// --- DATA SENDING FUNCTIONS ---

void sendECGDataJson(PubSubClient &client) {
  StaticJsonDocument<512> doc;
  doc["deviceId"] = mqtt_client_id;
  doc["lead_I"] = lead_I;   doc["lead_II"] = lead_II;
  doc["lead_III"] = lead_III; doc["avr"] = avr;
  doc["avl"] = avl;      doc["avf"] = avf;
  doc["v1"] = v1;         doc["v2"] = v2;
  doc["v3"] = v3;         doc["v4"] = v4;
  doc["v5"] = v5;         doc["v6"] = v6;
  char jsonBuffer[512];
  serializeJson(doc, jsonBuffer);
  client.publish(mqtt_topic, jsonBuffer);
}

void sendECGDataBinary(Stream &port) {
  leads_data[0] = lead_I;   leads_data[1] = lead_II; leads_data[2] = lead_III; leads_data[3] = avr;
  leads_data[4] = avl;      leads_data[5] = avf; leads_data[6] = v1;       leads_data[7] = v2;
  leads_data[8] = v3;       leads_data[9] = v4; leads_data[10] = v5;      leads_data[11] = v6;
  uint8_t checksum = 0;
  uint8_t* data_ptr = (uint8_t*)leads_data;
  for (int i = 0; i < sizeof(leads_data); i++) {
    checksum += data_ptr[i];
  }
  port.write(STX);
  port.write((uint8_t*)leads_data, sizeof(leads_data));
  port.write(checksum);
  port.write(ETX);
}


// --- HELPER & HARDWARE FUNCTIONS ---

float readBatteryVoltage() {
  int batteryLevel = analogRead(BATTERY_PIN);
  float rawVoltage = batteryLevel * (3.3 / 4095.0);
  float batteryVoltage = rawVoltage * 3.24; 
  return batteryVoltage;
}

int calculateBatteryPercentage(float batteryVoltage) {
  float minVoltage = 6.40;
  float maxVoltage = 8.40;
  int percentage = map(batteryVoltage * 100, minVoltage * 100, maxVoltage * 100, 0, 100);
  return constrain(percentage, 0, 100);
}

void drawHeart(float scale) {
  display.clearDisplay();
  int xCenter = SCREEN_WIDTH / 2;
  int yCenter = SCREEN_HEIGHT / 2 + 10;
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(5, 0);
  display.print(F("12-Lead ECG Device"));
  int radius = (int)(10 * scale);
  int bottomHeight = (int)(20 * scale);
  display.fillCircle(xCenter - radius, yCenter - radius / 2, radius, SSD1306_WHITE);
  display.fillCircle(xCenter + radius, yCenter - radius / 2, radius, SSD1306_WHITE);
  display.fillTriangle(xCenter - 2 * radius, yCenter - radius / 2,
                       xCenter + 2 * radius, yCenter - radius / 2,
                       xCenter, yCenter + bottomHeight, SSD1306_WHITE);
  display.display();
}

void startAnimation() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(22, 0);
  display.print(F("12-Lead Holter"));
  display.setCursor(20, 18);
  display.print(F("Initializing..."));
  display.setCursor(20, 50);
  display.print(F("CTAI Laboratory"));
  display.display();
  delay(2000);
  for (int i = 0; i < 2; i++) {
    for (float scale = 1.0; scale <= 1.3; scale += 0.05) { drawHeart(scale); delay(80); }
    for (float scale = 1.3; scale >= 1.0; scale -= 0.05) { drawHeart(scale); delay(80); }
  }
}

// --- ECG HARDWARE FUNCTIONS ---
void selectChip(int chip_SS) {
  digitalWrite(PIN_SS_MASTER, HIGH);
  digitalWrite(PIN_SS_SLAVE1, HIGH);
  digitalWrite(PIN_SS_SLAVE2, HIGH);
  digitalWrite(chip_SS, LOW);
}
void deselectAllChips() {
  digitalWrite(PIN_SS_MASTER, HIGH);
  digitalWrite(PIN_SS_SLAVE1, HIGH);
  digitalWrite(PIN_SS_SLAVE2, HIGH);
}
byte readRegister(int chip_SS, byte reg) {
  reg |= 0x80;
  hspi.beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE0));
  selectChip(chip_SS);
  hspi.transfer(reg);
  byte data = hspi.transfer(0);
  deselectAllChips();
  hspi.endTransaction();
  return data;
}
void writeRegister(int chip_SS, byte reg, byte data) {
  reg &= 0x7F;
  hspi.beginTransaction(SPISettings(spiClk, MSBFIRST, SPI_MODE0));
  selectChip(chip_SS);
  hspi.transfer(reg);
  hspi.transfer(data);
  deselectAllChips();
  hspi.endTransaction();
}

int32_t getValFromChannel(int chip_SS, int channel_num_on_chip) {
  byte r1, r2, r3;
  switch (channel_num_on_chip) {
    case 1: r1 = 0x37; r2 = 0x38; r3 = 0x39; break;
    case 2: r1 = 0x3A; r2 = 0x3B; r3 = 0x3C; break;
    case 3: r1 = 0x3D; r2 = 0x3E; r3 = 0x3F; break;
    default: return 0;
  }
  int32_t val = ((int32_t)readRegister(chip_SS, r1) << 16) |
                ((int32_t)readRegister(chip_SS, r2) << 8)  |
                 (int32_t)readRegister(chip_SS, r3);
  
  if (val & 0x00800000) {
    val |= 0xFF000000;
  }
  return val;
}

void setup_ECG_Master() {
  writeRegister(PIN_SS_MASTER, 0x00, 0x00);
  writeRegister(PIN_SS_MASTER, 0x01, 0x11);
  writeRegister(PIN_SS_MASTER, 0x02, 0x19);
  writeRegister(PIN_SS_MASTER, 0x0A, 0x07);
  writeRegister(PIN_SS_MASTER, 0x0C, 0x04);
  writeRegister(PIN_SS_MASTER, 0x0D, 0x01);
  writeRegister(PIN_SS_MASTER, 0x0E, 0x02);
  writeRegister(PIN_SS_MASTER, 0x0F, 0x03);
  writeRegister(PIN_SS_MASTER, 0x12, 0x01); delay(1);
  writeRegister(PIN_SS_MASTER, 0x12, 0x05);
  writeRegister(PIN_SS_MASTER, 0x14, 0x24);
  writeRegister(PIN_SS_MASTER, 0x21, 0x02);
  writeRegister(PIN_SS_MASTER, 0x22, 0x02);
  writeRegister(PIN_SS_MASTER, 0x23, 0x02);
  writeRegister(PIN_SS_MASTER, 0x27, 0x08);
  writeRegister(PIN_SS_MASTER, 0x28, 0x08);
  writeRegister(PIN_SS_MASTER, 0x2F, 0x30);
}
void setup_ECG_Slave(int chip_SS) {
  writeRegister(chip_SS, 0x00, 0x00);
  writeRegister(chip_SS, 0x01, 0x0C);
  writeRegister(chip_SS, 0x02, 0x14);
  writeRegister(chip_SS, 0x03, 0x1C);
  writeRegister(chip_SS, 0x12, 0x02); delay(1);
  writeRegister(chip_SS, 0x12, 0x06);
  writeRegister(chip_SS, 0x21, 0x02);
  writeRegister(chip_SS, 0x22, 0x02);
  writeRegister(chip_SS, 0x23, 0x02);
  writeRegister(chip_SS, 0x24, 0x02);
  writeRegister(chip_SS, 0x27, 0x00);
  writeRegister(chip_SS, 0x28, 0x40);
  writeRegister(chip_SS, 0x2F, 0x70);
}
void start_ECG_Conversion_All() {
  writeRegister(PIN_SS_MASTER, 0x00, 0x01);
  writeRegister(PIN_SS_SLAVE1, 0x00, 0x01);
  writeRegister(PIN_SS_SLAVE2, 0x00, 0x01);
}

