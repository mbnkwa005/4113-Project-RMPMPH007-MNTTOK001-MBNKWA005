#include <esp_now.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi credentials
const char* ssid = "kmaba";
const char* password = "00000000";  // Replace with actual password

// Flask server endpoint
const char* serverUrl = "http://192.168.247.73:5000/api/sensor";

// Flash LED pin (GPIO 33 on ESP32-CAM)
#define LED_PIN 33

// Struct for received data
typedef struct struct_message {
  char rfid[20];
  float weight;
  float temperature;
  float humidity;
  int light;
  int pressure;
} struct_message;

struct_message incomingData;
bool dataReceived = false;

void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  memcpy(&incomingData, data, sizeof(incomingData));
  Serial.println("✅ ESP-NOW data received:");
  Serial.printf("RFID: %s\n", incomingData.rfid);

  // Flash the LED
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);

  dataReceived = true;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  WiFi.mode(WIFI_STA);  // Important: must be in station mode for ESP-NOW
  Serial.println("✅ ESP32-CAM Ready to receive ESP-NOW data");

  if (esp_now_init() != ESP_OK) {
    Serial.println("❌ Error initializing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {
  if (dataReceived) {
    dataReceived = false;

    Serial.print("🔍 Connecting to SSID: ");
    Serial.println(ssid);

    WiFi.begin(ssid, password);
    unsigned long startAttempt = millis();

    while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 15000) {
      Serial.print(".");
      delay(500);
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n✅ WiFi connected!");
      Serial.print("📶 IP address: ");
      Serial.println(WiFi.localIP());

      // Send the data to Flask server
      HTTPClient http;
      http.begin(serverUrl);
      http.addHeader("Content-Type", "application/json");

      String payload = "{";
      payload += "\"rfid\":\"" + String(incomingData.rfid) + "\",";
      payload += "\"weight\":" + String(incomingData.weight) + ",";
      payload += "\"temperature\":" + String(incomingData.temperature) + ",";
      payload += "\"humidity\":" + String(incomingData.humidity) + ",";
      payload += "\"light\":" + String(incomingData.light) + ",";
      payload += "\"pressure\":" + String(incomingData.pressure);
      payload += "}";

      int httpResponseCode = http.POST(payload);

      if (httpResponseCode > 0) {
        Serial.printf("✅ Data sent! Server responded with code: %d\n", httpResponseCode);
        String response = http.getString();
        Serial.println("📥 Response: " + response);
      } else {
        Serial.printf("❌ Failed to send data. HTTP error: %s\n", http.errorToString(httpResponseCode).c_str());
      }

      http.end();

      // Optionally disconnect from Wi-Fi to save power
      WiFi.disconnect(true);
      WiFi.mode(WIFI_OFF);

      Serial.println("📴 WiFi disconnected\n");
    } else {
      Serial.println("\n❌ WiFi connection failed");
    }

    // Reinitialize ESP-NOW after Wi-Fi shutdown
    WiFi.mode(WIFI_STA);
    esp_now_init();
    esp_now_register_recv_cb(OnDataRecv);
  }
}
