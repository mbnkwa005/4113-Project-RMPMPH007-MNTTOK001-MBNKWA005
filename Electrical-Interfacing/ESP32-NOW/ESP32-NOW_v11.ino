#include <esp_now.h>
#include <WiFi.h>

#define LED_PIN 33  // Flash LED on ESP32-CAM

typedef struct struct_message {
  char rfid[20];
  float weight;
  float temperature;
  float humidity;
  int light;
  int pressure;
} struct_message;

struct_message incomingData;

// ✅ New callback signature for ESP-NOW receive
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingDataRaw, int len) {
  memcpy(&incomingData, incomingDataRaw, sizeof(incomingData));
  Serial.println("✅ Data received via ESP-NOW:");
  Serial.printf("RFID: %s\n", incomingData.rfid);
  Serial.printf("Weight: %.2f\n", incomingData.weight);
  Serial.printf("Temp: %.2f\n", incomingData.temperature);
  Serial.printf("Humidity: %.2f\n", incomingData.humidity);
  Serial.printf("Light: %d\n", incomingData.light);
  Serial.printf("Pressure: %d\n", incomingData.pressure);

  // Flash LED to show data reception
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  WiFi.mode(WIFI_STA);
  Serial.println("ESP32-CAM Receiver Ready");

  if (esp_now_init() != ESP_OK) {
    Serial.println("❌ Error initializing ESP-NOW");
    return;
  }

  // ✅ Register receive callback with new signature
  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {
  // Nothing here
}
