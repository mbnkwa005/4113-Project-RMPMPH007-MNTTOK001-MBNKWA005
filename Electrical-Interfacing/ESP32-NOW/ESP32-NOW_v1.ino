#include <esp_now.h>
#include <WiFi.h>

typedef struct struct_message {
  char rfid[20];
  float weight;
  float temperature;
  float humidity;
  int light;
  int pressure;
} struct_message;

struct_message testData = {
  "RFID1234567890", 2.5, 25.6, 55.2, 300, 1013
};

uint8_t receiverMac[] = {0x08, 0xB6, 0x1F, 0x99, 0x9B, 0xFC};

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("❌ ESP-NOW init failed");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (!esp_now_is_peer_exist(receiverMac)) {
    esp_now_add_peer(&peerInfo);
  }
}

void loop() {
  esp_err_t result = esp_now_send(receiverMac, (uint8_t *)&testData, sizeof(testData));
  if (result == ESP_OK) {
    Serial.println("✅ Data sent via ESP-NOW");
  } else {
    Serial.println("❌ Failed to send data");
  }
  delay(10000);  // Send every 10 seconds
}
