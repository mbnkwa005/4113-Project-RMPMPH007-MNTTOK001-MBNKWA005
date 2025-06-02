#include <WiFi.h>
#include <HTTPClient.h>
#include <HX711.h>
#include <MFRC522.h>
#include <SPI.h>
#include "base64.h"

// Wi-Fi credentials
const char* ssid = "Mpho’s iPhone";
const char* password = "00000000";

// Server URL
const char* server_url = "http://172.20.10.8:5000/api/update_penguin_data";

// HX711 Pins and calibration
#define DOUT1 12
#define DOUT2 14
#define SCK   13
HX711 scale1, scale2;
const float CALIBRATION_FACTOR_1 = 254.1732;
const float CALIBRATION_FACTOR_2 = 41.6208;

// Kalman filter
double Q = 0.01, R = 0.01;
double x_est_combined_kg = 0.0, P_est_combined_kg = 1.0;

// RFID
#define RST_PIN 21
#define SS_PIN 15
MFRC522 rfid(SS_PIN, RST_PIN);

// Onboard LED
#define LED_PIN 2

// Known penguins
const char* penguins[][3] = {
  { "CFD3F2", "Penguin A", "Male" },
  { "956FBB2", "Penguin B", "Female" }
};
const int NUM_PENGUINS = sizeof(penguins) / sizeof(penguins[0]);

double getKalmanFilteredWeight(double measurement) {
  double K = P_est_combined_kg / (P_est_combined_kg + R);
  x_est_combined_kg = x_est_combined_kg + K * (measurement - x_est_combined_kg);
  P_est_combined_kg = (1 - K) * P_est_combined_kg + Q;
  return x_est_combined_kg;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Wi-Fi setup
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n Wi-Fi connected");

  // HX711 setup
  scale1.begin(DOUT1, SCK);
  scale2.begin(DOUT2, SCK);
  scale1.set_scale(CALIBRATION_FACTOR_1);
  scale2.set_scale(CALIBRATION_FACTOR_2);
  scale1.tare(10);
  scale2.tare(10);

  // RFID setup
  SPI.begin();
  rfid.PCD_Init();
  Serial.println(" RFID reader ready");
}

void loop() {
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) {
    delay(100);
    return;
  }

  // Read RFID UID
  String rfid_tag = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    rfid_tag += String(rfid.uid.uidByte[i], HEX);
  }
  rfid_tag.toUpperCase();
  Serial.println("Detected RFID: " + rfid_tag);

  // Look up penguin
  String name = "", sex = "";
  bool known = false;
  for (int i = 0; i < NUM_PENGUINS; i++) {
    if (rfid_tag == penguins[i][0]) {
      name = penguins[i][1];
      sex = penguins[i][2];
      known = true;
      break;
    }
  }

  if (!known) {
    Serial.println("Unknown tag");
    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
    delay(2000);
    return;
  }

  Serial.println("Penguin: " + name + " (" + sex + ")");
  digitalWrite(LED_PIN, HIGH); // Turn on LED

  // Weight measurement
  Serial.println("⚖ Measuring...");
  double total = 0;
  int samples = 0;
  unsigned long start = millis();
  while (millis() - start < 2000) {
    if (scale1.is_ready() && scale2.is_ready()) {
      float w1 = scale1.get_units(1);
      float w2 = scale2.get_units(1);
      float raw = 1.03 * (abs(w1) + abs(w2)) / 2.0;
      double filtered = getKalmanFilteredWeight(raw / 1000.0); // kg
      total += filtered;
      samples++;
    }
    delay(100);
  }

  double final_kg = (samples > 0) ? (total / samples) : 0.0;
  float final_weight = final_kg ;//* 1000.0; // grams
  Serial.printf("Final Weight: %.2f g\n", final_weight);

  // Capture image
  String image_base64 = "";
  HTTPClient http;
  WiFiClient client;
  http.begin(client, "http://172.20.10.10/capture");
  int httpCode = http.GET();
  if (httpCode == HTTP_CODE_OK) {
    WiFiClient* stream = http.getStreamPtr();
    int len = http.getSize();
    if (len > 0) {
      uint8_t* buf = (uint8_t*)malloc(len);
      if (buf) {
        int i = 0;
        while (http.connected() && len > 0) {
          int bytes = stream->readBytes(buf + i, len);
          if (bytes > 0) {
            i += bytes;
            len -= bytes;
          } else break;
        }
        image_base64 = base64::encode(buf, i);
        free(buf);
      }
    }
  } else {
    Serial.println("Failed to get image");
  }
  http.end();

  // Send data to server
  HTTPClient post;
  post.begin(server_url);
  post.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"rfid\":\"" + rfid_tag + "\",";
  payload += "\"name\":\"" + name + "\",";
  payload += "\"sex\":\"" + sex + "\",";
  payload += "\"weight\":" + String(final_weight, 2) + ",";
  payload += "\"image\":\"" + image_base64 + "\"}";
  int responseCode = post.POST(payload);
  String response = post.getString();
  post.end();

  Serial.printf("Server Response: %d\n", responseCode);
  Serial.println("Response: " + response);

  // Cleanup
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
  digitalWrite(LED_PIN, LOW);
  delay(10000);
}
