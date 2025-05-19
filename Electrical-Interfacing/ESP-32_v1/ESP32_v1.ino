#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Network credentials
const char* ssid = "kmaba";
const char* password = "00000000";

// Server's IP and port
const char* serverUrl = "http://192.168.247.73:5000/api/data";

void setup() {
  Serial.begin(115200);
  
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    
    // Create a JSON document
    StaticJsonDocument<200> doc;
    doc["device_id"] = "ESP32_Dev_Module";
    doc["temperature"] = random(20, 30);  // Random test value
    doc["humidity"] = random(40, 60);     // Random test value
    doc["reading_time"] = millis();
    
    // Serialize JSON to string
    String jsonString;
    serializeJson(doc, jsonString);
    
    http.setInsecure(); // Allow HTTP (not HTTPS)

    // Start HTTP connection
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    
    // Send HTTP POST request
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(httpResponseCode);
      Serial.println(response);
    } else {
      Serial.print("Error on sending POST: ");
      Serial.println(httpResponseCode);
    }
    
    // Free resources
    http.end();
  } else {
    Serial.println("WiFi Disconnected");
  }
  
  // Wait 5 seconds before next reading
  delay(5000);
}