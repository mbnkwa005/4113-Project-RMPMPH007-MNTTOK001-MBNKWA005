#include "HX711.h" // For interfacing with HX711 load cell amplifier modules

#define SCK 13 // Serial Clock pin (GPIO13) (Shared)
#define DT1 12  // Data pin for the first HX711 module (GPIO12)
#define DT2 14  // Data pin for the second HX711 module (GPIO14)

HX711 scale1; // First load cell (5 kg)
HX711 scale2; // Second load cell (20 kg)

// Calibration factors (adjust after calibration with known weight)
// Convert raw ADC readiings to grams
#define SCALE_FACTOR_1 107.9 
#define SCALE_FACTOR_2 450

// Run once when the ESP32 starts
void setup() {
  Serial.begin(115200); // Initialize serial communication at 115200 baud rate

  scale1.begin(DT1, SCK); // Initialize the first HX711 MODULE
  scale2.begin(DT2, SCK); // Initialize the second HX711 MODULE

  // Wait until both HX711 modules are ready to communicate
  // Print the status message every 500ms while waiting
  while (!scale1.is_ready() || !scale2.is_ready()) {
    Serial.println("Waiting for HX711 modules...");
    delay(500); 
  }

  // Removes the weight of the platform
  scale1.tare(); 
  scale2.tare();

  // Set the calibration factor (Convert raw readings to grams)
  scale1.set_scale(SCALE_FACTOR_1);
  scale2.set_scale(SCALE_FACTOR_2);

  // Confirmation message: Setup complete
  Serial.println("Scales tared and calibrated. Ready for measurements in grams.");
}

// Runs continuously
void loop() {
  // Read the weight
  // Take 3 redaings and return the average
  float weight1 = scale1.get_units(3);
  float weight2 = scale2.get_units(3);

  // Calculate the total weight
  float totalWeight = weight1 + weight2;

  // Serial output
  // Individual weights
  Serial.print("Weight1: "); Serial.print(weight1, 2); Serial.print(" g\t");
  Serial.print("Weight2: "); Serial.print(weight2, 2); Serial.print(" g\t");

  // Total weight
  Serial.print(" | Total Weight: ");
  Serial.print(totalWeight, 2);
  Serial.println(" g");

  // Wait 500ms before the next reading
  delay(500);
}
