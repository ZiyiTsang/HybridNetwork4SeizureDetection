const int analogPin = A0; 
const float Vcc = 5.0;    
const int adcMax = 1023;   
const float R1 = 10000.0;  
const float R2 = 10000.0; 
#include <Adafruit_LiquidCrystal.h>


// defines LCD
Adafruit_LiquidCrystal lcd_1(0);

void setup() {
    Serial.begin(9600);
    // lcd_1.begin(16, 2);
}
void display_voltage(double input_voltage){
    lcd_1.setCursor(0, 1);
	  lcd_1.print("Voltage= ");               // prints the voltage value in the LCD display 
    lcd_1.print(input_voltage);
}
void loop() {
    int adcValue = analogRead(analogPin);
    float V_out = (adcValue / float(adcMax)) * Vcc;
    float V_in = V_out * ((R1 + R2) / R2);
	  // display_voltage(V_in);
    Serial.println(V_in);
    delay(20); 
}
