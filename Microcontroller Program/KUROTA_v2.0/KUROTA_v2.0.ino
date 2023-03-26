//#include <Adafruit_GPS.h>
#include <MPU6050_tockn.h>
#include <String.h>
#include <Wire.h>
#include "MAX30100_PulseOximeter.h"

//#define GPSSerial Serial1
#define gprsSerial Serial2
//Adafruit_GPS GPS(&GPSSerial);

TwoWire WIRE2 (2,I2C_FAST_MODE);
MPU6050 mpu6050(WIRE2);

char x, n;
int baterai, terguling, bantuan, latitudeSent, longitudeSent, bpmSent, o2Sent, bateraiSent, tergulingSent, bantuanSent;
long timerLed, timerReport, timerBuzzer, a[8], tim, timerTunda;
float bpm, o2, angleX, angleY, gyroZ, voltage, latitude, longitude;
bool ledState, buzzerState, tundaState0=0;
String arah;
char http_cmd[80];
char url_string_latitude[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field1";
char url_string_bpm[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field2";
char url_string_o2[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field3";
char url_string_baterai[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field4";
char url_string_terguling[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field6";
char url_string_bantuan[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field5";
char url_string_longitude[] = "GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field7";
char nilai_string[50];

PulseOximeter pox;

//// Callback (registered below) fired when a pulse is detected
//void onBeatDetected()
//{
//  beat=1;    
//}

void setup() {
  pinMode(PB1, OUTPUT);
  pinMode(PB0, OUTPUT);
  pinMode(PA7, OUTPUT);
  pinMode(PA4, OUTPUT);
  pinMode(PA5, OUTPUT);
  pinMode(PA6, OUTPUT);
  pinMode(PA8, OUTPUT);
  pinMode(PB8, OUTPUT);
  
  digitalWrite(PB1, LOW);
  digitalWrite(PB0, LOW);
  analogWrite(PA7, 0);
  digitalWrite(PA4, LOW);
  digitalWrite(PA5, LOW);
  analogWrite(PA6, 0);
  digitalWrite(PA8, HIGH);
  digitalWrite(PB8, HIGH);

  TIMER1_BASE->PSC = 7200;
  TIMER1_BASE->ARR = 8000;
  TIMER1_BASE->CNT = 0;
  timer_attach_interrupt(TIMER1, 0, motorAktif);
  TIMER1_BASE->CR1 |= 0x0001;
  
  Serial.begin(115200);
  gprsSerial.begin(9600);               // the GPRS baud rate   
  delay(1000);
//  if (gprsSerial.available()){
//    Serial.write(gprsSerial.read());}
  gprsSerial.println("AT");
  delay(1000);
  gprsSerial.println("AT+CPIN?");
  delay(1000);
  gprsSerial.println("AT+CREG?");
  delay(1000);
  gprsSerial.println("AT+CGATT?");
  delay(1000);
  
  WIRE2.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);
//  GPS.begin(57600);
//  GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
//  GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);
//  GPS.sendCommand(PGCMD_ANTENNA);
//  delay(1000);
//  GPSSerial.println(PMTK_Q_RELEASE);
  if (!pox.begin()) {
        Serial.println();
        Serial.println("FAILED");
        for(;;);
    } else {
        Serial.println();
        Serial.println("SUCCESS");
    }
}

void loop() {
//Serial.print("start: ");Serial.println(millis());  
//  terimaDataRaspberry();
  akuisisiOlahDataMPU();
//  cekTombolBantuan();
//  cekTombolDriverMotor();
  buzzer();
//  akuisisiDataGPS();
  pox.update();
//  akuisisiOlahDataTegangan();
  kirimDataInternet();
  if(millis() - timerReport > 1000){
    serialMonitor();
    timerReport = millis();}
  
//Serial.print("end: ");Serial.println(millis());
}

void terimaDataRaspberry(){  
  while (Serial.available()) {
    for (n=0; n<4; n++){
      a[n] = Serial.read();
    }
    x=a[0];
    Serial.println(x);
    if(digitalRead(PB3)==1){
      x='1';}}
}

void akuisisiOlahDataMPU(){
  mpu6050.update();
  angleX= abs(mpu6050.getAngleX());
  angleY= abs(mpu6050.getAngleY());
  gyroZ= abs(mpu6050.getGyroZ());
  if(angleX >= 30.0 || angleY >= 45.0 || gyroZ >= 400){
    terguling= 1;}
}

void cekTombolBantuan(){
  if(digitalRead(PA15)==0){
    bantuan= 1;
  } else{
    bantuan= 0;
  }
}

void cekTombolDriverMotor(){
  if(digitalRead(PB3)==1){
    x='1';
    digitalWrite(PB1, LOW);
    digitalWrite(PB0, LOW);
    analogWrite(PA7, 0);
    digitalWrite(PA4, LOW);
    digitalWrite(PA5, LOW);
    analogWrite(PA6, 0);
  }
}

void buzzer(){
  if(digitalRead(PB4)==0){
    terguling=0;
    digitalWrite(PA8, HIGH); //buzzer
  }
  if(terguling==1){
    x='1';
    digitalWrite(PB1, LOW);
    digitalWrite(PB0, LOW);
    analogWrite(PA7, 0);
    digitalWrite(PA4, LOW);
    digitalWrite(PA5, LOW);
    analogWrite(PA6, 0);
    if(millis() - timerBuzzer >= 500){
      digitalWrite(PA8, buzzerState);
      buzzerState=!buzzerState;
      timerBuzzer = millis();}    
  }
}

//void akuisisiDataGPS(){
//  char c = GPS.read();
//  if (GPS.newNMEAreceived()) {
//    if (!GPS.parse(GPS.lastNMEA()))
//    return;}
//}

void akuisisiOlahDataTegangan(){  
  voltage= (float)analogRead(PA0)*0.000805664063*5;
//  voltage= (float)analogRead(PA0)*0.0002017125*5;  
  if(voltage<8.5){
     baterai=1;
     if(millis() - timerLed >= 100){
     digitalWrite(PB8, ledState);
     ledState=!ledState;
     timerLed = millis();}}
  else if(voltage<9.5){
        baterai=2;
        if(millis() - timerLed >= 500){
        digitalWrite(PB8, ledState);
        ledState=!ledState;
        timerLed = millis();}}
  else if(voltage<10.0){
        baterai=3;
        digitalWrite(PB8, LOW);}  
  else {
    baterai=4;
    digitalWrite(PB8, HIGH);}
}

void kirimDataInternet(){
//  if (bantuanSent==0){
//    bantuanSent= gprsKirim(bantuan, url_string_bantuan);
//  }
  if (tergulingSent==0){
    tergulingSent= gprsKirim(terguling, url_string_terguling);
  }
  if(tergulingSent==1 && bpmSent==0){
    if(bpm==0.0 || o2==0.0){  //undetected
      bpm= 0;}    
    bpmSent= gprsKirim(bpm, url_string_bpm);
  }
  if (bpmSent==1 && o2Sent==0){
    if(bpm==0.0 || o2==0.0){  //undetected
      o2= 0;}
    o2Sent= gprsKirim(o2, url_string_o2);
  }
//  if (o2Sent==1 && latitudeSent==0){
//    latitudeSent= gprsKirim(latitude, url_string_latitude);
//  }
//  if (latitudeSent==1 && longitudeSent==0){
//    longitudeSent= gprsKirim(longitude, url_string_longitude);
//  }
//  if (longitudeSent==1 && bateraiSent==0){
//    bateraiSent= gprsKirim(baterai, url_string_baterai);
//  }
  if (o2Sent==1){
    latitudeSent=0; longitudeSent=0; bpmSent=0; o2Sent=0; bateraiSent=0;
    tergulingSent=0; bantuanSent=0;
    Serial.println("Berhasil Kirim Data ke Internet");
  }    
}

void serialMonitor(){  
  bpm= pox.getHeartRate();
  o2= pox.getSpO2();
//  latitude= GPS.latitude*100;
//  longitude= GPS.longitude*100;
  Serial.print("angleX : ");Serial.println(mpu6050.getAngleX());
  Serial.print("angleY : ");Serial.println(mpu6050.getAngleY());
  Serial.print("angleZ : ");Serial.println(mpu6050.getAngleZ());
  Serial.print("GyroZ : ");Serial.println(mpu6050.getGyroZ());
  Serial.println("-------------------------------------------");
  Serial.print("Heart rate:");Serial.print(bpm);
  Serial.print("bpm / SpO2:");Serial.print(o2);Serial.println("%");    
  Serial.println("-------------------------------------------");
//  Serial.print("Fix: "); Serial.print((int)GPS.fix);
//  Serial.print(" quality: "); Serial.println((int)GPS.fixquality);
//  if (GPS.fix) {
//    Serial.print("Location: ");
//    Serial.print(GPS.latitude, 4); Serial.print(GPS.lat);
//    Serial.print(", ");
//    Serial.print(GPS.longitude, 4); Serial.println(GPS.lon);
//    Serial.print("Speed (knots): "); Serial.println(GPS.speed);
//    Serial.print("Angle: "); Serial.println(GPS.angle);
//    Serial.print("Altitude: "); Serial.println(GPS.altitude);
//    Serial.print("Satellites: "); Serial.println((int)GPS.satellites);
//  }
//  Serial.println("-------------------------------------------");    
//  if(digitalRead(PB3)==1){
//    Serial.println("Motor Dimatikan");
//  } else{
//    Serial.println("Status Kursi Roda : ");
//    Serial.println(arah);}  
//  Serial.println("-------------------------------------------");    
  if(terguling==1){
    Serial.println("Bahaya!!! Kursi Roda Terguling");
    if(digitalRead(PB4)==1){
      Serial.println("Alarm Menyala");
    } else{Serial.println("PB Stop Alarm Ditekan");}
  }
//  if(bantuan==1){
//    Serial.println("Tombol Bantuan Aktif");
//  }
//  Serial.print("Tegangan Baterai : ");Serial.println(voltage);
//  if(voltage<10.0){
//    Serial.println("Lowbatt!! Segera charge baterai");
//  }
  Serial.println("-------------------------------------------");
  Serial.println();
  Serial.println();   
}

void motorAktif(void){      
    if(x=='3'){
      digitalWrite(PB1, HIGH);
      digitalWrite(PB0, LOW);
      analogWrite(PA7, 255);
      digitalWrite(PA4, HIGH);
      digitalWrite(PA5, LOW);
      analogWrite(PA6, 255);
      x='1';
      arah="Maju";
    } else if(x=='4'){
      digitalWrite(PB1, LOW);
      digitalWrite(PB0, HIGH);
      analogWrite(PA7, 255);
      digitalWrite(PA4, LOW);
      digitalWrite(PA5, HIGH);
      analogWrite(PA6, 255);
      x='1';
      arah="Mundur";
    } else if(x=='2'){
      digitalWrite(PB1, LOW);
      digitalWrite(PB0, HIGH);
      analogWrite(PA7, 128);
      digitalWrite(PA4, HIGH);
      digitalWrite(PA5, LOW);
      analogWrite(PA6, 128);
      x='1';
      arah="Kiri";
    } else if(x=='0'){
      digitalWrite(PB1, HIGH);
      digitalWrite(PB0, LOW);
      analogWrite(PA7, 128);
      digitalWrite(PA4, LOW);
      digitalWrite(PA5, HIGH);
      analogWrite(PA6, 128);
      x='1';
      arah="Kanan";
    } else{
      x='1';
      digitalWrite(PB1, LOW);
      digitalWrite(PB0, LOW);
      analogWrite(PA7, 0);
      digitalWrite(PA4, LOW);
      digitalWrite(PA5, LOW);
      analogWrite(PA6, 0);
      arah="Diam";
    }
}

int gprsKirim(int nilai, char url_string[]){
  static bool tundaState1=0, tundaState2=0, tundaState3=0;
  static bool shutState=0, statusState=0, muxState=0, csttState=0, ciicrState=0;
  static bool cifsrState=0, sprtState=0, startState=0, sendState=0, cmdState=0;
  static bool c26State=0, shut2State=0;
  static int fullState=0;
  fullState=0;
  
  if(millis() - timerTunda > 100){
    tundaState0=1;}

  if (tundaState0==1 && shutState==0){
    gprsSerial.println("AT+CIPSHUT");
    timerTunda = millis();
    shutState=1;
    tundaState1=0;}
  
  if(millis() - timerTunda > 1000){
    tundaState1=1;}
  
  if (shutState==1 && tundaState1==1 && statusState==0){
    gprsSerial.println("AT+CIPSTATUS");
    timerTunda = millis();
    statusState=1;
    tundaState1=0;}
  
  if (statusState==1 && tundaState1==1 && muxState==0){    
    gprsSerial.println("AT+CIPMUX=0");
    timerTunda = millis();
    muxState=1;
    tundaState1=0;}
  
  ShowSerialData();
 
  if (muxState==1 && tundaState1==1 && csttState==0){
    gprsSerial.println("AT+CSTT=\"myAPN\"");//start task and setting the APN,
    timerTunda = millis();
    csttState=1;
    tundaState1=0;}

  ShowSerialData();
 
  if (csttState==1 && tundaState1==1 && ciicrState==0){
    gprsSerial.println("AT+CIICR");//bring up wireless connection
    timerTunda = millis();
    ciicrState=1;
    tundaState1=0;}
 
  ShowSerialData();

  if (ciicrState==1 && tundaState1==1 && cifsrState==0){ 
    gprsSerial.println("AT+CIFSR");//get local IP adress
    timerTunda = millis();
    cifsrState=1;
    tundaState1=0;}
 
  ShowSerialData();

  if (cifsrState==1 && tundaState1==1 && sprtState==0){ 
    gprsSerial.println("AT+CIPSPRT=0");
    timerTunda = millis();
    sprtState=1;
    tundaState1=0;}
 
  ShowSerialData();

  if (cifsrState==1 && tundaState1==1 && startState==0){  
    gprsSerial.println("AT+CIPSTART=\"TCP\",\"api.thingspeak.com\",\"80\"");//start up the connection
    timerTunda = millis();
    startState=1;
    tundaState3=0;}
 
  ShowSerialData();

  if(millis() - timerTunda > 3000){
    tundaState3=1;}

  if (startState==1 && tundaState3==1 && sendState==0){ 
    gprsSerial.println("AT+CIPSEND");//begin send data to remote server
    timerTunda = millis();
    sendState=1;
    tundaState1=0;
//    atm_pressure1=atm_pressure1+5;
    dtostrf(nilai, 5, 0, nilai_string);
    sprintf(http_cmd,"%s=%s",url_string,nilai_string);}
    
  ShowSerialData();

  if (sendState==1 && tundaState1==1 && cmdState==0){  
  //  String str="GET https://api.thingspeak.com/update?api_key=FKWNOIPIPV8KY5DY&field1" + String(temp);
    gprsSerial.println(http_cmd);//begin send data to remote server
    timerTunda = millis();
    cmdState=1;
    tundaState2=0;}

  ShowSerialData();

  if(millis() - timerTunda > 2000){
    tundaState2=1;}

  if (cmdState==1 && tundaState2==1 && c26State==0){
    gprsSerial.println((char)26);//sending
    timerTunda = millis();
    c26State=1;
    tundaState2=0;}

  if (c26State==1 && tundaState2==1 && shut2State==0){  
    gprsSerial.println();
    ShowSerialData();
    gprsSerial.println("AT+CIPSHUT");//close the connection
    timerTunda = millis();
    tundaState0=0;
    tundaState1=0; tundaState2=0; tundaState3=0; shutState=0; statusState=0;
    muxState=0; csttState=0; ciicrState=0; cifsrState=0; sprtState=0;
    startState=0; sendState=0; cmdState=0; c26State=0; shut2State=0;
    fullState=1;}
 
  ShowSerialData();
  return fullState;
}

void ShowSerialData()
{
  while(gprsSerial.available()!=0)
    Serial.write(gprsSerial.read());
}
