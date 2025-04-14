print("THIS COSE SHOULD BE RUN IN RASPBERRY PICO")

import machine
import utime
from machine import Pin
from slave import I2CSlave, I2CTransaction, I2CState


I2C_ID = 0
SDA_PIN = 4
SCL_PIN = 5
I2C_ADDRESS = 0x44

led_13 = Pin(13, Pin.OUT)
led_13.on()

led_25 = Pin(25, Pin.OUT)
led_12 = Pin(12, Pin.OUT)
led_10 = Pin(10, Pin.OUT)
led_9 = Pin(9, Pin.OUT)
led_6 = Pin(6, Pin.OUT)


# establish I2C slave ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
print("starting I2C slave…")
s_i2c = I2CSlave(I2C_ID, sda=SDA_PIN, scl=SCL_PIN, slave_address=I2C_ADDRESS)


currentTransaction = I2CTransaction(0x00, [])
state = I2CState.I2C_START


# indicate startup…
for i in range(3):
    led_25.toggle()
    utime.sleep_ms(50)
    led_25.toggle()
    utime.sleep_ms(50)
    led_25.toggle()
utime.sleep_ms(333)


while True:
    state = s_i2c.handle_event()
    if state == I2CState.I2C_START:
        led_6.toggle()
    elif state == I2CState.I2C_RECEIVE:
        if currentTransaction.address == 0x00:
            _register_address = s_i2c.read_data_received()
            currentTransaction.address = _register_address

        _valid = False
        index = 0
        _expected_length = 0

        while s_i2c.available():
            _data_rx = s_i2c.read_data_received()
            _int_value = int(_data_rx)
            if _data_rx == 0x00:
                pass
            elif _data_rx == 0x01:
                _valid = True
            elif _data_rx == 0xFF:
                break
            else:
                if _data_rx == 0x02:
                    led_12.on()
                    led_10.off()
                    led_9.off()
                elif _data_rx == 0x03:
                    led_12.off()
                    led_10.on()
                    led_9.off()
                elif _data_rx == 0x04:
                    led_12.off()
                    led_10.off()
                    led_9.on()

            index = index + 1
            currentTransaction.data_byte.append(_data_rx)
    elif state == I2CState.I2C_REQUEST:
        while s_i2c.is_master_req_read():
            led_25.toggle()
            s_i2c.slave_write_data(0x46)

# EOF
