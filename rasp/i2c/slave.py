from machine import mem32
from rasp.i2c.registers import *


class I2CState:
    I2C_RECEIVE = 0
    I2C_REQUEST = 1
    I2C_FINISH = 2
    I2C_START = 3


class I2CTransaction:
    def __init__(self, address: int, data_byte: list):
        self.address = address
        self.data_byte = data_byte

    def reset(self):
        self.address = 0x00
        self.data_byte = []


class I2CSlave:
    I2C0_BASE = 0x40044000
    I2C1_BASE = 0x40048000
    IO_BANK0_BASE = 0x40014000

    # Atomic Register Access
    mem_rw = 0x0000  # Normal read/write access
    mem_xor = 0x1000  # XOR on write
    mem_set = 0x2000  # Bitmask set on write
    mem_clr = 0x3000  # Bitmask clear on write

    def __init__(self, i2c_id=0, sda=0, scl=1, slave_address=0x44):
        self.scl = scl
        self.sda = sda
        self.slave_address = slave_address
        self.i2c_id = i2c_id
        if self.i2c_id == 0:
            self.i2c_base = self.I2C0_BASE
        else:
            self.i2c_base = self.I2C1_BASE

        # 1. Disable the DW_apb_i2c by writing a ‘0’ to IC_ENABLE.ENABLE
        self.clear_32b(I2C_OFFSET["I2C_IC_ENABLE"],
                       self.get_bits_mask("ENABLE", I2C_IC_ENABLE))

        # 2. Write to the IC_SAR register (bits 9:0) to set the slave address.
        # This is the address to which the DW_apb_i2c responds.
        self.clear_32b(I2C_OFFSET["I2C_IC_SAR"],
                       self.get_bits_mask("IC_SAR", I2C_IC_SAR))

        self.set_32b(I2C_OFFSET["I2C_IC_SAR"],
                     self.slave_address & self.get_bits_mask("IC_SAR", I2C_IC_SAR))

        # 3. Write to the IC_CON register to specify which type of addressing is supported (7-bit or 10-bit by setting bit 3).
        # Enable the DW_apb_i2c in slave-only mode by writing a ‘0’ into bit six (IC_SLAVE_DISABLE) and a ‘0’ to bit zero
        # (MASTER_MODE).

        # Disable Master mode
        self.clear_32b(I2C_OFFSET["I2C_IC_CON"],
                       self.get_bits_mask("MASTER_MODE", I2C_IC_CON))

        # Enable slave mode
        self.clear_32b(I2C_OFFSET["I2C_IC_CON"],
                       self.get_bits_mask("IC_SLAVE_DISABLE", I2C_IC_CON))

        # Enable clock strech
        self.set_32b(I2C_OFFSET["I2C_IC_CON"],
                     self.get_bits_mask("RX_FIFO_FULL_HLD_CTRL", I2C_IC_CON))

        # 4. Enable the DW_apb_i2c by writing a ‘1’ to IC_ENABLE.ENABLE.
        self.set_32b(I2C_OFFSET["I2C_IC_ENABLE"],
                     self.get_bits_mask("IC_ENABLE", I2C_IC_ENABLE))

        # Reset GPIO0 function
        mem32[self.IO_BANK0_BASE | self.mem_clr | (4 + 8 * self.sda)] = 0x1f
        # Set GPIO0 as IC0_SDA function
        mem32[self.IO_BANK0_BASE | self.mem_set | (4 + 8 * self.sda)] = 0x03

        # Reset GPIO1 function
        mem32[self.IO_BANK0_BASE | self.mem_clr | (4 + 8 * self.scl)] = 0x1f
        # Set GPIO1 as IC0_SCL function
        mem32[self.IO_BANK0_BASE | self.mem_set | (4 + 8 * self.scl)] = 3

    @staticmethod
    def get_bits_mask(bits, register):
        """ This function return the bit mask based on bit name """
        bits_to_clear = bits
        bit_mask = sum([key for key, value in register.items() if value in bits_to_clear])
        return bit_mask

    def write_32b(self, register, data, atr=0):
        """ Write RP2040 I2C 32bits register """
        # < Base Addr > | < Atomic Register Access > | < Register >
        mem32[self.i2c_base | atr | register] = data

    def set_32b(self, register, data):
        """ Set bits in RP2040 I2C 32bits register """
        # < Base Addr > | 0x2000 | < Register >
        self.write_32b(register, data, atr=self.mem_set)

    def clear_32b(self, register, data):
        """ Clear bits in RP2040 I2C 32bits register """
        # < Base Addr > | 0x3000 | < Register >
        self.write_32b(register, data, atr=self.mem_clr)

    def read_32b(self, offset):
        """ Read RP2040 I2C 32bits register """
        return mem32[self.i2c_base | offset]

    def get_32b_bits(self, offset, bit_mask):
        return mem32[self.i2c_base | offset] & bit_mask

    def handle_event(self):

        # I2C Master has abort the transactions
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_INTR_STAT"],
                              self.get_bits_mask("R_TX_ABRT", I2C_IC_INTR_STAT))):
            # Clear int
            self.read_32b(I2C_OFFSET["I2C_IC_CLR_TX_ABRT"])

            return I2CState.I2C_FINISH

        # Last byte transmitted by I2C Slave but NACK from I2C Master
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_INTR_STAT"],
                              self.get_bits_mask("R_RX_DONE", I2C_IC_INTR_STAT))):
            # Clear int
            self.read_32b(I2C_OFFSET["I2C_IC_CLR_RX_DONE"])

            return I2CState.I2C_FINISH

        # Restart condition detected
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_INTR_STAT"],
                              self.get_bits_mask("R_RESTART_DET", I2C_IC_INTR_STAT))):
            # Clear int
            self.read_32b(I2C_OFFSET["I2C_IC_CLR_RESTART_DET"])

        # Start condition detected by I2C Slave
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_INTR_STAT"],
                              self.get_bits_mask("R_START_DET", I2C_IC_INTR_STAT))):
            # Clear start detection
            self.read_32b(I2C_OFFSET["I2C_IC_CLR_START_DET"])

            return I2CState.I2C_START

        # Stop condition detected by I2C Slave
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_INTR_STAT"],
                              self.get_bits_mask("R_STOP_DET", I2C_IC_INTR_STAT))):
            # Clear stop detection
            self.read_32b(I2C_OFFSET["I2C_IC_CLR_STOP_DET"])

            return I2CState.I2C_FINISH

        # Check if RX FIFO is not empty
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_STATUS"],
                              self.get_bits_mask("RFNE", I2C_IC_STATUS))):
            return I2CState.I2C_RECEIVE

        # Check if Master is requesting data
        if (self.get_32b_bits(I2C_OFFSET["I2C_IC_INTR_STAT"],
                              self.get_bits_mask("R_RD_REQ", I2C_IC_INTR_STAT))):
            # Shall Wait until transfer is done, timing recommended 10 * fastest SCL clock period
            # for 100 Khz = (1/100E3) * 10 = 100 uS
            # for 400 Khz = (1/400E3) * 10 = 25 uS

            return I2CState.I2C_REQUEST

    def is_master_req_read(self):
        """ Return status if I2C Master is requesting a read sequence """

        # Check RD_REQ Interrupt bit (master wants to read data from the slave)
        status = self.get_32b_bits(I2C_OFFSET["I2C_IC_RAW_INTR_STAT"],
                                   self.get_bits_mask("RD_REQ", I2C_IC_RAW_INTR_STAT))

        if status:
            return True
        return False

    def slave_write_data(self, data):
        """ Write 8bits of data at destination of I2C Master """

        # Send data
        self.write_32b(I2C_OFFSET["I2C_IC_DATA_CMD"], data &
                       self.get_bits_mask("DAT", I2C_IC_DATA_CMD))

        self.read_32b(I2C_OFFSET["I2C_IC_CLR_RD_REQ"])

    def available(self):
        """ Return true if data has been received from I2C Master """

        # Get RFNE Bit (Receive FIFO Not Empty)
        return self.get_32b_bits(I2C_OFFSET["I2C_IC_STATUS"],
                                 self.get_bits_mask("RFNE", I2C_IC_STATUS))

    def read_data_received(self):
        """ Return data from I2C Master """

        return self.read_32b(I2C_OFFSET["I2C_IC_DATA_CMD"]) & self.get_bits_mask("DAT", I2C_IC_DATA_CMD)
