#pragma once

// SG90 jaw on a PCA9685 (I2C). Fully open while speaking, closed otherwise.
// Re-initialises the PCA9685 automatically if it lost power / reset.

class Jaw {
public:
    Jaw() = default;
    ~Jaw();

    Jaw(const Jaw&) = delete;
    Jaw& operator=(const Jaw&) = delete;

    // Returns false if the I2C bus or board can't be opened.
    bool init(const char* dev = "/dev/i2c-1", int addr = 0x40);

    void open_jaw();
    void close_jaw();

private:
    // ---- tune these ----
    static constexpr int    CHANNEL   = 15;     // PCA9685 header the SG90 is plugged into
    static constexpr double OPEN_US   = 2400.0; // "complete right"; swap with CLOSED_US if direction is wrong
    static constexpr double CLOSED_US = 500.0;  // try 600 / 2300 if it buzzes at the stops
    // --------------------

    int fd_ = -1;

    bool write_bytes(const unsigned char* data, int len);
    bool read_reg(unsigned char reg, unsigned char& value);
    void write_reg(unsigned char reg, unsigned char value);

    void ensure_awake();
    void pulse(double us);
    void release();
};