#include "jaw.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdint>

namespace {
constexpr unsigned char REG_MODE1     = 0x00;
constexpr unsigned char REG_PRESCALE  = 0xFE;
constexpr unsigned char REG_LED0_ON_L = 0x06;
constexpr unsigned char PRESCALE_50HZ = 121;
}

Jaw::~Jaw() {
    if (fd_ >= 0) {
        close_jaw();
        usleep(300000);   // let the servo finish closing
        release();
        ::close(fd_);
        fd_ = -1;
    }
}

bool Jaw::init(const char* dev, int addr) {
    fd_ = ::open(dev, O_RDWR);
    if (fd_ < 0)
        return false;

    if (ioctl(fd_, I2C_SLAVE, addr) < 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    close_jaw();
    return true;
}

void Jaw::open_jaw() {
    if (fd_ >= 0)
        pulse(OPEN_US);
}

void Jaw::close_jaw() {
    if (fd_ >= 0)
        pulse(CLOSED_US);
}

bool Jaw::write_bytes(const unsigned char* data, int len) {
    return write(fd_, data, static_cast<size_t>(len)) == len;
}

bool Jaw::read_reg(unsigned char reg, unsigned char& value) {
    return write(fd_, &reg, 1) == 1 && read(fd_, &value, 1) == 1;
}

void Jaw::write_reg(unsigned char reg, unsigned char value) {
    const unsigned char b[2] = {reg, value};
    write_bytes(b, 2);
}

// The PCA9685 forgets its settings whenever its VCC drops, so check on every move.
void Jaw::ensure_awake() {
    unsigned char mode1 = 0, prescale = 0;

    if (!read_reg(REG_MODE1, mode1) || !read_reg(REG_PRESCALE, prescale))
        return;   // board not answering (not powered yet)

    if ((mode1 & 0x10) || prescale != PRESCALE_50HZ) {
        write_reg(REG_MODE1, 0x10);          // sleep so prescale can be set
        write_reg(REG_PRESCALE, PRESCALE_50HZ);
        write_reg(REG_MODE1, 0x20);          // wake, auto-increment
        usleep(1000);
        write_reg(REG_MODE1, 0xA0);          // restart
    }
}

void Jaw::pulse(double us) {
    ensure_awake();

    const auto off = static_cast<std::uint16_t>(us * 4096.0 / 20000.0);
    const unsigned char b[5] = {
        static_cast<unsigned char>(REG_LED0_ON_L + 4 * CHANNEL),
        0, 0,
        static_cast<unsigned char>(off & 0xFF),
        static_cast<unsigned char>(off >> 8)
    };
    write_bytes(b, 5);
}

// Stop sending pulses (no buzzing, no holding torque).
void Jaw::release() {
    const unsigned char b[5] = {
        static_cast<unsigned char>(REG_LED0_ON_L + 4 * CHANNEL),
        0, 0, 0, 0x10
    };
    write_bytes(b, 5);
}