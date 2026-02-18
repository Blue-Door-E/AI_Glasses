#include <esp_err.h>
#include <esp_camera.h>
#include "macros.h"
#include "esp_psram.h"
#include "esp_err.h"
#include <driver/spi_common.h>
#include <sd_protocol_types.h>
#include <driver/sdspi_host.h>
#include <esp_vfs_fat.h>
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_psram.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_vfs_fat.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "sdmmc_cmd.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdbool.h>
#include <freertos/queue.h>

char *TAG = "xiao_sense_cam";

extern QueueHandle_t s_frame_q;
void test () { 
    // Do nothing just declared to compile maybe and chain 
}
static inline bool has_jpeg_eoi(const uint8_t *buf, size_t len) {
    return len >= 2 && buf[len - 2] == 0xFF && buf[len - 1] == 0xD9;
}

esp_err_t init_camera(void) {
    camera_config_t config = {
        .pin_pwdn       = CAM_PWDN,
        .pin_reset      = CAM_RESET,
        .pin_xclk       = CAM_XCLK,
        .pin_sccb_sda   = CAM_SIOD,
        .pin_sccb_scl   = CAM_SIOC,
        .pin_d7         = CAM_D7, .pin_d6 = CAM_D6, .pin_d5 = CAM_D5, .pin_d4 = CAM_D4,
        .pin_d3         = CAM_D3, .pin_d2 = CAM_D2, .pin_d1 = CAM_D1, .pin_d0 = CAM_D0,
        .pin_vsync      = CAM_VSYNC, .pin_href = CAM_HREF, .pin_pclk = CAM_PCLK,

        .xclk_freq_hz   = 10000000,           // keep conservative at first
        .ledc_timer     = LEDC_TIMER_0,
        .ledc_channel   = LEDC_CHANNEL_0,
        .pixel_format   = PIXFORMAT_JPEG,
        .frame_size     = FRAMESIZE_QVGA,     // step up later (VGA/SVGA/UXGA)
        .jpeg_quality   = 11,                 // modest compression
        .fb_count       = 2,                  // extra headroom for SD writes
        .fb_location    = CAMERA_FB_IN_PSRAM, // <-- move FBs to PSRAM
        .grab_mode      = CAMERA_GRAB_LATEST  // drop stale frames if writer lags // CAMERA_GRAB_WHEN_EMPTY 
    };

    // Optional safety: verify PSRAM is up before init
    
    if (!esp_psram_is_initialized()) {
        ESP_LOGE(TAG, "PSRAM not initialized");
        return ESP_FAIL;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) ESP_LOGE(TAG, "Camera init failed: %s", esp_err_to_name(err));
    return err;
}


void photo_task(void *arg) {
    int n = 0;
    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb || !fb->buf || fb->len == 0) {
            if (fb) esp_camera_fb_return(fb);
            ESP_LOGW(TAG, "Empty frame dropped");
            vTaskDelay(pdMS_TO_TICKS(100)); // brief backoff
            continue;
        }
        if (!has_jpeg_eoi(fb->buf, fb->len)) {
            esp_camera_fb_return(fb);
            ESP_LOGW(TAG, "Truncated JPEG (no EOI) dropped (len=%u)", (unsigned)fb->len);
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        struct timeval tv; gettimeofday(&tv, NULL);
        long long ms = (long long)tv.tv_sec * 1000LL + tv.tv_usec / 1000;
        char path[64];
        snprintf(path, sizeof(path), "/sdcard/pics/%lld.jpg", ms);

        FILE *f = fopen(path, "wb");
        if (!f) {
            ESP_LOGE(TAG, "Open file failed: %s", path);
        } else {
            size_t wrote = fwrite(fb->buf, 1, fb->len, f);
            fclose(f);
            if (++n % 10 == 0) {
                ESP_LOGI(TAG, "Saved %s (%u/%u bytes)", path, (unsigned)wrote, (unsigned)fb->len);
            }
        }
        esp_camera_fb_return(fb);

        vTaskDelay(pdMS_TO_TICKS(1000)); // 1 fps
    }
}
void camera_task(void *arg) {
    vTaskDelay(pdMS_TO_TICKS(600));
    for (int i = 0; i < 5; ++i) {
        camera_fb_t *w = esp_camera_fb_get();
        if (w) esp_camera_fb_return(w);
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    for (;;) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Capture failed");
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        // Send to queue (drop if full to avoid blocking capture cadence)
        if (xQueueSend(s_frame_q, &fb, 0) != pdTRUE) {
            // Nobody waiting; return the buffer
            esp_camera_fb_return(fb);
        }
        vTaskDelay(pdMS_TO_TICKS(200)); // ~1 fps; tune as needed
    }
}




esp_err_t mount_sdcard(void)
{
    // Release LED/CS pin so SD can use it
    gpio_reset_pin(SD_CS_GPIO);
    gpio_set_direction(SD_CS_GPIO, GPIO_MODE_INPUT);

    // 1) Init the SPI bus (needed to set SCLK/MISO/MOSI pins)
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = SD_MOSI_GPIO,
        .miso_io_num = SD_MISO_GPIO,
        .sclk_io_num = SD_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO));

    // 2) Host config for SDSPI (5-arg API needs this)
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;  // use SPI2/HSPI

    // 3) Slot/device config (chip-select + host binding)
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SD_CS_GPIO;
    slot_config.host_id = host.slot;

    // 4) Mount options
    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024,
    };

    // 5) Do the mount (NOTE: 5 parameters)
    sdmmc_card_t *card = NULL;
    esp_err_t err = esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_config, &mount_cfg, &card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(err));
        return err;
    }

    // Optional: print card info
    sdmmc_card_print_info(stdout, card);

    // Make pictures dir (2nd arg is ignored on FAT but required by API)
    mkdir("/sdcard/pics", 0775);

    return ESP_OK;
}
