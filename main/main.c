// main.c — XIAO ESP32-S3 Sense: capture 1 JPEG/sec to SD (ESP-IDF)
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_psram.h"

#include "driver/gpio.h"
#include "driver/spi_common.h"
#include "macros.h"
#include "esp_camera.h"
#include "cam.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "esp_vfs_fat.h"

// add near your includes:
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include <sys/stat.h>   // <-- for mkdir
#include <sys/unistd.h>

#include "AP.h"


QueueHandle_t s_frame_q = NULL;
void app_main(void)
{

    s_frame_q = xQueueCreate(3, sizeof(camera_fb_t *));
    configASSERT(s_frame_q);
    ESP_LOGI(TAG, "XIAO ESP32S3 Sense — 1fps capture to SD");

    // Camera first (uses PSRAM)
    ESP_ERROR_CHECK(init_camera());
    sensor_t *s = esp_camera_sensor_get();
    // optional: keep auto controls off while stabilizing
    s->set_exposure_ctrl(s, 1);
    //s->set_gain_ctrl(s, 0);
    //s->set_awb_gain(s, 0);
    // SD next (LED pin gets released inside)
    // ESP_ERROR_CHECK(mount_sdcard());
    // Might reset the bloody SD CARD stuff since it is not needed 
     wifi_ap_start();

    ESP_LOGI(TAG, "Camera task on core %d", xPortGetCoreID());
    vTaskDelay(pdMS_TO_TICKS(500));       // let Wi-Fi settle

    // warm-up: grab & drop a few frames so sensor PLL locks
    for (int i = 0; i < 5; ++i) {
        camera_fb_t *w = esp_camera_fb_get();
        if (w) esp_camera_fb_return(w);
        vTaskDelay(pdMS_TO_TICKS(60));
    }
    // Start capture loop
    //xTaskCreatePinnedToCore(photo_task, "photo_task", 4096, NULL, 5, NULL, 1);  // Core 1 task 

    // Camera task on core 1
    xTaskCreatePinnedToCore(camera_task, "camera_task",
                            8192, NULL, tskIDLE_PRIORITY + 5, NULL, 1);

    xTaskCreatePinnedToCore(ap_task, "ap_task",
                            6144, NULL, tskIDLE_PRIORITY + 4, NULL, 0);


}
