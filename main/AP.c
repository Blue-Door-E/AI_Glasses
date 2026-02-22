// code for the access point for connection
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
// AP.c
#include <string.h>
#include <unistd.h>            // close()

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_err.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "esp_event.h"         // esp_event_loop_create_default()
#include "esp_netif.h"         // esp_netif_init(), esp_netif_create_default_wifi_ap()
#include "esp_wifi.h"          // wifi_* types, esp_wifi_* APIs
#include "configs.h"
#include "lwip/sockets.h"      // socket(), bind(), listen(), accept(), send()
                               // (alternatively you can include <sys/socket.h>, <netinet/in.h>)

// Will want to just make it boot up 
extern QueueHandle_t s_frame_q;
void wifi_ap_start(void) {
    // 1) Init NVS with recovery
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(err);
    }

    // 2) Netif + default loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // 3) Create default AP netif (sets up LwIP, DHCP server, etc.)
    ESP_ERROR_CHECK(esp_netif_create_default_wifi_ap() ? ESP_OK : ESP_FAIL);

    // 4) Wi-Fi driver
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_country_t country = { .cc="US", .schan=1, .nchan=11, .policy=WIFI_COUNTRY_POLICY_MANUAL };
    esp_wifi_set_country(&country);
    esp_wifi_set_ps(WIFI_PS_NONE);
    // 5) Configure AP
    wifi_config_t ap_cfg = {
        .ap = {
            .ssid = AP_SSID,
            .ssid_len = 0,
            .channel = 6,
            .password = AP_PASS,
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = { .required = true },
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
}
// ---------- SoftAP + TCP streaming in the same task ----------
void ap_task(void *arg) {
    ESP_LOGI(TAG, "AP task on core %d", xPortGetCoreID());

    // Wi-Fi/AP is already up (wifi_ap_start() ran).
    // Optional: short wait to ensure DHCP is up
    vTaskDelay(pdMS_TO_TICKS(200));

    ESP_LOGI(TAG, "SoftAP ready: connect to 192.168.4.1:%d", TCP_PORT);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);

    for (;;) {
        ESP_LOGI(TAG, "Waiting for client...");
        int client = accept(srv, NULL, NULL);
        if (client < 0) continue;

        ESP_LOGI(TAG, "Client connected");
        while (1) {
            camera_fb_t *fb = NULL;
            if (xQueueReceive(s_frame_q, &fb, pdMS_TO_TICKS(2000)) == pdTRUE && fb) {
                uint32_t len_be = htonl(fb->len);
                if (send(client, &len_be, sizeof(len_be), 0) != sizeof(len_be)) { esp_camera_fb_return(fb); break; }
                if (send(client, fb->buf, fb->len, 0) != (int)fb->len)        { esp_camera_fb_return(fb); break; }
                esp_camera_fb_return(fb);
            } else {
                // no frame; loop
            }
        }
        close(client);
    }
}