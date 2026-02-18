# main/CMakeLists.txt
set(SRCS
    blink_example_main.c
    configs.c
    cam.c 
    AP.c
)

# If cam.h is only for this component, keep it private:
idf_component_register(
    SRCS         ${SRCS}
    INCLUDE_DIRS .          # public headers (if any)
    PRIV_INCLUDE_DIRS .     # cam.h if it’s internal to main
    PRIV_REQUIRES  fatfs sdmmc esp32-camera esp_psram nvs_flash esp_event esp_netif esp_wifi lwip
)
