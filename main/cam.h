// cam.h
#pragma once
#include "esp_err.h"
// Camera set up file 
esp_err_t init_camera(void) ;

void photo_task(void *arg) ; 
void camera_task(void *arg); 
esp_err_t mount_sdcard(void); 