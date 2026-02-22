// ---------- XIAO ESP32S3 Sense pin map ----------
// SD (Sense expansion board uses SPI mode)
#define SD_CS_GPIO      21   // conflicts with user LED -> do NOT drive LED
#define SD_SCLK_GPIO     7
#define SD_MISO_GPIO     8
#define SD_MOSI_GPIO     9

// Camera (OV2640/OV5640 via DVP + SCCB)
// PWDN/RESET are not wired on Sense: set to -1
#define CAM_PWDN       -1
#define CAM_RESET      -1
#define CAM_XCLK       10
#define CAM_SIOD       40   // SCCB SDA
#define CAM_SIOC       39   // SCCB SCL
#define CAM_D7         48   // Y9
#define CAM_D6         11   // Y8
#define CAM_D5         12   // Y7
#define CAM_D4         14   // Y6
#define CAM_D3         16   // Y5
#define CAM_D2         18   // Y4
#define CAM_D1         17   // Y3
#define CAM_D0         15   // Y2
#define CAM_VSYNC      38
#define CAM_HREF       47
#define CAM_PCLK       13

// ------------------------------------------------

extern char *TAG ;


//-------------------------------------------------
// Camera setting 

#define AP_CHANNEL    6
#define AP_MAX_CONN   1                // keep it simple: one viewer
#define TCP_PORT      8080             // connect to 192.168.4.1:8080
