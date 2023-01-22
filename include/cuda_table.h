

#pragma once

typedef struct uuid_raw {                                /**< CUDA definition of UUID */
    unsigned char bytes[16];
} uuid_raw_t;

static const uuid_raw_t tls_guid = {
    .bytes = { 0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc }
};

static const uuid_raw_t runtime_callback_hooks_guid = {
    .bytes = { 0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74, 0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66 }
};

static const uuid_raw_t context_local_storage_interface_v0301_guid = {
    .bytes = { 0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11, 0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93 }
};

static const uuid_raw_t cudart_interface_guid = {
    .bytes = { 0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a, 0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9 }
};

static const uuid_raw_t unknown_guid = {
    .bytes = { 0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b, 0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2 }
};