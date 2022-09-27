

#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#include "rpc/crypto.h"
#include "rpc/rpc.h"
#include "network_macros.h"

static char *buffer;
static rpc_header_t *header;

int buffer_size_in_bytes;

extern void *mmap(void *start, size_t len, int prot, int flags, int fd, long off);

int rpc_open(void* uuid, int buffer_size_in_mb) {
	
    buffer_size_in_bytes = buffer_size_in_mb * 1024 * 1024;

    buffer = mmap(0, buffer_size_in_bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    if (buffer == (void*)-1) {
        fprintf(stderr, "out of memory in rpc\n");
        exit(0);
        return 1;
    }

    header = (rpc_header_t*) buffer;
    header->is_running = IS_RUNNING_START;

    fprintf(stderr, "open session succeed\n");

    return 0;
}

extern int sockfd;

int rpc_ecall(uint32_t idx, void *ecall_buf, int bufsize) {
    int total_size = (bufsize + sizeof(uint32_t));
    int ret = 0, r, cur;

    uint32_t *idx_ptr = (uint32_t*)(buffer + sizeof(rpc_header_t));
    *idx_ptr = idx;

    header->size = total_size;
    header->status = STATUS_START;

    r = write(sockfd, buffer, total_size + sizeof(rpc_header_t));
    READ_UNTIL(sockfd, buffer, r, sizeof(rpc_header_t), cur);

    if (r <= 0) {
        fprintf(stderr, "error in reading headers\n");
        return -1;
    }

    READ_UNTIL(sockfd, buffer + sizeof(rpc_header_t), r, header->size, cur);

    return ret;
}

void rpc_close() {
    // do unmap
    header->is_running = IS_RUNNING_STOP;
    header->size = 0;
    write(sockfd, buffer, sizeof(rpc_header_t));
    close(sockfd);
}

char* rpc_buffer() {
    return buffer + sizeof(rpc_header_t) + sizeof(uint32_t);
}

int rpc_handle(void* buffer) {

}
