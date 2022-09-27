
#define READ_UNTIL(sock, buffer, n, total, cur) \
    n = 0;                                          \
    cur = 0;                                        \
    while (cur < total) {                           \
        n = read(sock, buffer + cur, total - cur);  \
        if (n < 0) break;                           \
        cur += n;                                   \
    }