
#pragma

#define LOG_COLOR_RED "\033[0;31m"
#define LOG_COLOR_YELLOW "\033[0;33m"
#define LOG_COLOR_GREEN "\033[0;32m"
#define LOG_COLOR_END "\033[0m "

#define log_info(format, ...) fprintf(stderr, LOG_COLOR_GREEN "[INFO]" LOG_COLOR_END format "\n", ## __VA_ARGS__)
#define log_warn(format, ...) fprintf(stderr, LOG_COLOR_YELLOW "[WARN]" LOG_COLOR_END format "\n", ## __VA_ARGS__)
#define log_err(format, ...) fprintf(stderr, LOG_COLOR_RED "[ERROR]" LOG_COLOR_END format "\n", ## __VA_ARGS__)
