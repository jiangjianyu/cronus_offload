#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "rpc/rpc.h"
#include "network_macros.h"

#define MAX 80
#define PORT 8080
#define SA struct sockaddr

// Function designed for chat between client and server.

void func(int connfd)
{
    
    int n, cur;
    rpc_header_t *header;
    char *buff = (char*) malloc(512 * 1024 * 1024);
    // infinite loop for chat
    for (;;) {
        bzero(buff, MAX);
   
        // read the message from client and copy it in buffer
        READ_UNTIL(connfd, buff, n, sizeof(rpc_header_t), cur);

        if (n < sizeof(rpc_header_t)) {
            fprintf(stderr, "failed connection\n");
            break;
        }

        // print buffer which contains the client contents
        header = (rpc_header_t*) buff;

        printf("execute header %d %d %d\n", header->is_running, header->size, header->status);
        READ_UNTIL(connfd, buff + sizeof(rpc_header_t), n, header->size, cur);

        if (n < sizeof(rpc_header_t)) {
            fprintf(stderr, "failed connection\n");
            break;
        }

        if (!header->is_running) break;

        rpc_dispatch(buff + sizeof(rpc_header_t));

        write(connfd, buff, sizeof(rpc_header_t) + header->size);
    }
}
   
// Driver function
int main()
{
    int sockfd, connfd, len;
    struct sockaddr_in servaddr, cli;
   
    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));
   
    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(PORT);
   
    // Binding newly created socket to given IP and verification
    if ((bind(sockfd, (SA*)&servaddr, sizeof(servaddr))) != 0) {
        printf("socket bind failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully binded..\n");
   
    // Now server is ready to listen and verification
    if ((listen(sockfd, 5)) != 0) {
        printf("Listen failed...\n");
        exit(0);
    }
    else
        printf("Server listening..\n");
    len = sizeof(cli);
   
    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, (socklen_t*) &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server accept the client...\n");
   
    // Function for chatting between client and server
    func(connfd);
    close(connfd);
   
    // After chatting close the socket
    close(sockfd);
    fprintf(stderr, "fin\n");
}
