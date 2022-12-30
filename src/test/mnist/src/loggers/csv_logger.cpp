#include "csv_logger.hpp"

#include <iostream>

CSVLogger::CSVLogger(std::string fileName) {

    this->epoch = 0;
    std::cout << "epoch,"
                  << "trainingLoss,"
                  << "trainingAccuracy,"
                  << "testLoss,"
                  << "testAccuracy,"
                  << "totalForwardTime,"
                  << "totalBackwardTime,"
                  << "batchForwardTime,"
                  << "batchBackwardTime\n";
}

CSVLogger::~CSVLogger() {

}

void CSVLogger::logEpoch(double trainingLoss, double trainingAccuracy,
                         double testLoss, double testAccuracy,
                         double totalForwardTime, double totalBackwardTime,
                         double batchForwardTime, double batchBackwardTime) {
    std::cout << this->epoch++ << ","
                  << trainingLoss << ","
                  << trainingAccuracy << ","
                  << testLoss << ","
                  << testAccuracy << ","
                  << totalForwardTime << ","
                  << totalBackwardTime << ","
                  << batchForwardTime << ","
                  << batchBackwardTime << "\n";
}

